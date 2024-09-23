import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata, SamplingMetadataCache
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalInputs)
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available, make_tensor_with_pad, PyObjectCache
# from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    InterDataForSeqGroup,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class ModelInputForNeuron(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    input_block_ids: Optional[torch.Tensor] = None
    attn_metadata: Optional["NeuronAttentionMetadata"] = None
    sampling_metadata: Optional["SamplingMetadata"] = None
    # multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    request_ids_to_seq_ids: Optional[Dict[str, List[int]]] = None
    finished_requests_ids: Optional[List[str]] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "attn_metadata": self.attn_metadata,
            "request_ids_to_seq_ids": self.request_ids_to_seq_ids,
            "finished_requests_ids": self.finished_requests_ids,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForNeuron":
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls.from_broadcasted_tensor_dict(tensor_dict)


class ModelInputForNeuronBuilder(ModelRunnerInputBuilderBase[ModelInputForNeuron]):
    """Build ModelInputForNeuron from SequenceGroupMetadata."""

    def gen_inter_data_builder(self, num_seqs: int):
        return lambda: InterDataForSeqGroup(
            request_id="",
            seq_ids=[0] * num_seqs,
            is_prompt=True,
            block_tables=None,
            computed_block_nums=[])

    def init_cached_inter_data(self, *args, **kwargs):
        assert len(args) == 0
        assert "seq_ids" in kwargs
        seq_ids = kwargs["seq_ids"]
        num_seqs = len(seq_ids)

        # The inter-data cache is per model_runner
        inter_data_cache = self.runner.inter_data_cache
        if num_seqs not in inter_data_cache:
            inter_data_cache[num_seqs] = PyObjectCache(
                self.gen_inter_data_builder(num_seqs))

        obj = inter_data_cache[num_seqs].get_object()
        obj.__init__(*args, **kwargs)
        return obj

    def reset_cached_inter_data(self):
        for cache in self.runner.inter_data_cache.values():
            cache.reset()

    def __init__(self,
                 runner: "NeuronModelRunnerBase",
                 finished_requests_ids: Optional[List[str]] = None):
        super().__init__()
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
        ]

        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.scheduler_config = self.runner.scheduler_config
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.finished_requests_ids = finished_requests_ids
        self.decode_only = True

        # Intermediate data (data in CPU before going to GPU) for
        # the current sequence group.
        self.inter_data_list: List[
            ModelInputForNeuronBuilder.InterDataForSeqGroup] = []

        # Attention metadata inputs.
        self.attn_metadata_builder = self.attn_backend.make_metadata_builder(
            weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = (
            self.scheduler_config is not None
            and self.scheduler_config.chunked_prefill_enabled)

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int,
                      seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).
        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
        else:
            # get_num_computed_tokens is incorrect for spec decoding.
            # So, we should have a special logic here.
            # TODO(sang): Fix it.
            context_len = seq_len - 1
        seq_len = min(seq_len, context_len + token_chunk_size)

        # Compute tokens.
        if inter_data.is_prompt:
            tokens = seq_data.get_token_ids()
            if context_len != 0 or seq_len < len(tokens):
                tokens = tokens[context_len:seq_len]
        else:
            # Optimization. get_token_ids requires the entire copy of
            # tokens.
            tokens = seq_data.get_last_token_id()

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len

        if isinstance(tokens, list):
            inter_data.input_tokens[seq_idx].extend(tokens)
        else:
            inter_data.input_tokens[seq_idx].append(tokens)

        if (seq_len - context_len) == 1:
            inter_data.input_positions[seq_idx].append(seq_len - 1)
        else:
            inter_data.input_positions[seq_idx].extend(
                range(context_len, seq_len))

        inter_data.query_lens[
            seq_idx] = seq_len - context_len if inter_data.is_prompt else 1

    def _compute_for_prefix_cache_hit(
            self, inter_data: InterDataForSeqGroup, seq_idx: int,
            seq_group_metadata: SequenceGroupMetadata):
        """Check if hit prefix cache (i.e., some blocks are already computed).
        If hit, update input tokens and positions to only compute the
        remaining blocks.
        """
        computed_block_nums = inter_data.computed_block_nums

        # Note that prefix caching does not support sliding window.
        prefix_cache_hit = (computed_block_nums is not None
                            and len(computed_block_nums) > 0
                            and self.sliding_window is None
                            and inter_data.is_prompt)
        inter_data.prefix_cache_hit = prefix_cache_hit

        if not prefix_cache_hit:
            return

        assert computed_block_nums is not None
        # The cache hit prompt tokens in this sequence. Note that
        # this may be larger than the sequence length if chunked
        # prefill is enabled.
        prefix_cache_len = len(computed_block_nums) * self.block_size
        # The number of so far computed prompt tokens in this sequence.
        context_len = inter_data.context_lens[seq_idx]
        # The total number of prompt tokens in this sequence.
        # When chunked prefill is enabled, this is the token number of
        # computed chunks + current chunk.
        seq_len = inter_data.seq_lens[seq_idx]
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass
        elif context_len < prefix_cache_len < seq_len:
            # Partial hit. Compute the missing part.
            uncomputed_start = prefix_cache_len - context_len
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][uncomputed_start:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][uncomputed_start:]
            context_len = prefix_cache_len

            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len
        elif seq_len <= prefix_cache_len:
            # Full hit. Only compute the last token to avoid
            # erroneous behavior. FIXME: Ideally we should directly
            # mark all tokens as computed in the scheduler and do not
            # schedule this sequence, so this case should not happen.
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][-1:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][-1:]
            inter_data.query_lens[seq_idx] = 1
            inter_data.context_lens[seq_idx] = inter_data.seq_lens[seq_idx] - 1

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = []
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)

        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        input_positions = []
        for inter_data in self.inter_data_list:
            for cur_input_positions in inter_data.input_positions:
                input_positions.extend(cur_input_positions)

        seq_lens = []
        max_decode_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
        query_lens = []
        for inter_data in self.inter_data_list:
            query_lens.extend(inter_data.query_lens)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        batch_size = len(input_tokens)

        # Tokens and positions.
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory)
        input_positions_tensor = async_tensor_h2d(input_positions, torch.long,
                                                  self.runner.device,
                                                  self.runner.pin_memory)

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = seq_group_metadata.seq_data.keys()
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        inter_data = self.init_cached_inter_data(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums,
            reinit=True,
            reinit_use_defaults=True)

        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)
        # for per_seq_group_fn in self.per_seq_group_compute_fns:
        #     per_seq_group_fn(inter_data, seq_group_metadata)

    def build(self) -> ModelInputForNeuron:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = []
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)

        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        input_positions = []
        for inter_data in self.inter_data_list:
            for cur_input_positions in inter_data.input_positions:
                input_positions.extend(cur_input_positions)

        seq_lens = []
        max_decode_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
        query_lens = []
        for inter_data in self.inter_data_list:
            query_lens.extend(inter_data.query_lens)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        batch_size = len(input_tokens)
        # use_captured_graph = self._use_captured_graph(batch_size,
        #                                               max_decode_seq_len)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        # cuda_graph_pad_size = -1
        # if use_captured_graph:
        #     graph_batch_size = _get_graph_batch_size(batch_size)
        #     assert graph_batch_size >= batch_size
        #     cuda_graph_pad_size = graph_batch_size - batch_size
        #     batch_size = graph_batch_size

        # Tokens and positions.
        # if cuda_graph_pad_size:
        #     input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
        #     input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
        # assert self.runner.device is not None
        input_tokens_tensor = torch.tensor(input_tokens, device=self.runner.device)
        # async_tensor_h2d(input_tokens, torch.long,
        #                                        self.runner.device,
        #                                        self.runner.pin_memory)
        input_positions_tensor = torch.tensor(input_positions, device=self.runner.device)
        # async_tensor_h2d(input_positions, torch.long,
        #                                           self.runner.device,
        #                                           self.runner.pin_memory)

        # Sequence and query lengths.
        # if cuda_graph_pad_size:
        #     seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, batch_size)

        # LoRA data.
        # lora_requests = set()
        # lora_mapping = None
        # if self.enable_lora:
        #     lora_requests = set(r for data in self.inter_data_list
        #                         for r in data.lora_requests)
        #     lora_index_mapping = flatten_2d_lists([
        #         flatten_2d_lists(inter_data.lora_index_mapping)
        #         for inter_data in self.inter_data_list
        #     ])
        #     if cuda_graph_pad_size:
        #         lora_index_mapping.extend(
        #             itertools.repeat(0, cuda_graph_pad_size))
        #     lora_prompt_mapping = flatten_2d_lists([
        #         flatten_2d_lists(inter_data.lora_prompt_mapping)
        #         for inter_data in self.inter_data_list
        #     ])

        #     lora_mapping = LoRAMapping(
        #         **dict(index_mapping=lora_index_mapping,
        #                prompt_mapping=lora_prompt_mapping,
        #                is_prefill=not self.decode_only))

        # Prompt adapter data.
        # prompt_adapter_requests: Set[PromptAdapterRequest] = set()
        # prompt_adapter_mapping = None
        # if self.enable_prompt_adapter:
        #     prompt_adapter_requests = set(
        #         data.prompt_adapter_request for data in self.inter_data_list
        #         if data.prompt_adapter_request is not None)
        #     prompt_adapter_index_mapping = flatten_2d_lists([
        #         inter_data.prompt_adapter_index_mapping
        #         for inter_data in self.inter_data_list
        #     ])
        #     if cuda_graph_pad_size:
        #         prompt_adapter_index_mapping.extend(
        #             itertools.repeat(0, cuda_graph_pad_size))
        #     prompt_adapter_prompt_mapping = flatten_2d_lists([
        #         inter_data.prompt_adapter_prompt_mapping
        #         for inter_data in self.inter_data_list
        #     ])
        #     prompt_adapter_mapping = PromptAdapterMapping(
        #         prompt_adapter_index_mapping,
        #         prompt_adapter_prompt_mapping,
        #     )

        # Multi-modal data.
        # multi_modal_inputs_list = [
        #     data.multi_modal_inputs for data in self.inter_data_list
        #     if data.multi_modal_inputs is not None
        # ]
        # multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            # lora_mapping=lora_mapping,
            # lora_requests=lora_requests,
            # multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids,
            # prompt_adapter_mapping=prompt_adapter_mapping,
            # prompt_adapter_requests=prompt_adapter_requests,
        )


class NeuronModelRunner(ModelRunnerBase[ModelInputForNeuron]):
    # _model_input_cls: Type[ModelInputForNeuronWithSamplingMetadata] = (
    #     ModelInputForNeuronWithSamplingMetadata)
    _model_input_cls: Type[ModelInputForNeuron] = ModelInputForNeuron
    _builder_cls: Type[ModelInputForNeuronBuilder] = ModelInputForNeuronBuilder

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config

        self.sliding_window = None
        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = "bfloat16"
        self.block_size = cache_config.block_size

        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.attn_backend = get_attn_backend(
            num_heads=num_attn_heads,
            head_size=self.model_config.get_head_size(),
            num_kv_heads=self.model_config.get_num_kv_heads(self.parallel_config),
            sliding_window=self.model_config.get_sliding_window(),
            dtype=self.model_config.dtype if model_config is not None else None,
            kv_cache_dtype=self.kv_cache_dtype,
            block_size=self.block_size,
            )

        # Multi-modal data support
        self.multi_modal_input_mapper = MULTIMODAL_REGISTRY \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.
        self.block_size: int  # Set after initial profiling.

        # Used to cache python objects
        self.inter_data_cache: Dict[int, PyObjectCache] = {}
        self.sampling_metadata_cache: SamplingMetadataCache = \
            SamplingMetadataCache()

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config,
                                      cache_config=self.cache_config)

    def compile_model(self) -> None:
        self.model.model.to_neuron()

        # import os
        # name_or_path = self.model.config._name_or_path
        # ps_dir = os.path.join("/home/ubuntu/workspace/models", f"{name_or_path}-presharded")
        # self.model.model.preshared_dir = ps_dir
        # # on_device_embedding must be True to save and load presharded weights
        # self.model.model.save(ps_dir, sharded_weights=True)

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForNeuron:
        return ModelInputForNeuron.from_broadcasted_tensor_dict(tensor_dict)

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.
        """
        builder = self._builder_cls(weakref.proxy(self), finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        builder.reset_cached_inter_data()

        return builder.build()  # type: ignore

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForNeuron:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        # if is_prompt:
        #     (input_tokens, input_positions, input_block_ids, seq_lens,
        #      ) = self._prepare_prompt(seq_group_metadata_list)
        # else:
        #     (input_tokens, input_positions,
        #      input_block_ids) = self._prepare_decode(seq_group_metadata_list)
        #     seq_lens = []
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list, model_input.seq_lens,
            model_input.query_lens, self.device, self.pin_memory,
            self.get_generators(finished_requests_ids))
        return ModelInputForNeuron(input_tokens=model_input.input_tokens,
                                   input_positions=model_input.input_positions,
                                   input_block_ids=model_input.input_block_ids,
                                   attn_metadata=model_input.attn_metadata,
                                   sampling_metadata=sampling_metadata)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForNeuron,
        kv_caches: Optional[List[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "NeuronModelRunner does not support multi-step execution.")
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            input_metadata=model_input.attn_metadata,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)
        # print("model_input.sampling_metadata", model_input.sampling_metadata)
        # Sample the next token.
        # Before sampling we only keep tokens that are to be sampled (to check if logic is actually correct or not)
        seqs_to_sample = []
        for idx, seq_group in enumerate(model_input.sampling_metadata.seq_groups):
            if seq_group.sample_indices:
                seqs_to_sample += [idx]
        logits = logits[seqs_to_sample, :]
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        # print("output token: ", output)
        return [output]

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
