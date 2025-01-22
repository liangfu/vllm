import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed
import torch.nn as nn
import torch_xla.core.xla_model as xm

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalDataDict
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, cdiv, is_pin_memory_available)
from vllm.v1.attention.backends.neuron_attn import (NeuronAttentionBackend,
                                               NeuronAttentionMetadata, BlockDiagonalCausalFromBottomRightMask)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.model_runner_base import ModelRunnerBase
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import torch_xla.runtime as xr
from unittest.mock import patch
from vllm.config import set_current_vllm_config

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)

# Here we utilize the behavior that out-of-bound index is ignored.
# FIXME: Find a more reliable way to prevent possible bugs.
_PAD_SLOT_ID = 1_000_000_000
# build neuron program
B_P_SIZE = 128
LARGE_TILE_SZ = 2048


class NeuronModelRunner(ModelRunnerBase):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        # TODO: use 
        ModelRunnerBase.__init__(self, vllm_config=vllm_config, device=device)
        # self.vllm_config = vllm_config
        # self.model_config = vllm_config.model_config
        # self.cache_config = vllm_config.cache_config
        # self.lora_config = vllm_config.lora_config
        # self.load_config = vllm_config.load_config
        # self.parallel_config = vllm_config.parallel_config
        # self.scheduler_config = vllm_config.scheduler_config
        # self.device_config = vllm_config.device_config
        # self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        # self.observability_config = vllm_config.observability_config

        # model_config = self.model_config
        # cache_config = self.cache_config
        # scheduler_config = self.scheduler_config
        # parallel_config = self.parallel_config
        # self.device = self.device_config.device
        # self.pin_memory = is_pin_memory_available()
        # self.dtype = self.model_config.dtype
        # if cache_config.cache_dtype == "auto":
        #     self.kv_cache_dtype = self.dtype
        # else:
        #     self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
        #         cache_config.cache_dtype]

        # self.sliding_window = model_config.get_sliding_window()
        # self.block_size = cache_config.block_size
        # self.max_model_len = model_config.max_model_len
        # self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        # self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # # Model-related.
        # # HACK AOYU use latest implementation
        # # self.num_attn_layers = model_config.get_num_attention_layers(
        # #     parallel_config)
        # self.num_attn_layers = model_config.get_num_layers_by_block_type(
        #     parallel_config, LayerBlockType.attention)
        # self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        # self.head_size = model_config.get_head_size()

        # # List[k_cache, v_cache]
        # self.kv_caches: List[Tuple[torch.Tensor, torch.Tensor]] = []

        # # Request states.
        # self.requests: Dict[str, CachedRequestState] = {}
        # # Persistent batch.
        # self.input_batch = InputBatch(
        #     max_num_reqs=self.scheduler_config.max_num_seqs,
        #     max_model_len=self.max_model_len,
        #     max_num_blocks_per_req=self.max_num_blocks_per_req,
        #     device=self.device,
        #     pin_memory=self.pin_memory,
        #     vocab_size=model_config.get_vocab_size(),
        # )

        # self.prefill_positions = torch.tensor(
        #     range(self.max_model_len),
        #     device="cpu",
        # ).to(torch.int32).reshape(1, -1)

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)

        # Remove the requests from the persistent batch.
        stopped_req_ids = set().union(
            scheduler_output.preempted_req_ids,
            scheduler_output.finished_req_ids,
        )
        removed_req_indices: List[int] = []
        for req_id in stopped_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Update the states of the running requests.
        for req_data in scheduler_output.scheduled_running_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]

            # Update the num_computed_tokens.
            req_state.num_computed_tokens = req_data.num_computed_tokens
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                req_data.num_computed_tokens)

            # Update the block table.
            num_new_blocks = len(req_data.new_block_ids)
            if num_new_blocks == 0:
                continue
            start_index = len(req_state.block_ids)
            end_index = start_index + num_new_blocks
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table_cpu[
                req_index, start_index:end_index] = req_data.new_block_ids

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            # HACK AOYU reuse cached request state in gpu_input_batch.py, may change in neuron
            # self.requests[req_id] = CachedRequestState(
            #     req_id=req_id,
            #     prompt_token_ids=req_data.prompt_token_ids,
            #     prompt=req_data.prompt,
            #     # multi_modal_data=req_data.multi_modal_data,
            #     multi_modal_data=req_data.mm_inputs,
            #     sampling_params=sampling_params,
            #     generator=generator,
            #     block_ids=req_data.block_ids,
            #     num_computed_tokens=req_data.num_computed_tokens,
            #     output_token_ids=[],
            # )
            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt=new_req_data.prompt,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for res_req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = res_req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = res_req_data.block_ids
            req_state.num_computed_tokens = res_req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # THIS MOVES ALL THE DECODES TO THE FIRST N IN BATCH.
        # Condense the batched states if there are empty indices.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        # ALL THE PREFILLS ARE THE LAST M IN THE BATCH.
        # These are added at the end after the bacth is condensed.
        self.input_batch.num_prefills = len(req_ids_to_add)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            self.input_batch.add_request(req_state, None)


    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = np.concatenate(
            [self.arange_np[:n] for n in num_scheduled_tokens])

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        block_table_cpu = self.input_batch.block_table.get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens,
                  out=self.query_start_loc_np[1:num_reqs + 1])

        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max()
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=self.seq_start_loc_np[1:num_reqs + 1])

        # Copy the tensors to the GPU.
        # self.input_ids[:total_num_scheduled_tokens].copy_(
        #     self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        # self.positions[:total_num_scheduled_tokens].copy_(
        #     self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        copy_index = torch.arange(total_num_scheduled_tokens, device=self.device)
        self.input_ids.index_copy_(0, copy_index,
                                   self.input_ids_cpu[:total_num_scheduled_tokens].to(device=self.device))
        self.positions.index_copy_(0, copy_index,
                                   self.positions_cpu[:total_num_scheduled_tokens].to(device=self.device))
        query_start_loc = self.query_start_loc_cpu[:num_reqs + 1].to(self.device)
        seq_start_loc = self.seq_start_loc_cpu[:num_reqs + 1].to(self.device)
        slot_mapping = self.slot_mapping_cpu[:total_num_scheduled_tokens].to(self.device).long()

        # Prepare for cascade attention if needed.
        common_prefix_len = (scheduler_output.num_common_prefix_blocks *
                             self.block_size)
        use_cascade = False
        cu_prefix_query_lens = None
        cu_prefix_kv_lens = None
        cu_suffix_kv_lens = None

        block_table = self.input_batch.block_table.get_device_tensor()[:num_reqs]

        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=block_table,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            cu_prefix_kv_lens=cu_prefix_kv_lens,
            cu_suffix_kv_lens=cu_suffix_kv_lens,
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return attn_metadata, logits_indices

    def _prepare_sampling(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> SamplingMetadata:
        skip_copy = True
        if (scheduler_output.finished_req_ids
                or scheduler_output.preempted_req_ids):
            skip_copy = False
        if (scheduler_output.scheduled_new_reqs
                or scheduler_output.scheduled_resumed_reqs):
            skip_copy = False
        # Create the sampling metadata.
        sampling_metadata = self.input_batch.make_sampling_metadata(skip_copy)
        return sampling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        if self.is_multimodal_model:
            # Run the multimodal encoder if any.
            self._execute_encoder(scheduler_output)
            encoder_outputs = self._gather_encoder_outputs(scheduler_output)
        else:
            encoder_outputs = []

        # Prepare the decoder inputs.
        attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if (self.use_cuda_graph
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = num_scheduled_tokens
        attn_metadata.num_input_tokens = num_input_tokens

        if self.is_multimodal_model:
            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            input_ids = self.input_ids[:num_scheduled_tokens]
            if encoder_outputs:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, encoder_outputs)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids,
                positions=self.positions[:num_input_tokens],
                kv_caches=self.kv_caches,
                attn_metadata=None,
                inputs_embeds=inputs_embeds,
            )
        hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_token_ids = sampler_output.sampled_token_ids
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            assert req_id is not None
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                self.input_batch.num_tokens[i] += 1
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids.cpu()
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs.cpu()

        # num_reqs entries should be non-None
        assert all(
            req_id is not None for req_id in
            self.input_batch.req_ids[:num_reqs]), "req_ids contains None"
        req_ids = cast(List[str], self.input_batch.req_ids[:num_reqs])

        model_runner_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    def load_model(self) -> None:

        # NOTE(woosuk): While the executor assigns the TP ranks to the worker
        # process, the ranks can be different from the ranks internally assigned
        # by the xm runtime. Therefore, there is a mismatch in the rank
        # assignment between the gloo (cpu) runtime and the xm (tpu) runtime.
        # This is not a problem in linear layers because all-reduce is
        # rank-agnostic. However, it matters for all-gather as the ranks
        # determine the order of concatenating the output tensors.
        # As a workaround, we use the xm's rank assignment only when loading
        # the embedding weights.

        # xm_tp_rank = xr.global_ordinal()
        # with patch(
        #         "vllm.model_executor.layers.vocab_parallel_embedding."
        #         "get_tensor_model_parallel_rank",
        #         return_value=xm_tp_rank):
        #     model = get_model(vllm_config=self.vllm_config)
        model = get_model(vllm_config=self.vllm_config)
        model = model.eval()
        xm.wait_device_ops()
        with set_current_vllm_config(self.vllm_config):
            self.model = ModelWrapper(model)

    def _dummy_run(self, batch_size: int, seq_len: int,
                   kv_caches: List[torch.Tensor], is_prompt: bool) -> None:
        """Dummy warmup run for memory usage and graph compilation."""

        input_ids = torch.zeros((batch_size, seq_len),
                                dtype=torch.int32,
                                device=self.device)
        position_ids = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int32,
                                   device=self.device)
        slot_mapping = torch.zeros((batch_size, seq_len),
                                   dtype=torch.int64,
                                   device=self.device)
        block_tables = None if is_prompt else torch.zeros(
            (batch_size, self.max_num_blocks_per_req),
            dtype=torch.int32,
            device=self.device,
        )
        context_lens = None if is_prompt else torch.ones(
            (batch_size, ),
            dtype=torch.int32,
            device=self.device,
        )
        attn_metadata = NeuronAttentionMetadata(
            is_prompt=is_prompt,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
        )

        # NOTE: There are two stages of compilation: torch.compile and
        # XLA compilation. Using `mark_dynamic` can reduce the torch.compile
        # overhead by reusing the FX graph for different shapes.
        # However, the XLA graph will still require static shapes and needs to
        # be re-compiled for every different shapes. This overhead is inevitable
        # in the first run, but can be skipped afterwards as we cache the XLA
        # graphs in the disk (VLLM_XLA_CACHE_PATH).
        if is_prompt:
            torch._dynamo.mark_dynamic(input_ids, 1)
            torch._dynamo.mark_dynamic(position_ids, 1)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 1)
        else:
            torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(position_ids, 0)
            torch._dynamo.mark_dynamic(attn_metadata.slot_mapping, 0)
            torch._dynamo.mark_dynamic(attn_metadata.context_lens, 0)
            torch._dynamo.mark_dynamic(attn_metadata.block_tables, 0)

        # Dummy run.
        self.model(input_ids,
                   position_ids,
                   attn_metadata,
                   kv_caches,
                   is_prompt=is_prompt)

    def profile_run(self) -> None:
        """Profile to measure peak memory during forward pass."""

        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value `None`.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        # it is important to create tensors inside the loop, rather than
        # multiplying the list, to avoid Dynamo from treating them as
        # tensor aliasing.
        dummy_kv_caches = [(
            torch.tensor([], dtype=torch.float32, device=self.device),
            torch.tensor([], dtype=torch.float32, device=self.device),
        ) for _ in range(self.num_attn_layers)]

        # Round to multiple of 16.
        seq_len = (self.max_num_tokens + 15) // 16 * 16

        # Run empty forward.
        self._dummy_run(batch_size=1,
                        seq_len=seq_len,
                        kv_caches=dummy_kv_caches,
                        is_prompt=True)

    def capture_model(self) -> None:
        """Compile the model."""

        logger.info("Compiling the model with different input shapes.")

        # Prefill shapes.
        start = time.perf_counter()
        for batch_size in [1]:
            seq_len = 16
            while True:
                self._dummy_run(batch_size,
                                seq_len,
                                self.kv_caches,
                                is_prompt=True)
                xm.wait_device_ops()
                logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)
                if seq_len >= self.model_config.max_model_len:
                    break
                num_tokens = batch_size * seq_len
                if num_tokens >= self.scheduler_config.max_num_batched_tokens:
                    break
                seq_len = seq_len * 2

        end = time.perf_counter()
        logger.info("Compilation for prefill done in %.2f s.", end - start)

        # Decode shapes.
        start = time.time()
        seq_len = 1
        batch_size = 8  # Must be in sync with _get_padded_batch_size()
        while True:
            self._dummy_run(batch_size,
                            seq_len,
                            self.kv_caches,
                            is_prompt=False)
            xm.wait_device_ops()
            logger.info("batch_size: %d, seq_len: %d", batch_size, seq_len)

            if batch_size >= self.scheduler_config.max_num_seqs:
                break
            batch_size = batch_size + 16 if batch_size >= 16 else batch_size * 2

        end = time.time()
        logger.info("Compilation for decode done in %.2f s.", end - start)

    def initialize_kv_cache(self, num_blocks: int) -> None:
        assert len(self.kv_caches) == 0
        kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        for _ in range(self.num_attn_layers):
            self.kv_caches.append((
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device),
                torch.zeros(kv_cache_shape,
                            dtype=self.kv_cache_dtype,
                            device=self.device),
            ))

class ModelWrapper(TorchCompileWrapperWithCustomDispatcher):

    def __init__(self, model: nn.Module):
        self.model = model
        compiled_callable = torch.compile(self.forward,
                                          backend="openxla",
                                          fullgraph=False,
                                          dynamic=False)
        super().__init__(compiled_callable)

    # HACK AOYU bypass compiled number check
    def __call__(self, *args, is_prompt: bool, **kwargs):
        if len(self.compiled_codes) < 3 or not self.use_custom_dispatcher:
            # not fully compiled yet, or not using the custom dispatcher,
            # let PyTorch handle it
            return self.compiled_callable(*args, **kwargs)
        # the 3 compiled codes are:
        # 0: for profiling
        # 1: for prompt
        # 2: for decode
        # dispatch to the compiled code directly, skip PyTorch
        if is_prompt:
            with self.dispatch_to_code(1):
                return self.forward(*args, **kwargs)
        else:
            with self.dispatch_to_code(2):
                return self.forward(*args, **kwargs)
        return self.forward(*args, **kwargs)

    def forward(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """Executes the forward pass of the model and samples the next token.

        Args:
            token_ids: The input token IDs of shape [batch_size, seq_len].
            position_ids: The input position IDs of shape [batch_size, seq_len].
            attn_metadata: The Neuron attention metadata.
            kv_caches: The key and value caches. They can be None during the
                memory profiling at initialization.
        """

        # Skip this in memory profiling at initialization.
        if kv_caches[0][0].numel() > 0:
            # index_copy_(slot_mapping) only works when the inserted dimension
            # is 0. However, the KV cache in the Pallas backend has the shape
            # [num_kv_heads, num_blocks, block_size, head_size]. To make it
            # work, we need to flatten the first three dimensions and modify
            # the slot_mapping accordingly.
            num_kv_heads, num_blocks, block_size, _ = kv_caches[0][0].shape
            slot_mapping = attn_metadata.slot_mapping
            slot_mapping = slot_mapping.flatten()
            head_indicies = torch.arange(0,
                                         num_kv_heads,
                                         device=slot_mapping.device,
                                         dtype=slot_mapping.dtype)
            head_indicies *= block_size * num_blocks
            slot_mapping = slot_mapping.repeat_interleave(num_kv_heads).view(
                -1, num_kv_heads)
            slot_mapping = slot_mapping + head_indicies.view(1, -1)
            slot_mapping = slot_mapping.flatten()
            attn_metadata.slot_mapping = slot_mapping

        hidden_states = self.model(
            token_ids,
            position_ids,
            kv_caches,
            attn_metadata,
        )
        hidden_states = hidden_states.flatten(0, 1)
        logits = self.model.compute_logits(hidden_states, None)

        # Greedy sampling.
        # HACK AOYU only use greddy sampling, refer to tpu_model_runner_v1
        # # Greedy sampling.
        # argmax_token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        _, argmax_token_ids = torch.max(logits, dim=-1, keepdim=True)
        return argmax_token_ids


def _get_padded_batch_size(batch_size: int) -> int:
    # The GMM Pallas kernel requires num_tokens * topk to be a multiple of 16.
    # To meet this requirement in the simplest way, we set the minimal batch
    # size to 8.
    if batch_size <= 8:
        return 8
    else:
        return ((batch_size + 15) // 16) * 16


def _get_padded_prefill_len(x: int) -> int:
    # NOTE(woosuk): The pallas FlashAttention kernel requires the sequence
    # length to be a multiple of 16. We pad the prompt length to the nearest
    # multiple of 16. This is also good for performance.
    if x <= 16:
        return 16
    return 1 << (x - 1).bit_length()
