import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed
import torch.nn as nn

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, 
                        LayerBlockType, cdiv)
from vllm.v1.attention.backends.neuron_attn import NeuronAttentionBackend, \
    NeuronAttentionMetadata
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


B_P_SIZE = 128
LARGE_TILE_SZ = 2048


class NeuronModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        input_registry: InputRegistry = INPUT_REGISTRY,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = False
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_multimodal_model = model_config.is_multimodal_model
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()

        # Multi-modal data support
        self.input_registry = input_registry

        # Lazy initialization
        self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device="cpu",
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
        )

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device="cpu")
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device="cpu")
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device="cpu")

        # TODO(gnovack) - use compile sizes...
        self.neuron_compilation_batch_sizes = list(reversed(self.vllm_config.compilation_config.cudagraph_capture_sizes))

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        # Remove stopped requests from the cached states.
        # Keep the states of the pre-empted requests.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.encoder_cache.pop(req_id, None)

        # Free the cached encoder outputs.
        for req_id, input_id in scheduler_output.free_encoder_input_ids:
            encoder_outputs = self.encoder_cache.get(req_id)
            if encoder_outputs is not None:
                encoder_outputs.pop(input_id, None)
                if not encoder_outputs:
                    self.encoder_cache.pop(req_id, None)

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
            req_state.block_ids.extend(req_data.new_block_ids)
            self.input_batch.block_table.append_row(req_index, start_index,
                                                    req_data.new_block_ids)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            sampling_params = req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=req_data.prompt_token_ids,
                prompt=req_data.prompt,
                mm_inputs=req_data.mm_inputs,
                mm_positions=req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=req_data.block_ids,
                num_computed_tokens=req_data.num_computed_tokens,
                output_token_ids=[],
            )
            req_ids_to_add.append(req_id)

        # Update the cached states of the resumed requests.
        for req_data in scheduler_output.scheduled_resumed_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            req_state.block_ids = req_data.block_ids
            req_state.num_computed_tokens = req_data.num_computed_tokens
            req_ids_to_add.append(req_id)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (num_reqs, 1))
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
        arange = arange_matrix[mask]

        # Get positions.
        positions = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        positions_np = positions.numpy()
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices = torch.from_numpy(token_indices)
        input_ids = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.index_select(torch.from_numpy(
            self.input_batch.token_ids_cpu).flatten(),
                           0,
                           token_indices,
                           out=input_ids)

        # Calculate the slot mapping.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size` here
        # because M (max_model_len) is not necessarily divisible by block_size.
        block_numbers = self.input_batch.block_table.get_cpu_tensor().flatten()[
            req_indices * self.max_num_blocks_per_req +
            positions_np // self.block_size]
        block_offsets = torch.from_numpy(positions_np % self.block_size)
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)
        
        _PAD_SLOT_ID = self.num_blocks * self.block_size
        padded_num_tokens = self._get_padded_batch_size(total_num_scheduled_tokens)
        slot_mapping_pad_length = padded_num_tokens - slot_mapping.shape[0]
        slot_mapping = torch.nn.functional.pad(
            slot_mapping,
            (0, slot_mapping_pad_length),
            'constant',
            _PAD_SLOT_ID
        )

        # Prepare the attention metadata.
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max()
        seq_start_loc = torch.empty((num_reqs + 1, ),
                                    dtype=torch.int32,
                                    device="cpu",
                                    pin_memory=self.pin_memory)
        seq_start_loc_np = seq_start_loc.numpy()
        seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=seq_start_loc_np[1:])

        self.input_ids[:total_num_scheduled_tokens].copy_(input_ids,
                                                          non_blocking=True)
        self.positions[:total_num_scheduled_tokens].copy_(positions,
                                                          non_blocking=True)

        seq_lens = torch.diff(seq_start_loc)
        query_lens = torch.diff(query_start_loc)
        context_lens = seq_lens - query_lens
        num_active_blocks_shifted = shift_bit_length(
            ((context_lens+ self.block_size - 1) // self.block_size).sum().item()
        )
        num_active_blocks_factor = max(LARGE_TILE_SZ // self.block_size // num_active_blocks_shifted, 1)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
        assert (num_active_blocks * self.block_size) % LARGE_TILE_SZ == 0, "invalid {num_active_blocks=}"

        context_kv_len = num_active_blocks * self.block_size


        block_table = self.input_batch.block_table.get_cpu_tensor()[:num_reqs]
        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor(query_lens),
            torch.tensor(seq_lens),
            self.block_size,
            num_active_blocks,
        )

        prior_mask, active_mask = (
            BlockDiagonalCausalFromBottomRightMask.from_seqlens(
                query_lens=query_lens.tolist(), seq_lens=seq_lens.tolist(), block_size=self.block_size
            )
        )
        
        attn_mask = torch.concat(
            [
                nn.functional.pad(
                    prior_mask,
                    (
                        0,
                        max(context_kv_len, LARGE_TILE_SZ) - prior_mask.shape[1],
                        0,
                        B_P_SIZE - prior_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
                nn.functional.pad(
                    active_mask,
                    (
                        0,
                        padded_num_tokens - active_mask.shape[1],
                        0,
                        B_P_SIZE - active_mask.shape[0],
                    ),
                    "constant",
                    0,
                ).bool(),
            ],
            dim=1,
        )
        
        logits_indices = query_start_loc[1:] - 1
        query_start_loc = query_start_loc.to(self.device, non_blocking=True)
        seq_start_loc = seq_start_loc.to(self.device, non_blocking=True)
        slot_mapping = slot_mapping.long().to(self.device, non_blocking=True)
        active_block_table = active_block_table.to(torch.int32).to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device)
        attn_metadata = NeuronAttentionMetadata(
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_start_loc=seq_start_loc,
            block_table=self.input_batch.block_table.get_device_tensor()[:num_reqs],
            slot_mapping=slot_mapping,
            num_active_blocks=num_active_blocks,
            active_block_table=active_block_table,
            attn_mask=attn_mask
        )
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
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
        req_id_output_token_ids: Dict[str, List[int]] = \
            {req_id: req.output_token_ids \
                for req_id, req in self.requests.items()}
        
        sampling_metadata = self.input_batch.make_sampling_metadata(req_id_output_token_ids, skip_copy)
        return sampling_metadata

    def _execute_encoder(self, scheduler_output: "SchedulerOutput"):
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return

        # Batch the multi-modal inputs.
        mm_inputs: List[MultiModalKwargs] = []
        req_input_ids: List[Tuple[int, int]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]
            for input_id in encoder_input_ids:
                mm_inputs.append(req_state.mm_inputs[input_id])
                req_input_ids.append((req_id, input_id))
        batched_mm_inputs = MultiModalKwargs.batch(mm_inputs)
        batched_mm_inputs = MultiModalKwargs.as_kwargs(batched_mm_inputs,
                                                       device=self.device)

        # Run the encoder.
        # `encoder_outputs` is either of the following:
        # 1. A tensor of shape [num_images, feature_size, hidden_size]
        # in case when feature_size is fixed across all images.
        # 2. A list (length: num_images) of tensors, each of shape
        # [feature_size, hidden_size] in case when the feature size is
        # dynamic depending on input images.
        encoder_outputs = self.model.get_multimodal_embeddings(
            **batched_mm_inputs)

        # Cache the encoder outputs.
        for (req_id, input_id), output in zip(req_input_ids, encoder_outputs):
            if req_id not in self.encoder_cache:
                self.encoder_cache[req_id] = {}
            self.encoder_cache[req_id][input_id] = output

    def _gather_encoder_outputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> List[torch.Tensor]:
        encoder_outputs: List[torch.Tensor] = []
        num_reqs = self.input_batch.num_reqs
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens
            mm_positions = req_state.mm_positions
            for i, pos_info in enumerate(mm_positions):
                start_pos = pos_info["offset"]
                num_encoder_tokens = pos_info["length"]

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens)
                assert start_idx < end_idx
                assert req_id in self.encoder_cache
                assert i in self.encoder_cache[req_id]
                encoder_output = self.encoder_cache[req_id][i]
                encoder_outputs.append(encoder_output[start_idx:end_idx])
        return encoder_outputs

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
        num_input_tokens = self._get_padded_batch_size(num_scheduled_tokens)
        
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
            input_ids = self.input_ids[:num_input_tokens]
            inputs_embeds = None

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                positions=self.positions[:num_input_tokens].unsqueeze(0).to(self.device),
                kv_caches=self.kv_caches,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds.to(self.device) if inputs_embeds is not None else None,
            ).cpu()
        hidden_states = hidden_states[0, :num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices.cpu()]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_token_ids = sampler_output.sampled_token_ids.tolist()
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
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
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    def load_model(self) -> None:
        # TODO(gnovack) - Add memory profiler during model load
        with torch.inference_mode():
            logger.info("Starting to load model %s...", self.model_config.model)
            model = get_model(vllm_config=self.vllm_config).eval().to(self.device)
            self.model = torch.compile(model, backend="openxla", fullgraph=True, dynamic=False)


    @torch.inference_mode()
    def _dummy_run(
        self,
        model: nn.Module,
        num_tokens: int,
        kv_caches: List[torch.Tensor],
    ) -> torch.Tensor:
        
        num_active_blocks_shifted = shift_bit_length(
            ((self.block_size - 1) // self.block_size)
        )
        num_active_blocks_factor = (LARGE_TILE_SZ // self.block_size // num_active_blocks_shifted)
        num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
        block_table = torch.arange((num_tokens // self.block_size) + 1).unsqueeze(0)
        active_block_table = get_active_block_tables(
            block_table,
            torch.tensor([num_tokens]),
            torch.tensor([num_tokens]),
            self.block_size,
            num_active_blocks,
        )

        attn_mask, _  = BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens=[num_tokens], seq_lens=[num_tokens]
        )
        attn_mask = nn.functional.pad(
            attn_mask,
            (
                0,
                LARGE_TILE_SZ + num_tokens - attn_mask.shape[1],
                0,
                B_P_SIZE - attn_mask.shape[0],
            ),
            "constant",
            0,
        ).bool()

        attn_metadata = NeuronAttentionMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens,
            query_start_loc=torch.tensor([0, num_tokens-1]).to(self.device, non_blocking=True),
            max_seq_len=num_tokens,
            seq_start_loc=torch.tensor([0, num_tokens-1]).to(self.device, non_blocking=True),
            block_table=block_table,
            slot_mapping=torch.arange(0, num_tokens).long().to(self.device, non_blocking=True),
            num_active_blocks=num_active_blocks,
            active_block_table=active_block_table.to(torch.int32).to(self.device, non_blocking=True),
            attn_mask=attn_mask.to(self.device, non_blocking=True)
        )

        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                positions=self.positions[:num_tokens].unsqueeze(0).to(self.device),
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                inputs_embeds=inputs_embeds.to(self.device) if inputs_embeds is not None else None,
            )
        return hidden_states

    def profile_run(self) -> None:
        # TODO(gnovack): implement profiling run for neuron
        dummy_kv_caches = [
            (torch.tensor([], dtype=torch.float32, device=self.device), torch.tensor([], dtype=torch.float32, device=self.device))
            for _ in range(self.num_attn_layers)
        ]
        num_tokens = max(self.neuron_compilation_batch_sizes)
        self._dummy_run(self.model, num_tokens, dummy_kv_caches)

    def capture_model(self) -> None:

        start_time = time.perf_counter()

        # Trigger Neuron compilation for specific shapes
        for num_tokens in reversed(self.neuron_compilation_batch_sizes):
            self._dummy_run(self.model, num_tokens, self.kv_caches)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info("Neuron compilation finished in %.0f secs", elapsed_time)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        assert len(self.kv_caches) == 0
        self.num_blocks = kv_cache_config.num_blocks

        kv_caches: Dict[str, torch.Tensor] = {}

        with torch.inference_mode():
            kv_cache_shape = NeuronAttentionBackend.get_kv_cache_shape(
                self.num_blocks + 1, self.block_size, self.num_kv_heads, self.head_size)
            for layer_name, layer_spec in kv_cache_config.kv_cache_spec.items():
                cache = torch.zeros(kv_cache_shape,
                                dtype=self.kv_cache_dtype,
                                device='cpu')
                kv_caches[layer_name] = cache.to(self.device)
        
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def _get_padded_batch_size(self, batch_size: int) -> Optional[int]:
        # TODO: Optimize this?
        for size in self.neuron_compilation_batch_sizes:
            if batch_size <= size:
                return size
        return None

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each 
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache 
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: KVCacheSpec = {}
        for layer_name, attn_module in forward_ctx.items():
            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention, MLA.
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                )
            else:
                raise NotImplementedError
        return kv_cache_spec


def get_active_block_tables(block_tables, query_lens, seq_lens, block_size,
                                    num_blocks):
    context_lens = seq_lens - query_lens
    blocks_per_seq = (context_lens + block_size - 1) // block_size
    num_seqs = len(seq_lens)
    active_blocks: list[int] = []
    for seq_id in range(num_seqs):
        active_blocks = (
            active_blocks +
            block_tables[seq_id, :blocks_per_seq[seq_id]].tolist())
    return nn.functional.pad(
        torch.tensor(active_blocks),
        (0, num_blocks - len(active_blocks)),
        "constant",
        0,
    )


class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = (torch.arange(n_queries).reshape(n_queries,
                                             1).expand(n_queries, n_keys))
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens)
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size)
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens)
        return prior_mask, active_mask

def shift_bit_length(x):
    return 1 << (x - 1).bit_length()