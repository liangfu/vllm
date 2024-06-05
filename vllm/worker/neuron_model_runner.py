from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import (AttentionMetadata, AttentionMetadataPerStage,
                            get_attn_backend)
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader.neuron import get_neuron_model
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import (async_tensor_h2d, is_pin_memory_available,
                        make_tensor_with_pad, maybe_expand_dim)

logger = init_logger(__name__)


class NeuronModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config

        self.sliding_window = None
        if model_config is not None and model_config.get_sliding_window():
            logger.warning("Sliding window is not supported on Neuron. "
                           "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.attn_backend = get_attn_backend(
            self.model_config.dtype if model_config is not None else None)

        # Lazy initialization.
        self.model: nn.Module  # initialize after load_model.
        self.block_size: int  # Set after initial profiling.

    def load_model(self) -> None:
        self.model = get_neuron_model(self.model_config,
                                      parallel_config=self.parallel_config,
                                      scheduler_config=self.scheduler_config)

    def compile_model(self) -> None:
        self.model.model.to_neuron()

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        prompt_lens: List[int] = []
        context_lens: List[int] = []
        prefix_block_tables: List[List[int]] = []

        if len(seq_group_metadata_list) == 0:
            return PreparePromptMetadata.empty()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            computed_block_nums = seq_group_metadata.computed_block_nums
            if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None
                             or computed_block_nums == [])):
                raise RuntimeError(
                    "chunked prefill cannot be used with prefix caching "
                    "now.")

            token_chunk_size = seq_group_metadata.token_chunk_size
            seq_data = seq_group_metadata.seq_data[seq_id]
            computed_len = seq_data.get_num_computed_tokens()
            # We should use get_len here because in case of preemption
            # it contains output tokens.
            prefill_end = min(seq_data.get_len(),
                              computed_len + token_chunk_size)
            prompt_tokens = seq_data.get_token_ids()[computed_len:prefill_end]
            prompt_len = prefill_end
            prompt_lens.append(prompt_len)

            prefix_block_tables.append([])
            # Right now, prefill start is always 0. However, this
            # assumption can be changed once chunked prefill is introduced.
            assert computed_len == 0

            # actual prompt lens
            context_lens.append(computed_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prefill_end)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0

            for i in range(computed_len, prefill_end):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_prompt_len = max(prompt_lens)

        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device=self.device)

        # Prepare prefix block tables
        max_prompt_block_table_len = max(len(t) for t in prefix_block_tables)
        block_tables = make_tensor_with_pad(
            prefix_block_tables,
            max_len=max_prompt_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        prompt_lens_tensor = torch.tensor(prompt_lens,
                                          dtype=torch.long,
                                          device=self.device)
        seq_start_loc = torch.zeros(prompt_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=self.device)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device=self.device)

        torch.cumsum(prompt_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            prompt_lens=prompt_lens,
            prompt_lens_tensor=prompt_lens_tensor,
            num_prefills=len(prompt_lens),
            num_prefill_tokens=sum(prompt_lens),
            num_decode_tokens=0,
            prefill_metadata=None,
            decode_metadata=None,
            max_context_len=None,
            context_lens=None,
            block_tables=torch.tensor([]),
            slot_mapping=slot_mapping,
            kv_cache_dtype="bfloat16", # "auto", # "auto" means use model weight data type
        )
        input_tokens_tensor = torch.tensor(input_tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        input_positions_tensor = torch.tensor(input_positions, dtype=torch.long, device=self.device).unsqueeze(0)
        return (input_tokens_tensor, input_positions_tensor, attn_metadata, prompt_lens)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_context_len = max(context_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device).unsqueeze(-1)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device).unsqueeze(-1)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            prompt_lens_tensor=None,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            max_context_len=max_context_len,
            num_prefills=0,
            prefill_metadata=None,
            decode_metadata=None,
            context_lens=context_lens,
            block_tables=block_tables,
            kv_cache_dtype="auto",
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        generators: List[torch.Generator] = []
        selected_token_start_idx = 0
        categorized_sample_indices: Dict[SamplingType,
                                         List[Tuple[int, int]]] = {
                                             t: []
                                             for t in SamplingType
                                         }
        categorized_sample_indices_start_idx = 0
        categorized_sampled_token_indices_start_idx = 0

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                assert prompt_lens is not None
                prompt_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += prompt_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        (categorized_sample_indices_start_idx,
                         categorized_sampled_token_indices_start_idx))
                categorized_sample_indices_start_idx += 1
                categorized_sampled_token_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + prompt_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              prompt_len - 1)
                selected_token_start_idx += prompt_len

                if sampling_params.seed is not None:
                    seq_group_metadata.state.generator = torch.Generator(
                        device=self.device).manual_seed(sampling_params.seed)
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        zip(
                            range(
                                categorized_sample_indices_start_idx,
                                categorized_sample_indices_start_idx +
                                num_seqs),
                            range(
                                categorized_sampled_token_indices_start_idx,
                                categorized_sampled_token_indices_start_idx +
                                num_seqs)))
                categorized_sample_indices_start_idx += num_seqs
                categorized_sampled_token_indices_start_idx += num_seqs

            if sampling_params.seed is not None:
                generators.append(seq_group_metadata.state.generator)

        selected_token_indices = async_tensor_h2d(selected_token_indices,
                                                  dtype=torch.long,
                                                  target_device=self.device,
                                                  pin_memory=self.pin_memory)

        categorized_sample_indices = {
            t: maybe_expand_dim(
                async_tensor_h2d(seq_ids,
                                 dtype=torch.int,
                                 target_device=self.device,
                                 pin_memory=self.pin_memory), 2, 2)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            generators=generators,
        )
        return sampling_metadata

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, SamplingMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_metadata,
             prompt_lens) = self._prepare_prompt(seq_group_metadata_list)
        else:
            (input_tokens, input_positions,
             input_metadata) = self._prepare_decode(seq_group_metadata_list)
            prompt_lens = []
        sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                 prompt_lens)

        return (input_tokens, input_positions, input_metadata,
                sampling_metadata)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, input_metadata, sampling_metadata
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            input_metadata=input_metadata,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()
