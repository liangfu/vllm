# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from vllm.utils import round_up, cdiv
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadataBuilder,
                                              AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

B_P_SIZE = 128

@torch.library.custom_op("vllm::neuron_paged_attn", mutates_args=())
def neuron_paged_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: int,
) -> torch.Tensor:
    global B_P_SIZE

    from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_func

    _, _, _, block_size, _ = kv_cache.shape

    # block-aligned k/v lengths
    # [3, 5, 4, 0] -(divide)-> [1, 2, 1, 0] -(multiply)-> [4, 8, 4, 0]
    num_blocks_per_seq = (context_lens + block_size - 1) // block_size
    zero = torch.tensor([0], dtype=context_lens.dtype, device=context_lens.device)
    cu_ctx_lens_blockaligned = (
        torch.cat((zero, context_lens), dim=0) * block_size
    ).cumsum(dim=0)

    # build active block table
    # TODO(liangfu): move this implementation into NKI kernel
    active_block_table = torch.tensor([], dtype=torch.int32, device=context_lens.device)
    print(f"neuron_attn.py: {block_tables.shape=}")
    print(f"{num_blocks_per_seq=}")
    print(f"{num_seqs=}")
    for seq_idx in range(num_seqs):
        active_block_table = torch.cat((active_block_table, block_tables[seq_idx, :num_blocks_per_seq[seq_idx]]), dim=0)
    num_blocks = active_block_table.numel()
    active_block_table = F.pad(active_block_table, (0, round_up(num_blocks, B_P_SIZE)-num_blocks), "constant", 0).to(dtype=torch.int32)

    max_seqlen_q = 128
    max_seqlen_k = 2048

    output = flash_attn_varlen_func(
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        block_tables=active_block_table,
        cu_seqlens_q=query_start_loc,
        cu_seqlens_k=cu_ctx_lens_blockaligned,
        seqused_k=context_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_seqs=num_seqs,
    )
    return output


@neuron_paged_attn.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: int,
) -> torch.Tensor:
    return torch.empty_like(query.transpose(-2, -1))


class NeuronAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "NEURON"

    @staticmethod
    def get_impl_cls() -> type["NeuronAttentionBackendImpl"]:
        return NeuronAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> type["NeuronAttentionMetadata"]:
        return NeuronAttentionMetadata

    @staticmethod
    def get_state_cls() -> type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> type["NeuronAttentionMetadataBuilder"]:
        return NeuronAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (2, num_blocks, num_kv_heads, block_size, head_size)


@dataclass
class NeuronAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Used in the NeuronAttentionBackendImpl
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_seqs: int  # Number of actual sequences


class NeuronAttentionMetadataBuilder(
        AttentionMetadataBuilder[NeuronAttentionMetadata]):
    ...


class NeuronAttentionBackendImpl(AttentionImpl[NeuronAttentionMetadata]):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list[float]] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.scale = scale

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from vllm.attention.ops.nki_flash_attn import reshape_and_cache

        num_tokens = query.shape[1]
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        if kv_cache.numel() > 0:
            torch.ops.xla.dynamo_set_buffer_donor_(kv_cache, True)
            slot_mapping = attn_metadata.slot_mapping
            reshape_and_cache(key, value, kv_cache, slot_mapping)
        else:
            # profiling run
            return query

        query = query.view(num_tokens, self.num_heads, self.head_size)
        query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
        value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

        output = neuron_paged_attn(
            query,
            key,
            value,
            kv_cache,
            attn_metadata.context_lens,
            attn_metadata.block_tables,
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
        )
        output = output.transpose(1, 2).reshape(
            1, num_tokens, self.num_heads * self.head_size)
        return output
