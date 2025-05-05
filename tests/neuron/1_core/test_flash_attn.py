# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from vllm.utils import cdiv, round_up
from vllm.platforms import current_platform
from vllm.attention.ops.nki_flash_attn import flash_attn_varlen_func

B_P_SIZE = 128

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
DTYPES = [torch.float16]
QDTYPES = [None]
# one value large enough to test overflow in index calculation.
# one value small enough to test the schema op check
NUM_BLOCKS = [2048]

def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def sample_inputs(
    query_lens,
    kv_lens,
    num_blocks,
    block_size,
    num_heads,
    num_kv_heads,
    head_size,
    dtype,
):
    seq_lens = [a + b for a, b in zip(query_lens, kv_lens)]
    num_seqs = len(seq_lens)
    max_block_per_request = cdiv(max(seq_lens), block_size)

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1, 1)
    torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1, 1)
    key, value = kv.unbind(dim=1)

    k_cache = torch.zeros(num_blocks,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    v_cache = torch.zeros(num_blocks,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)

    # build block_tables
    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]
    block_tables = values[:num_seqs * max_block_per_request].view(
        num_seqs, max_block_per_request)

    b_ctx_len = torch.tensor(kv_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(num_seqs):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_tables[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1

    return (
        query,
        k,
        v,
        k_cache,
        v_cache,
        block_tables,
        query_lens,
        seq_lens,
    )


@pytest.mark.parametrize("seq_lens", [
    [(5, 18), (19, 463), (1, 1328)],
    [(1, 523), (1, 37), (1, 2011)]
])
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@torch.inference_mode()
def test_varlen_with_paged_kv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
) -> None:
    torch.set_default_device("cpu")
    current_platform.seed_everything(0)
    num_seqs = torch.tensor(len(seq_lens), dtype=torch.int32)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32).cumsum(dim=0,
                                                           dtype=torch.int32)

    (
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_tables,
        query_lens,
        seq_lens,
    ) = sample_inputs(
        query_lens,
        kv_lens,
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )

    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    # build neuron program
    kv_cache = torch.stack([key_cache, value_cache])

    # pad input tensors
    max_num_queries = round_up(sum(query_lens), 128)
    pad_dims = (0, 0, 0, 0, 0, max_num_queries - query.shape[0])
    query = F.pad(query, pad_dims, "constant", 0)
    key = F.pad(key, pad_dims, "constant", 0)
    value = F.pad(value, pad_dims, "constant", 0)

    # permute QKV tensors
    # query: (1, n_heads, d, seq_q)
    # key:   (1, n_kv_heads, d, seq_k)
    # value: (1, n_kv_heads, seq_v, d)
    query = query.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    key = key.unsqueeze(0).permute(0, 2, 3, 1).contiguous()
    value = value.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
    kv_cache = kv_cache.permute(0, 1, 3, 2, 4).contiguous()

    # block-aligned k/v lengths
    # [3, 5, 4, 0] -(divide)-> [1, 2, 1, 0] -(multiply)-> [4, 8, 4, 0]
    num_blocks_per_seq = cdiv(torch.tensor(kv_lens), block_size).tolist()
    cu_ctx_lens_blockaligned = (
        torch.tensor([0] + num_blocks_per_seq, dtype=torch.int32) * block_size
    ).cumsum(dim=0, dtype=torch.int32)

    # build active block table
    # TODO(liangfu): move this implementation into NKI kernel
    active_block_table = torch.tensor([], dtype=torch.int32)
    for seq_idx in range(num_seqs):
        active_block_table = torch.cat((active_block_table, block_tables[seq_idx, :num_blocks_per_seq[seq_idx]]), dim=0)
    num_blocks = active_block_table.numel()
    active_block_table = F.pad(active_block_table, (0, round_up(num_blocks, B_P_SIZE)-num_blocks), "constant", 0).to(dtype=torch.int32)
    print(f"{active_block_table.shape=}")

    max_seqlen_q = round_up(max(query_lens), 128)
    max_seqlen_k = round_up(max(kv_lens), 2048)
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    output = flash_attn_varlen_func(
        query=query.numpy(),
        key=key.numpy(),
        value=value.numpy(),
        kv_cache=kv_cache.numpy(),
        cu_seqlens_q=cu_query_lens.numpy(),
        cu_seqlens_k=cu_ctx_lens_blockaligned.numpy(),  # TODO(liangfu): remove this argument
        seqused_k=kv_lens.numpy(),
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        num_seqs=num_seqs.item(),
        block_tables=active_block_table.numpy(),  # TODO(liangfu): transform to dense block_table
        softmax_scale=scale,
    )

    # atol, rtol = 1.5e-2, 1e-2
    # torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol), \
    #     f"{torch.max(torch.abs(output - ref_output))}"
