import numpy as np
import pytest

import neuronxcc.nki.language as nl
from neuronxcc import nki
import neuronxcc.nki.typing as nt

from vllm.attention.ops.nki_flash_attn import build_attention_mask


@pytest.mark.parametrize(
    "query_lens,seq_lens",
    [
        ([2, 3, 1, 0], [4, 8, 4, 0]),
    ],
)
def test_context_mask(query_lens, seq_lens) -> None:

    # max_query_lens = 8
    # max_seq_lens = 20
    max_query_lens = 128
    max_seq_lens = 512
    max_num_seqs = 4
    block_size = 4

    # block-aligned k/v lengths
    # [3, 5, 4, 0] -(divide)-> [1, 2, 1, 0] -(multiply)-> [4, 8, 4, 0]
    ctx_lens = np.array(seq_lens) - np.array(query_lens)
    num_blocks_per_seq = (np.array(ctx_lens) + block_size - 1) // block_size
    cu_ctx_lens_blockaligned = (np.array([0] + num_blocks_per_seq.tolist(), dtype=np.int32) * block_size).cumsum(axis=0, dtype=np.int32)

    cu_query_lens = np.array([0] + query_lens, dtype=np.int32).cumsum(axis=0, dtype=np.int32)
    seq_lens = np.array(seq_lens).astype(np.float32)

    expected = (
        # prior_mask
        np.array(
            [
                # i0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19
                # c0,c0,p0,p0,c1,c1,c1,c1,c1,p1,p1,p1,c2,c2,c2,p2,p3,p3,p3,p3 - (c - cached token, n - new token, p - padding)
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sequence 0 (2 cached tokens, aligned to 1 block)
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sequence 0 (2 cached tokens, aligned to 1 block)
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sequence 1 (5 cached tokens, aligned to 2 blocks)
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sequence 1 (5 cached tokens, aligned to 2 blocks)
                [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # sequence 1 (5 cached tokens, aligned to 2 blocks)
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # sequence 2 (3 cached tokens, aligned to 1 block)
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # padding
            ],
            dtype=np.int32,
        ),
        # active_mask
        np.array(
            [
                # i0, 1, 2, 3, 4, 5, 6, 7
                # n0,n0,n1,n1,n1,n2,p3,p3 - (c - cached token, n - new token, p - padding)
                [1, 0, 0, 0, 0, 0, 0, 0],  # sequence 0 (2 active tokens)
                [1, 1, 0, 0, 0, 0, 0, 0],  # sequence 0 (2 active tokens)
                [0, 0, 1, 0, 0, 0, 0, 0],  # sequence 1 (3 active tokens)
                [0, 0, 1, 1, 0, 0, 0, 0],  # sequence 1 (3 active tokens)
                [0, 0, 1, 1, 1, 0, 0, 0],  # sequence 1 (3 active tokens)
                [0, 0, 0, 0, 0, 1, 0, 0],  # sequence 2 (1 active tokens)
                [0, 0, 0, 0, 0, 0, 0, 0],  # padding
                [0, 0, 0, 0, 0, 0, 0, 0],  # padding
            ],
            dtype=np.int32,
        ),
    )

    print(f"{cu_query_lens=}, {cu_ctx_lens_blockaligned=}, {seq_lens=}, " f"{max_query_lens=}, {max_seq_lens=}, {max_num_seqs=}")

    ref_prior_mask, ref_active_mask = expected
    pnr, pnc = ref_prior_mask.shape
    anr, anc = ref_active_mask.shape

    block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    prior_mask = np.zeros((max_query_lens, max_seq_lens), dtype=np.int32)
    active_mask = np.zeros((max_query_lens, max_query_lens), dtype=np.int32)

    # prior mask
    ctx_lens = np.array(ctx_lens, dtype=np.int32)
    build_attention_mask_wrapper(
        prior_mask,
        cu_query_lens,
        cu_ctx_lens_blockaligned,
        ctx_lens,
        max_query_lens,
        max_seq_lens,
        max_num_seqs=max_num_seqs,
        block_size=block_size,
        tiled_block_size=tiled_block_size,
    )
    prior_mask = prior_mask.astype(bool).astype(np.int32)
    np.testing.assert_allclose(ref_prior_mask, prior_mask[:pnr, :pnc])

    # active mask
    query_lens = np.array(query_lens, dtype=np.int32)
    build_attention_mask_wrapper(
        active_mask, cu_query_lens, cu_query_lens, query_lens, max_query_lens, max_query_lens, max_num_seqs=max_num_seqs, block_size=block_size, tiled_block_size=tiled_block_size, contexted=True
    )
    active_mask = active_mask.astype(bool).astype(np.int32)
    np.testing.assert_allclose(ref_active_mask, active_mask[:anr, :anc])


@nki.jit(experimental_flags="enable-mutable-parameter")
def build_attention_mask_wrapper(
    prior_mask: nt.mutable_tensor,
    cu_query_lens,
    cu_ctx_lens,
    seq_lens,
    max_num_queries,
    max_num_keys,
    max_num_seqs,
    block_size,
    tiled_block_size,
    contexted=False,
):
    i_p, i_f = nl.mgrid[0:1, 0:max_num_seqs]
    seq_lens_sbuf = nl.load(seq_lens[i_f])
    prior_mask = build_attention_mask(
        prior_mask,
        cu_query_lens,
        cu_ctx_lens,
        seq_lens_sbuf,
        max_num_queries,
        max_num_keys,
        max_num_seqs,
        block_size,
        tiled_block_size,
        contexted,
    )
    return prior_mask
