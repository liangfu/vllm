import numpy as np
import pytest

@pytest.mark.parametrize("query_lens,seq_lens", [
    ([2, 3, 1, 0], [4, 8, 4, 0]),
])
def test_context_mask(
    query_lens, seq_lens
) -> None:
    from vllm.attention.ops.nki_flash_attn import build_attention_mask

    max_query_lens = 8
    max_seq_lens = 20
    max_num_seqs = 4
    block_size = 4

    # block-aligned k/v lengths
    # [3, 5, 4, 0] -(divide)-> [1, 2, 1, 0] -(multiply)-> [4, 8, 4, 0]
    ctx_lens = np.array(seq_lens) - np.array(query_lens)
    num_blocks_per_seq = (np.array(ctx_lens) + block_size - 1) // block_size
    cu_ctx_lens = (np.array([0] + num_blocks_per_seq.tolist(), dtype=np.int32) * block_size).cumsum(axis=0, dtype=np.int32)

    cu_query_lens = np.array([0] + query_lens, dtype=np.int32).cumsum(axis=0, dtype=np.int32)
    seq_lens = np.array(seq_lens).astype(np.float32)

    expected = (
        # prior_mask
        np.array([[
        # i0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19
        # c0,c0,p0,p0,c1,c1,c1,c1,c1,p1,p1,p1,c2,c2,c2,p2,p3,p3,p3,p3 - (c - cached token, n - new token, p - padding)
        [  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 0 (2 cached tokens, aligned to 1 block)
        [  1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 0 (2 cached tokens, aligned to 1 block)
        [  0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 cached tokens, aligned to 2 blocks)
        [  0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 cached tokens, aligned to 2 blocks)
        [  0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sequence 1 (5 cached tokens, aligned to 2 blocks)
        [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], # sequence 2 (3 cached tokens, aligned to 1 block)
        [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        [  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # padding
        ],], dtype=np.int32),

        # active_mask
        np.array([[
        # i0, 1, 2, 3, 4, 5, 6, 7
        # n0,n0,n1,n1,n1,n2,p3,p3 - (c - cached token, n - new token, p - padding)
        [  1, 0, 0, 0, 0, 0, 0, 0], # sequence 0 (2 active tokens)
        [  1, 1, 0, 0, 0, 0, 0, 0], # sequence 0 (2 active tokens)
        [  0, 0, 1, 0, 0, 0, 0, 0], # sequence 1 (3 active tokens)
        [  0, 0, 1, 1, 0, 0, 0, 0], # sequence 1 (3 active tokens)
        [  0, 0, 1, 1, 1, 0, 0, 0], # sequence 1 (3 active tokens)
        [  0, 0, 0, 0, 0, 1, 0, 0], # sequence 2 (1 active tokens)
        [  0, 0, 0, 0, 0, 0, 1, 0], # padding
        [  0, 0, 0, 0, 0, 0, 1, 1], # padding
        ],], dtype=np.int32)
    )

    print(f"{cu_query_lens=}, {cu_ctx_lens=}, {seq_lens=}, {max_query_lens=}, {max_seq_lens=}, {max_num_seqs=}")

    # prior mask
    prior_mask, *debug_tensors = build_attention_mask(
        cu_query_lens, cu_ctx_lens, seq_lens, max_query_lens, max_seq_lens, max_num_seqs, block_size)
    prior_mask = prior_mask.astype(bool).astype(np.int32)
    loop_var = debug_tensors[-1]
    np.testing.assert_allclose(expected[0], prior_mask)

    # active mask
    cu_ctx_lens = np.array([0] + query_lens, dtype=np.int32).cumsum(axis=0, dtype=np.int32)
    seq_lens = np.array(query_lens, dtype=np.int32)
    active_mask, *debug_tensors = build_attention_mask(
        cu_query_lens, cu_ctx_lens, seq_lens, max_query_lens, max_seq_lens, max_num_seqs, block_size, contexted=True)
    active_mask = active_mask.astype(bool).astype(np.int32)
    loop_var = debug_tensors[-1]
    np.testing.assert_allclose(expected[1], active_mask)
