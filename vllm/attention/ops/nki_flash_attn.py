"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

"""

import torch
import torch.nn.functional as F

from dataclasses import dataclass

import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc import nki
from neuronxcc.nki.language import par_dim
from torch_neuronx import nki_jit


@dataclass(frozen=True)
class FlashConfig:
    """
    Config class for flash attention with default values
    """

    seq_tile_size: int = 2048
    training: bool = False
    should_transpose_v: bool = False

    __annotations__ = {
        "seq_tile_size": int,
        "training": bool,
        "should_transpose_v": bool,
    }


@nki.jit
def transpose_p_local(p_local_transposed,
                      p_local,
                      LARGE_TILE_SZ,
                      forward_mask,
                      B_F_SIZE=512):
    for i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
        if nisa.get_nc_version() == nisa.nc_version.gen3:
            p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE),
                                       buffer=nl.sbuf,
                                       dtype=p_local.dtype)
        else:
            p_local_t_tmp = nl.ndarray((par_dim(128), B_F_SIZE),
                                       buffer=nl.psum,
                                       dtype=np.float32)

        for j in nl.affine_range(B_F_SIZE // 128):
            j_128_slice = nl.ds(j * 128, 128)
            i_j_128_slice = nl.ds(i * B_F_SIZE + j * 128, 128)

            if nisa.get_nc_version() == nisa.nc_version.gen3:
                p_local_t_tmp[:, j_128_slice] = nisa.dma_transpose(
                    p_local[:, i_j_128_slice], mask=forward_mask)
            else:
                p_local_t_tmp[:, j_128_slice] = nisa.nc_transpose(
                    p_local[:, i_j_128_slice], mask=forward_mask)

        p_local_transposed[:, nl.ds(i * B_F_SIZE, B_F_SIZE)] = nl.copy(
            p_local_t_tmp, dtype=p_local_transposed.dtype, mask=forward_mask)


@nki.jit
def _flash_attention_core(
    q_local_tile,
    k,
    v,
    q_h_per_k_h,
    seqlen_q,
    nheads,
    o_buffer,
    l_buffer,
    m_buffer,
    batch_id,
    head_id,
    gqa_head_idx,
    q_tile_idx,
    local_k_large_tile_idx,
    kernel_dtype,
    acc_type,
    flash_config: FlashConfig,
    use_causal_mask=False,
    continuous_batching_mask=None,
    initialize=False,
    B_P_SIZE=128,
    B_F_SIZE=512,
    B_D_SIZE=128,
    dropout_p=0.0,
    dropout_p_tensor=None,
    seed_tensor=None,
    logit_bias_tile=None,
    qk_res_buffer=None,
):
    """
    The flash attention core function to calculate self attention between a tile
    of q and a block of K and V.
    The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF 
    already. The block size of K and V
    is defined in the seq_tile_size of the flash_config. The results are stored
    in the following three buffers
    o_buffer: (B_P_SIZE, d)
    l_buffer: (B_P_SIZE, 1)
    m_buffer: (B_P_SIZE, 1)
    """
    LARGE_TILE_SZ = flash_config.seq_tile_size
    num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
    seqlen_k = k.shape[-1]
    seqlen_q // B_P_SIZE
    seqlen_k // B_F_SIZE

    # TODO : support logit_bias with continuous_batching_mask
    assert not use_causal_mask, "causal mask is not supported."
    assert (continuous_batching_mask
            is not None), "continuous_batching_mask input is required."
    if continuous_batching_mask is not None:
        assert (logit_bias_tile is
                None), "continuous_batching_mask does not support logit_bias!"

    # mask are used to only apply computation to the lower half of the matrix,
    # which reduce the arthimetic intensity by half
    forward_mask = (q_tile_idx * B_P_SIZE >= local_k_large_tile_idx *
                    LARGE_TILE_SZ if use_causal_mask else None)

    qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                            buffer=nl.sbuf,
                            dtype=acc_type)
    max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile),
                           dtype=acc_type)
    # nl.device_print("_flash_attention_core: value of q ",q_local_tile)
    # nl.device_print("_flash_attention_core: value of k ",k)
    # print(f"num_k_tile_per_large_tile {num_k_tile_per_large_tile}")
    for k_i in nl.affine_range(num_k_tile_per_large_tile):
        k_i_b_f_slice = nl.ds(k_i * B_F_SIZE, B_F_SIZE)

        qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                           dtype=np.float32,
                           buffer=nl.psum)  # (128, 512)
        qk_psum[:, :] = nl.matmul(q_local_tile,
                                  k[:, k_i_b_f_slice],
                                  transpose_x=True,
                                  mask=None)  # (p(128), 512)

        qk_res_buf[:, k_i_b_f_slice] = nl.where(
            continuous_batching_mask[:, k_i_b_f_slice],
            qk_psum[:, nl.ds(0, B_F_SIZE)],
            -9984.0,
            dtype=acc_type,
        )
        # nl.device_print("_flash_attention_core: qk_psum ",qk_psum)
        # nl.device_print("_flash_attention_core: continuous_batching_mask",continuous_batching_mask)
        # nl.device_print("_flash_attention_core: qk_res_buf ",qk_res_buf)

        # Calculate max of the current tile
        max_local[:, k_i] = nisa.tensor_reduce(
            np.max,
            qk_res_buf[:, k_i_b_f_slice],
            axis=(1, ),
            dtype=acc_type,
            negate=False,
            mask=forward_mask,
        )

    if qk_res_buffer is not None:
        qk_res_buffer[:, :] = nl.copy(qk_res_buf[:, :])

    max_ = nisa.tensor_reduce(
        np.max,
        max_local[:, :],
        axis=(1, ),
        dtype=acc_type,
        negate=False,
        mask=forward_mask,
    )

    o_previous_scaled = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE),
                                   dtype=o_buffer.dtype)

    if initialize:
        m_buffer[:, 0] = nl.copy(max_)
        m_current = max_
    else:
        m_previous = nl.copy(m_buffer[:, 0])
        m_buffer[:, 0] = nl.maximum(m_previous, max_,
                                    mask=forward_mask)  # (128,1)

        m_current = m_buffer[:, 0]
        # Compute scaling factor
        alpha = nisa.activation(
            np.exp,
            m_previous,
            bias=-1 * m_current,
            scale=1.0,
            mask=forward_mask,
        )
        o_previous_scaled[...] = nl.multiply(o_buffer[:, :],
                                             alpha,
                                             mask=forward_mask)

    p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                         dtype=kernel_dtype)
    REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)

    p_partial_sum = nl.ndarray(
        (par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)

    # print(f"LARGE_TILE_SIZ is {LARGE_TILE_SZ}")
    # print(f"REDUCTION_TILE is {REDUCTION_TILE}")
    for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
        k_r_i_reduce_slice = nl.ds(k_r_i * REDUCTION_TILE, REDUCTION_TILE)

        # compute exp(qk - max)
        # Compute partial row - tile sum of exp(qk - max))
        # FIXME : Use activation accumulate to accumulate over k_r_i loop ?
        p_local[:, k_r_i_reduce_slice] = nisa.activation_reduce(
            np.exp,
            qk_res_buf[:, k_r_i_reduce_slice],
            bias=-1 * m_current,
            scale=1.0,
            reduce_op=nl.add,
            reduce_res=p_partial_sum[:, k_r_i],
            dtype=kernel_dtype,
            mask=forward_mask,
        )

    ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)

    p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                                    dtype=kernel_dtype)
    transpose_p_local(
        p_local_transposed=p_local_transposed,
        p_local=p_local,
        LARGE_TILE_SZ=LARGE_TILE_SZ,
        forward_mask=forward_mask,
        B_F_SIZE=B_F_SIZE,
    )

    # nl.device_print("_flash_attention_core: p_local ",p_local)

    pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE),
                       dtype=np.float32,
                       buffer=nl.psum)
    for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
        pv_psum[:, :] += nl.matmul(
            p_local_transposed[:, nl.ds(k_i * B_P_SIZE, B_P_SIZE)],
            v[k_i, :, :],
            transpose_x=True,
            mask=forward_mask,
        )  # (128, 128) (p(Br), d)

    if initialize:
        o_buffer[:, :] = nl.copy(pv_psum[:, :])
        l_buffer[:, 0] = nl.add(nl.log(ps), max_)
    else:
        o_buffer[:, :] = nl.add(o_previous_scaled, pv_psum, mask=forward_mask)

        l_prev = l_buffer[:, 0]
        l_exp = nl.add(
            nl.exp(
                nl.subtract(l_prev, m_current, mask=forward_mask),
                mask=forward_mask,
            ),
            ps,
            mask=forward_mask,
        )
        l_buffer[:, 0] = nl.add(m_current,
                                nl.log(l_exp, mask=forward_mask),
                                mask=forward_mask)
    # nl.device_print("_flash_attention_core: o_buffer",o_buffer)
    # nl.device_print("_flash_attention_core: m_buffer",m_buffer)
    # nl.device_print("_flash_attention_core: l_buffer",l_buffer)


@nki.jit
def load_v_tile(v_hbm_tile, cur_v_tile, j, v_i, config):
    LARGE_TILE_SZ = config.seq_tile_size
    B_P_SIZE = 128

    if not config.should_transpose_v:
        cur_v_tile[v_i, :, :] = nl.load(
            v_hbm_tile[nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE), :],
            dtype=cur_v_tile.dtype,
        )
        return

    if nisa.get_nc_version() == nisa.nc_version.gen3:
        cur_v_tile_transposed = nisa.dma_transpose(
            v_hbm_tile[:,
                       nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)])
        cur_v_tile[v_i, :, :] = nisa.tensor_copy(cur_v_tile_transposed,
                                                 dtype=cur_v_tile.dtype)
        return

    cur_v_tile[v_i, :, :] = nl.load_transpose2d(
        v_hbm_tile[:, nl.ds(j * LARGE_TILE_SZ + B_P_SIZE * v_i, B_P_SIZE)],
        dtype=cur_v_tile.dtype,
    )

@nki.jit
def flash_attention(
    query,
    key,
    value,
    mask,
    softmax_scale=None,
    mixed_precision=True,
    config=None,
    return_debug_tensors=False,
):
    """
    Flash Attention Forward Kernel.
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape   (1, n_heads, d, seq_q)
      - key:   shape   (1, n_kv_heads, d, seq_k)
      - value: shape   (1, n_kv_heads, seq_v, d)
      - mask: (seq_q, num_active_blocks * block_size)
      - o: shape (1, n_heads, seq_q, d)
      - l_m: shape (1, n_heads, seq_q, 2)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is
        always 1, and different requests are concatenated along sequence
        dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for 
        block_tables (int32) and mask (int32)
      - If mixed_percision is True, then all Tensor Engine operation will be 
        performed in bfloat16 and accumulation will be performed in float32. 
        Otherwise the intermediates will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default
        is set to `true`, if false, we use same precision as input types 
      - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig`
          with Performance config parameters for flash attention with default
          values
        seq_tile_size: `default=2048`, size of the kv tile size for attention 
          computation reduction
        training: bool to indicate training vs inference `default=True`

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of 
      nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    config = config or FlashConfig()
    B_F_SIZE = 512
    B_P_SIZE = 128
    b, h, d, seqlen_q = query.shape
    B_D_SIZE = d
    LARGE_TILE_SZ = config.seq_tile_size
    n_tile_q = seqlen_q // B_P_SIZE  # since q will be loaded on tensor engine
    b, k_h, d, seqlen_q  = key.shape
    q_h_per_k_h = h // k_h
    assert b == 1, f"invalid batch size {b=}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

    o = nl.ndarray((b, h, seqlen_q, d),
                   dtype=query.dtype,
                   buffer=nl.shared_hbm)
    hbm_l_buffer, hbm_m_buffer, hbm_qk_res, qk_res_buffer = (
        None,
        None,
        None,
        None,
    )
    if return_debug_tensors:
        hbm_l_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_m_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_qk_res = nl.ndarray((b, h, B_P_SIZE, seqlen_q),
                                dtype=acc_type,
                                buffer=nl.shared_hbm)
        qk_res_buffer = nl.zeros(
            (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), seqlen_q),
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=True,
        )

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)
    # nl.device_print("head id is ", head_id)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    context_kv_len = LARGE_TILE_SZ
    # print("context_kv_len 2048 is ",context_kv_len)
    assert (config.seq_tile_size >= 512
            ), f" seq tile_size {config.seq_tile_size} cannot be less than 512"
    assert (
        LARGE_TILE_SZ % B_P_SIZE == 0
    ), f"Need LARGE_TILE_SZ ({LARGE_TILE_SZ}) to be divisible by {B_P_SIZE=}"

    # inference flag
    assert not config.training, "flash_paged_attention supports inference only."

    # Global Flash Attention accumulators
    o_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), d),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    l_buffer = nl.zeros(
        (par_dim(B_P_SIZE), n_tile_q, q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    m_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )

    # compute attention between input query, key and value
    if key is not None and value is not None:
        B_F_SIZE = seqlen_q
        LARGE_TILE_SZ = seqlen_q
        active_config = FlashConfig(
            seq_tile_size=LARGE_TILE_SZ,
            training=False,
            should_transpose_v=config.should_transpose_v,
        )

        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ),
                                dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE),
            dtype=kernel_dtype,
        )

        cur_k_tile[:, :] = nl.load(key[batch_id, head_id, :, :])

        load_tile_size = B_P_SIZE
        # v_hbm_tile: (seq_v, d)
        v_hbm_tile = value[batch_id, head_id]
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
            # FIXME AOYU copy same value to different cur_v_tile
            cur_v_tile[v_i, :, :] = nl.load_transpose2d(
                v_hbm_tile,
                dtype=cur_v_tile.dtype,
            )
            # load_v_tile(
            #     v_hbm_tile=v_hbm_tile,
            #     cur_v_tile=cur_v_tile,
            #     j=0,
            #     v_i=v_i,
            #     config=active_config,
            # )

        cur_mask = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), dtype=mask.dtype)
        cur_mask[:, :] = nl.load(mask[:, nl.ds(context_kv_len, B_F_SIZE)])

        for i_q_h in nl.affine_range(q_h_per_k_h):
            for i in nl.affine_range(n_tile_q):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(
                    q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                    dtype=kernel_dtype,
                )  # load (d, 128) tile in SBUF
                q_tile[:, :] = q_sbuf_tile * softmax_scale
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    q_h_per_k_h=q_h_per_k_h,
                    seqlen_q=seqlen_q,
                    nheads=h,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[:, i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    batch_id=batch_id,
                    head_id=head_id,
                    gqa_head_idx=i_q_h,
                    q_tile_idx=i,
                    local_k_large_tile_idx=0,
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    flash_config=active_config,
                    use_causal_mask=False,
                    continuous_batching_mask=cur_mask,
                    initialize=True,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                    dropout_p=0.0,
                    dropout_p_tensor=None,
                    seed_tensor=None,
                    logit_bias_tile=None,
                    qk_res_buffer=qk_res_buffer[i, i_q_h]
                    if qk_res_buffer is not None else None,
                )

    # -- -- -- -- write output to buffer on HBM -- -- -- -- -- -- #
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i in nl.affine_range(n_tile_q):
            out = nl.multiply(
                o_buffer[i, i_q_h, :, :],
                nl.exp(m_buffer[i, i_q_h, :, :] - l_buffer[:, i, i_q_h]),
                dtype=kernel_dtype,
            )

            nl.store(
                o[batch_id, head_id * q_h_per_k_h + i_q_h,
                  nl.ds(i * B_P_SIZE, B_P_SIZE), :, ],
                out,
            )
            # maximum and summation statistics
            if return_debug_tensors:
                nl.store(
                    hbm_m_buffer[batch_id, head_id * q_h_per_k_h + i_q_h,
                                 nl.ds(i * B_P_SIZE, B_P_SIZE), ],
                    m_buffer[i, i_q_h, :, :],
                )
                nl.store(
                    hbm_l_buffer[batch_id, head_id * q_h_per_k_h + i_q_h,
                                 nl.ds(i * B_P_SIZE, B_P_SIZE), ],
                    l_buffer[:, i, i_q_h],
                )
                nl.store(
                    hbm_qk_res[batch_id, head_id * q_h_per_k_h + i_q_h, :, :],
                    qk_res_buffer[batch_id, i_q_h, :, :],
                )

    if return_debug_tensors:
        return o, hbm_m_buffer, hbm_l_buffer, hbm_qk_res
    return o

@nki.jit
def flash_paged_attention(
    query,
    key,
    value,
    key_cache,
    value_cache,
    block_tables,
    mask,
    softmax_scale=None,
    mixed_precision=True,
    config=None,
    return_debug_tensors=False,
):
    """
    Flash PagedAttention Forward Kernel.
      - PagedAttention Paper: https://arxiv.org/abs/2309.06180
      - Chunked Prefill Paper: https://arxiv.org/abs/2403.02310

    IO tensor layouts:
      - query: shape   (1, n_heads, d, seq_q)
      - key:   shape   (1, n_kv_heads, d, seq_k)
      - value: shape   (1, n_kv_heads, seq_v, d)
      - key_cache: (num_blocks, block_size, n_kv_heads, d)
      - value_cache: (num_blocks, block_size, n_kv_heads, d)
      - block_tables: (num_active_blocks, )
      - mask: (seq_q, num_active_blocks * block_size)
      - o: shape (1, n_heads, seq_q, d)
      - l_m: shape (1, n_heads, seq_q, 2)

      - This kernel requires seq_k == seq_v
      - We use continuous batching by default, so the batch dimension is
        always 1, and different requests are concatenated along sequence
        dimension.
      - We use paged cache blocks (key_cache, value_cache) to store KV cache.

    IO tensor dtypes:
      - This kernel assumes all IO tensors have the same dtype except for 
        block_tables (int32) and mask (int32)
      - If mixed_percision is True, then all Tensor Engine operation will be 
        performed in bfloat16 and accumulation will be performed in float32. 
        Otherwise the intermediates will be in the same type as the inputs.

    Compile-time Constants:
      - softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
      - mixed_precision: flag to set non-matmul ops in fp32 precision, default
        is set to `true`, if false, we use same precision as input types 
      - config: Instance of dataclass :class:`nki.kernels.attention.FlashConfig`
          with Performance config parameters for flash attention with default
          values
        seq_tile_size: `default=2048`, size of the kv tile size for attention 
          computation reduction
        training: bool to indicate training vs inference `default=True`

    GQA support Notes:
      the spmd kernel for launching kernel should be on kv_heads instead of 
      nheads

    Example usage:
      MHA: q: [b, h, d, s], k: [b, h, d, s], v: [b, h, s, d]
        usage: `flash_fwd[b, h](q, k, v, ...)`
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s], v: [b, kv_h, s, d]
        usage: `flash_fwd[b, kv_h](q, k, v, ...)`
    """
    config = config or FlashConfig()
    B_F_SIZE = 512
    B_P_SIZE = 128
    b, h, d, seqlen_q = query.shape
    B_D_SIZE = d
    LARGE_TILE_SZ = config.seq_tile_size
    n_tile_q = seqlen_q // B_P_SIZE  # since q will be loaded on tensor engine
    num_blocks, block_size, k_h, _ = key_cache.shape
    q_h_per_k_h = h // k_h
    seq_q, num_active_blocks_block_size  = mask.shape
    assert tuple(key_cache.shape) == (
        num_blocks,
        block_size,
        k_h,
        d,
    ), f"Input shape mismatch! key_cache.shape {key_cache.shape} v.s. (num_blocks, block_size, k_h, d): ({num_blocks}, {block_size}, {k_h}, {d})"
    assert tuple(value_cache.shape) == (
        num_blocks,
        block_size,
        k_h,
        d,
    ), f"Input shape mismatch! value_cache.shape {value_cache.shape} v.s. (num_blocks, block_size, k_h, d): ({num_blocks}, {block_size}, {k_h}, {d})"
    assert b == 1, f"invalid batch size {b=}"
    assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
    kernel_dtype = nl.bfloat16 if mixed_precision else query.dtype
    acc_type = np.dtype(np.float32) if mixed_precision else kernel_dtype

    o = nl.ndarray((b, h, seqlen_q, d),
                   dtype=query.dtype,
                   buffer=nl.shared_hbm)
    hbm_l_buffer, hbm_m_buffer, hbm_qk_res, qk_res_buffer = (
        None,
        None,
        None,
        None,
    )
    if return_debug_tensors:
        hbm_l_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_m_buffer = nl.ndarray((b, h, seqlen_q),
                                  dtype=acc_type,
                                  buffer=nl.shared_hbm)
        hbm_qk_res = nl.ndarray((b, h, B_P_SIZE, seqlen_q),
                                dtype=acc_type,
                                buffer=nl.shared_hbm)
        qk_res_buffer = nl.zeros(
            (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), seqlen_q),
            dtype=acc_type,
            buffer=nl.sbuf,
            lazy_initialization=True,
        )

    assert (
        nl.program_ndim() == 2
    ), f"Expect spmd grid with 2 dimensions, got {nl.program_ndim()} instead!"
    batch_id = nl.program_id(axis=0)
    head_id = nl.program_id(axis=1)
    # nl.device_print("head id is ", head_id)

    softmax_scale = softmax_scale or (1.0 / (d**0.5))

    # nl.device_print("block_tables is ",block_tables)
    (num_active_blocks, ) = block_tables.shape
    context_kv_len = num_active_blocks * block_size
    assert (context_kv_len % LARGE_TILE_SZ == 0
            ), f"Need {context_kv_len=} to be divisible by {LARGE_TILE_SZ=}, mask shape {mask.shape}, block_tables shape {block_tables.shape}, cache size {key_cache.shape}"
    num_large_k_tile = context_kv_len // LARGE_TILE_SZ
    # print("context_kv_len 2048 is ",context_kv_len)
    assert (config.seq_tile_size >= 512
            ), f" seq tile_size {config.seq_tile_size} cannot be less than 512"
    assert (
        LARGE_TILE_SZ % B_P_SIZE == 0
    ), f"Need LARGE_TILE_SZ ({LARGE_TILE_SZ}) to be divisible by {B_P_SIZE=}"
    assert (B_P_SIZE % block_size == 0
            ), f"Need B_P_SIZE ({B_P_SIZE}) to be divisible by {block_size=}"
    num_blocks_per_large_tile = LARGE_TILE_SZ // block_size

    # inference flag
    assert not config.training, "flash_paged_attention supports inference only."

    block_tables_sbuf = nl.full((par_dim(B_P_SIZE), num_large_k_tile),
                                0,
                                dtype=np.int32,
                                buffer=nl.sbuf)
    for j in nl.affine_range(num_large_k_tile):
        i_p = nl.arange(num_blocks_per_large_tile)[:, None]
        block_tables_sbuf[i_p, j] = nl.load(
            block_tables[j * num_blocks_per_large_tile + i_p], dtype=np.int32)

    # Global Flash Attention accumulators
    o_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), d),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    l_buffer = nl.zeros(
        (par_dim(B_P_SIZE), n_tile_q, q_h_per_k_h),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )
    m_buffer = nl.zeros(
        (n_tile_q, q_h_per_k_h, par_dim(B_P_SIZE), 1),
        dtype=acc_type,
        buffer=nl.sbuf,
        lazy_initialization=True,
    )

    for j in nl.sequential_range(0, num_large_k_tile):
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ),
                                dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE),
            dtype=kernel_dtype,
        )

        # nl.device_print("key cache ",key_cache)
        # nl.device_print("Need key cache ",key_cache[2,:,head_id,:])
        for k_i in nl.affine_range(num_blocks_per_large_tile):
            block_id = block_tables_sbuf[k_i, j]
            loaded = nl.load(key_cache[block_id, :, head_id, :])
            # nl.device_print("block number ",block_tables_sbuf[k_i, j])
            # nl.device_print("loaded ",loaded)
            cur_k_tile[:, nl.ds(k_i *
                                block_size, block_size)] = nl.transpose(loaded)
        # nl.device_print("cur_k_tile ",cur_k_tile)

        load_tile_size = B_P_SIZE
        num_blocks_per_partition = load_tile_size // block_size
        for partition_idx in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
            for block_in_partition in nl.affine_range(
                    num_blocks_per_partition):
                v_i = (partition_idx * num_blocks_per_partition +
                    block_in_partition)
                # loaded_v: (block_size, d)
                # loaded_v = nl.load(value_cache[block_tables_sbuf[v_i, j], :,
                #                                head_id, :])
                # FIXME block_table_sbuf will trigger compiler error
                loaded_v = nl.load(value_cache[2, :,
                                            head_id, :])
                cur_v_tile[partition_idx,
                        nl.ds(block_in_partition *
                                block_size, block_size), :, ] = loaded_v

        cur_mask = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ),
                            dtype=mask.dtype)
        for m_i in nl.affine_range(LARGE_TILE_SZ // B_F_SIZE):
            cur_mask[:, nl.ds(m_i * B_F_SIZE, B_F_SIZE)] = nl.load(
                mask[:, nl.ds(j * LARGE_TILE_SZ + m_i * B_F_SIZE, B_F_SIZE)])

        for i_q_h in nl.affine_range(q_h_per_k_h):
            for i in nl.affine_range(n_tile_q):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                # debug value
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(
                    q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                    dtype=kernel_dtype,
                )  # load (d, 128) tile in SBUF
                # print(f"softmax_scale is {softmax_scale}")
                q_tile[:, :] = q_sbuf_tile * softmax_scale
                # nl.device_print("value of query tile is ",q_tile)
                # nl.device_print("value of k tile is ",cur_k_tile)

                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    q_h_per_k_h=q_h_per_k_h,
                    seqlen_q=seqlen_q,
                    nheads=h,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[:, i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    batch_id=batch_id,
                    head_id=head_id,
                    gqa_head_idx=i_q_h,
                    q_tile_idx=i,
                    local_k_large_tile_idx=j,
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    flash_config=config,
                    use_causal_mask=False,
                    continuous_batching_mask=cur_mask,
                    initialize=j == 0,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                    dropout_p=0.0,
                    dropout_p_tensor=None,
                    seed_tensor=None,
                    logit_bias_tile=None,
                )

    # compute attention between input query, key and value
    if key is not None and value is not None:
        B_F_SIZE = seqlen_q
        LARGE_TILE_SZ = seqlen_q
        active_config = FlashConfig(
            seq_tile_size=LARGE_TILE_SZ,
            training=False,
            should_transpose_v=config.should_transpose_v,
        )

        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ),
                                dtype=kernel_dtype)
        cur_v_tile = nl.ndarray(
            (LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE),
            dtype=kernel_dtype,
        )

        cur_k_tile[:, :] = nl.load(key[batch_id, head_id, :, :])

        load_tile_size = B_P_SIZE
        v_hbm_tile = value[batch_id, head_id]
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
            # FIXME AOYU copy same value to different cur_v_tile
            cur_v_tile[v_i, :, :] = nl.load_transpose2d(
                v_hbm_tile,
                dtype=cur_v_tile.dtype,
            )
            # load_v_tile(
            #     v_hbm_tile=v_hbm_tile,
            #     cur_v_tile=cur_v_tile,
            #     j=0,
            #     v_i=v_i,
            #     config=active_config,
            # )

        cur_mask = nl.ndarray((par_dim(B_P_SIZE), B_F_SIZE), dtype=mask.dtype)
        cur_mask[:, :] = nl.load(mask[:, nl.ds(context_kv_len, B_F_SIZE)])

        for i_q_h in nl.affine_range(q_h_per_k_h):
            for i in nl.affine_range(n_tile_q):
                q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE), dtype=kernel_dtype)
                q_hbm_tile = query[batch_id, head_id * q_h_per_k_h + i_q_h]
                q_sbuf_tile = nl.load(
                    q_hbm_tile[:, nl.ds(i * B_P_SIZE, B_P_SIZE)],
                    dtype=kernel_dtype,
                )  # load (d, 128) tile in SBUF
                q_tile[:, :] = q_sbuf_tile * softmax_scale
                _flash_attention_core(
                    q_local_tile=q_tile,
                    k=cur_k_tile,
                    v=cur_v_tile,
                    q_h_per_k_h=q_h_per_k_h,
                    seqlen_q=seqlen_q,
                    nheads=h,
                    o_buffer=o_buffer[i, i_q_h],
                    l_buffer=l_buffer[:, i, i_q_h],
                    m_buffer=m_buffer[i, i_q_h],
                    batch_id=batch_id,
                    head_id=head_id,
                    gqa_head_idx=i_q_h,
                    q_tile_idx=i,
                    local_k_large_tile_idx=0,
                    kernel_dtype=kernel_dtype,
                    acc_type=acc_type,
                    flash_config=active_config,
                    use_causal_mask=False,
                    continuous_batching_mask=cur_mask,
                    initialize=False,
                    B_P_SIZE=B_P_SIZE,
                    B_F_SIZE=B_F_SIZE,
                    B_D_SIZE=B_D_SIZE,
                    dropout_p=0.0,
                    dropout_p_tensor=None,
                    seed_tensor=None,
                    logit_bias_tile=None,
                    qk_res_buffer=qk_res_buffer[i, i_q_h]
                    if qk_res_buffer is not None else None,
                )

    # -- -- -- -- write output to buffer on HBM -- -- -- -- -- -- #
    for i_q_h in nl.affine_range(q_h_per_k_h):
        for i in nl.affine_range(n_tile_q):
            out = nl.multiply(
                o_buffer[i, i_q_h, :, :],
                nl.exp(m_buffer[i, i_q_h, :, :] - l_buffer[:, i, i_q_h]),
                dtype=kernel_dtype,
            )

            nl.store(
                o[batch_id, head_id * q_h_per_k_h + i_q_h,
                  nl.ds(i * B_P_SIZE, B_P_SIZE), :, ],
                out,
            )
            # maximum and summation statistics
            if return_debug_tensors:
                nl.store(
                    hbm_m_buffer[batch_id, head_id * q_h_per_k_h + i_q_h,
                                 nl.ds(i * B_P_SIZE, B_P_SIZE), ],
                    m_buffer[i, i_q_h, :, :],
                )
                nl.store(
                    hbm_l_buffer[batch_id, head_id * q_h_per_k_h + i_q_h,
                                 nl.ds(i * B_P_SIZE, B_P_SIZE), ],
                    l_buffer[:, i, i_q_h],
                )
                nl.store(
                    hbm_qk_res[batch_id, head_id * q_h_per_k_h + i_q_h, :, :],
                    qk_res_buffer[batch_id, i_q_h, :, :],
                )

    if return_debug_tensors:
        return o, hbm_m_buffer, hbm_l_buffer, hbm_qk_res
    return o


def context_attention_fwd(
    query,
    key,
    value,
    key_cache,
    value_cache,
    block_table,
    attn_mask,
    n_kv_head = None,
    head_size = None,
    B_P_SIZE = 128,
    LARGE_TILE_SZ = 2048,
    simulate=False,
    return_debug_tensors=False,
    mixed_precision=True,
):
    B_P_SIZE = 128
    LARGE_TILE_SZ = 2048
    import neuronxcc.nki as nki
    # FIXME use original head_size for softmax_scale
    softmax_scale = 1.0 / (head_size**0.5)

    _, n_kv_head, head_size, _ = key.shape
    assert value.shape[-1] == head_size, "invalid input shape"
    config = FlashConfig(
        seq_tile_size=LARGE_TILE_SZ,
        training=False,
        should_transpose_v=False,
    )
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_table,
        mask=attn_mask,
        softmax_scale=softmax_scale,
        config=config,
        mixed_precision=mixed_precision,
        return_debug_tensors=return_debug_tensors,
    )
    # print(f"key cache {key_cache} !!!!!!!!!!!!!!!!!")
    # print(f"Need key cache kv_head_id = 0 {key_cache[2,:,0,:]}")
    # print(f"Need key cache kv_head_id = 1 {key_cache[2,:,1,:]}")

    def _run_kernel(func, **kwargs):
        if simulate:
            return nki.simulate_kernel(func, **kwargs)
        else:
            return func(**kwargs)

    if simulate:
        kwargs.update(query=query.cpu().numpy())
        kwargs.update(key=key.cpu().numpy())
        kwargs.update(value=value.cpu().numpy())
        kwargs.update(key_cache=key_cache.cpu().numpy())
        kwargs.update(value_cache=value_cache.cpu().numpy())
        kwargs.update(block_tables=block_table.cpu().numpy())
        kwargs.update(mask=attn_mask.cpu().numpy())

    if return_debug_tensors:
        o, *debug_tensors = _run_kernel(
            flash_paged_attention[1, n_kv_head], **kwargs)
        return o, *debug_tensors
    else:
        # o = _run_kernel(flash_paged_attention[1, n_kv_head], **kwargs)
        o = flash_paged_attention[1, n_kv_head](**kwargs)
        return o

def context_flash_attention_fwd(
    query,
    key,
    value,
    attn_mask,
    head_size = None,
    B_P_SIZE = 128,
    LARGE_TILE_SZ = 2048,
    simulate=False,
    return_debug_tensors=False,
    mixed_precision=True,
):
    B_P_SIZE = 128
    LARGE_TILE_SZ = 2048
    import neuronxcc.nki as nki
    # FIXME use original head_size for softmax_scale
    softmax_scale = 1.0 / (head_size**0.5)

    _, n_kv_head, head_size, _ = key.shape
    assert value.shape[-1] == head_size, "invalid input shape"
    config = FlashConfig(
        seq_tile_size=LARGE_TILE_SZ,
        training=False,
        should_transpose_v=False,
    )
    kwargs = dict(
        query=query,
        key=key,
        value=value,
        mask=attn_mask,
        softmax_scale=softmax_scale,
        config=config,
        mixed_precision=mixed_precision,
        return_debug_tensors=return_debug_tensors,
    )

    def _run_kernel(func, **kwargs):
        if simulate:
            return nki.simulate_kernel(func, **kwargs)
        else:
            return func(**kwargs)

    if simulate:
        kwargs.update(query=query.cpu().numpy())
        kwargs.update(key=key.cpu().numpy())
        kwargs.update(value=value.cpu().numpy())
        kwargs.update(mask=attn_mask.cpu().numpy())

    if return_debug_tensors:
        o, *debug_tensors = _run_kernel(
            flash_attention[1, n_kv_head], **kwargs)
        return o, *debug_tensors
    else:
        # o = _run_kernel(flash_paged_attention[1, n_kv_head], **kwargs)
        o = flash_attention[1, n_kv_head](**kwargs)
        return o


def nki_context_attention(
        query, key, value, key_cache, value_cache,
        query_lens, seq_lens,
        max_num_seqs,
        block_size,
        max_model_len,
        simulate = False,
        return_debug_tensors = False,
        mixed_precision = True,
):
    # build neuron program
    B_P_SIZE = 128
    LARGE_TILE_SZ = 2048
    # max_num_seqs = batch_size
    n_blocks = (max_model_len * max_num_seqs) // block_size
    max_num_queries = (
        (sum(query_lens) + block_size - 1) // block_size) * block_size

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
        return F.pad(
            torch.tensor(active_blocks),
            (0, num_blocks - len(active_blocks)),
            "constant",
            0,
        )

    def shift_bit_length(x):
        return 1 << (x - 1).bit_length()

    # Build and pad input tensors
    max_num_queries_shifted = shift_bit_length(max_num_queries)
    max_num_queries_factor = B_P_SIZE // max_num_queries_shifted
    max_num_queries_padded = max_num_queries_shifted * max_num_queries_factor
    assert (max_num_queries_padded == B_P_SIZE
            ), "invalid {max_num_queries_padded=}"
    head_size_padded = B_P_SIZE
    context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
    num_active_blocks_shifted = shift_bit_length(
        ((context_lens + block_size - 1) // block_size).sum().item())
    num_active_blocks_factor = (LARGE_TILE_SZ // block_size //
                                num_active_blocks_shifted)
    num_active_blocks = num_active_blocks_shifted * num_active_blocks_factor
    assert (num_active_blocks *
            block_size) == LARGE_TILE_SZ, "invalid {num_active_blocks=}"
    pad_dims = (
        0,
        head_size_padded - query.shape[2],
        0,
        0,
        0,
        max_num_queries_padded - query.shape[0],
    )
    query = F.pad(query, pad_dims, "constant", 0)
    k = F.pad(key, pad_dims, "constant", 0)
    v = F.pad(value, pad_dims, "constant", 0)
    output_ref_padded = F.pad(
        output_ref,
        (0, 0, 0, 0, 0, 0, 0, max_num_queries_padded - output_ref.shape[0]),
        "constant",
        0,
    )
    active_block_table = get_active_block_tables(
        block_table,
        torch.tensor(query_lens),
        torch.tensor(seq_lens),
        block_size,
        num_active_blocks,
    )
    k_cache = F.pad(k_cache, (0, head_size_padded - head_size), "constant", 0)
    v_cache = F.pad(v_cache, (0, head_size_padded - head_size), "constant", 0)

    # Build attention masks
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    prior_mask, active_mask = (
        BlockDiagonalCausalFromBottomRightMask.from_seqlens(
            query_lens, seq_lens, block_size=block_size))
    context_kv_len = num_active_blocks * block_size
    assert context_kv_len == LARGE_TILE_SZ, f"invalid {context_kv_len=}"
    attn_mask = torch.concat(
        [
            F.pad(
                prior_mask,
                (
                    0,
                    context_kv_len - prior_mask.shape[1],
                    0,
                    B_P_SIZE - prior_mask.shape[0],
                ),
                "constant",
                0,
            ).bool(),
            F.pad(
                active_mask,
                (
                    0,
                    B_P_SIZE - active_mask.shape[1],
                    0,
                    B_P_SIZE - active_mask.shape[0],
                ),
                "constant",
                0,
            ).bool(),
        ],
        dim=1,
    )

    example = (
        # query: shape (1, n_heads, d, seq_q)
        # key:   shape (1, n_kv_heads, d, seq_k)
        # value: shape (1, n_kv_heads, seq_v, d)
        query.unsqueeze(1).transpose(0, 1).permute(
            0, 2, 3, 1).contiguous().to(device=device),
        k.unsqueeze(1).transpose(0,
                                 1).permute(0, 2, 3,
                                            1).contiguous().to(device=device),
        v.unsqueeze(1).transpose(0,
                                 1).permute(0, 2, 1,
                                            3).contiguous().to(device=device),
        k_cache.to(device=device),
        v_cache.to(device=device),
        active_block_table.to(torch.int32).to(device=device),
        attn_mask.to(device=device),
    )

    if return_debug_tensors:
        output_nki, *debug_tensors = context_attention_fwd(*example,
                                                           simulate=simulate)
    else:
        output_nki = context_attention_fwd(*example, simulate=simulate)
        debug_tensors = []
    if simulate:
        output_nki = torch.tensor(output_nki)
        debug_tensors = [torch.tensor(dt) for dt in debug_tensors]
    else:
        output_nki = torch.tensor(output_nki).cpu()
        debug_tensors = [torch.tensor(dt).cpu() for dt in debug_tensors]

    n_queries = sum(query_lens)

    # - o: shape (bs, n_heads, seq_q, d) -> (bs, seq_q, n_heads, d)
    output_nki = output_nki.permute(
        0, 2, 1, 3)[:, :, :, :head_size].cpu()[0, :n_queries, :, :]

    return output_nki