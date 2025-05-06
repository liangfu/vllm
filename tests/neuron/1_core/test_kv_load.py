# SPDX-License-Identifier: Apache-2.0

import os

import neuronxcc.nki.language as nl
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from neuronxcc import nki

from vllm.utils import cdiv, round_up
from vllm.attention.ops.nki_flash_attn import (
    load_active_block_tables, transform_block_tables_for_indirect_load,
    load_kv_tile_from_cache)

B_P_SIZE = 128

def is_power_of_2(n):
    return n > 0 and (n & (n - 1) == 0)


def nki_load_and_transform_block_tables(
    block_tables,
    context_lens,
    block_size,
    num_tiles,
    num_blocks_per_tile,
    num_head,
    head_id,
    block_size_tiling_factor,
):
    assert is_power_of_2(
        num_blocks_per_tile), f"{num_blocks_per_tile=} must be power of 2"
    block_tables_sbuf = load_active_block_tables(
        block_tables, context_lens, block_size, num_tiles, num_blocks_per_tile)

    # we need to pass an Index as head_id
    head_id = nl.arange(1)[None, :] + head_id

    block_tables_transposed = transform_block_tables_for_indirect_load(
        block_tables_sbuf,
        block_size_tiling_factor, num_head, head_id)
    assert block_tables_transposed.shape[1] == B_P_SIZE

    out = nl.ndarray(
        block_tables_transposed.shape,
        dtype=nl.int32,
        buffer=nl.shared_hbm,
    )
    for i in nl.affine_range(block_tables_transposed.shape[0]):
        nl.store(dst=out[i], value=block_tables_transposed[i])
    return out


def ref_block_tables_transform(
    block_tables,
    context_lens,
    block_size,
    num_tiles,
    num_blocks_per_tile,
    num_head,
    head_id,
    block_size_tiling_factor,
):
    assert block_tables.numel() == num_tiles * num_blocks_per_tile
    block_tables = block_tables.view(num_tiles, num_blocks_per_tile)
    num_tiles_padded = round_up(num_tiles, B_P_SIZE)
    block_tables = F.pad(
        block_tables,
        (0, 0, 0, num_tiles_padded - num_tiles),
        "constant",
        0,
    )

    block_tables = block_tables * num_head + head_id
    block_tables = block_tables.view(num_tiles_padded, num_blocks_per_tile, 1)
    offset = torch.arange(0, block_size_tiling_factor).view(1, 1, -1)
    block_tables = block_tables * block_size_tiling_factor + offset
    block_tables_transposed = block_tables.view(num_tiles_padded, -1).t()

    num_blocks_per_tile = block_tables_transposed.shape[0]
    assert num_blocks_per_tile % B_P_SIZE == 0
    return block_tables_transposed.view(num_blocks_per_tile // B_P_SIZE,
                                        B_P_SIZE, num_tiles_padded)


# @nki.compiler.skip_middle_end_transformations
@nki.jit
def nki_kv_load(
    key_cache,
    value_cache,
    block_tables,
    kernel_dtype,
):
    B_P_SIZE = 128
    kernel_dtype = getattr(nl, kernel_dtype)
    num_blocks, num_head, block_size, B_D_SIZE = key_cache.shape
    assert value_cache.shape == key_cache.shape
    assert value_cache.dtype == key_cache.dtype
    assert block_tables.dtype == nl.int32

    num_tiles, num_blocks_per_tile = block_tables.shape
    assert is_power_of_2(num_blocks_per_tile), f"{num_blocks_per_tile=} must be power of 2"
    block_tables_sbuf = load_active_block_tables(block_tables.reshape(shape=(num_tiles*num_blocks_per_tile,)), num_tiles, num_blocks_per_tile)
    LARGE_KV_TILE_SIZE = num_blocks_per_tile * block_size

    if num_blocks_per_tile < B_P_SIZE:
        assert B_P_SIZE % num_blocks_per_tile == 0
        block_size_tiling_factor = B_P_SIZE // num_blocks_per_tile
        assert block_size % block_size_tiling_factor == 0
        num_blocks_per_tile *= block_size_tiling_factor
    else:
        block_size_tiling_factor = 1
    tiled_block_size = block_size // block_size_tiling_factor

    # we need to pass an Index as head_id
    head_id = nl.program_id(axis=0)

    block_tables_transposed = transform_block_tables_for_indirect_load(
        block_tables_sbuf, block_size_tiling_factor, num_head, head_id
    )
    assert block_tables_transposed.shape[0] == B_P_SIZE

    identity_hbm = nl.shared_constant(
        np.identity(n=B_P_SIZE, dtype=np.uint8),
        dtype=key_cache.dtype,
    )
    identity_for_transpose_k = nl.load(identity_hbm)

    k_out = nl.ndarray(
        (num_head, num_tiles, B_D_SIZE, LARGE_KV_TILE_SIZE),
        dtype=key_cache.dtype,
        buffer=nl.shared_hbm,
    )
    num_loads = cdiv(num_blocks_per_tile, B_P_SIZE)
    v_out = nl.ndarray(
        (num_head, num_tiles, B_P_SIZE, num_loads * tiled_block_size * B_D_SIZE),
        dtype=value_cache.dtype,
        buffer=nl.shared_hbm,
    )
    new_cache_shape = (
        num_blocks * num_head * block_size_tiling_factor,
        tiled_block_size * B_D_SIZE,
    )
    key_cache = key_cache.reshape(new_cache_shape)
    value_cache = value_cache.reshape(new_cache_shape)

    k_buffer = nl.ndarray(
        (nl.par_dim(B_P_SIZE), num_loads, block_size * B_D_SIZE),
        dtype=key_cache.dtype,
    )
    v_buffer = nl.ndarray(
        (nl.par_dim(B_P_SIZE), num_loads * tiled_block_size * B_D_SIZE),
        dtype=value_cache.dtype,
    )
    for large_tile_idx in nl.sequential_range(num_tiles):
        cur_k_tile, cur_v_tile = load_kv_tile_from_cache(
            kv_cache=kv_cache,
            block_tables=block_tables_transposed,
            large_k_tile_idx=large_tile_idx,
            num_blocks_per_large_tile=num_blocks_per_tile,
            block_size=tiled_block_size,
            B_D_SIZE=B_D_SIZE,
            kernel_dtype=kernel_dtype,
            k_load_buffer=k_buffer,
            v_load_buffer=v_buffer,
            # identity_for_transpose=identity_for_transpose_k,
        )
        nl.store(dst=k_out[head_id, large_tile_idx], value=cur_k_tile)
        nl.store(dst=v_out[head_id, large_tile_idx], value=cur_v_tile)

    return k_out, v_out


def ref_gather_kv_blocks(k_cache, v_cache, tile_block_tables):
    block_tables = tile_block_tables.view(-1)
    keys = k_cache[block_tables]
    values = v_cache[block_tables]
    return keys.transpose(0, 1), values.transpose(0, 1)


@pytest.mark.parametrize(
    "num_kv_heads,head_size",
    [
        # (4, 128),
        # (2, 64),
        # (1, 128),
        (8, 64),
    ],
)
@pytest.mark.parametrize(
    "block_size,tile_size_kv",
    [
        (32, 1024),
        # (16, 2048),
        # (32, 4096),
        # (64, 4096),
    ],
)
@pytest.mark.parametrize("batch_size", [15]) # , 34])
@pytest.mark.parametrize(
    "data_dtype,kernel_dtype",
    [
        ("float32", "bfloat16"),
        # ("bfloat16", "bfloat16"),
        # ("float32", "float32"),
    ],
)
@torch.inference_mode()
def test_nki_kv_load(
    batch_size: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    tile_size_kv: int,
    data_dtype: str,
    kernel_dtype: str,
) -> None:
    compiler_flags = [
        "-O1",
        "--retry_failed_compilation",
        # "--internal-compiler-debug-mode=penguin",
        # "--tensorizer-options='--print-stats --dump-after=All'",
        "--enable-internal-data-race-checker",
    ]
    compiler_flags_str = " ".join(compiler_flags)
    os.environ["NEURON_CC_FLAGS"] = compiler_flags_str

    torch.manual_seed(0)
    torch.set_printoptions(sci_mode=False)

    max_model_len = 8192
    assert max_model_len % block_size == 0
    assert max_model_len % tile_size_kv == 0
    B_P_SIZE = 128
    assert tile_size_kv % B_P_SIZE == 0
    data_dtype = getattr(torch, data_dtype)

    max_block_per_request = max_model_len // block_size
    cache_size = (batch_size * max_block_per_request) * 2
    k_cache = torch.empty(cache_size, num_kv_heads, block_size, head_size, dtype=data_dtype)
    v_cache = torch.empty(cache_size, num_kv_heads, block_size, head_size, dtype=data_dtype)
    indices = torch.randperm(cache_size, dtype=torch.int32)
    tile_block_tables = indices[: batch_size * max_block_per_request].view(
        batch_size * max_model_len // tile_size_kv, tile_size_kv // block_size
    )
    k_cache.uniform_(-1, 1)
    v_cache.uniform_(-1, 1)

    # reference version
    k_ref, v_ref = ref_gather_kv_blocks(k_cache, v_cache, tile_block_tables)

    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    k_nki, v_nki = nki_kv_load[num_kv_heads](
        k_cache.to(device),
        v_cache.to(device),
        tile_block_tables.to(device),
        kernel_dtype,
    )

    # reorder back
    tiled_block_size = tile_size_kv // B_P_SIZE
    k_nki = k_nki.cpu()
    k_nki = k_nki.view(
        num_kv_heads,
        batch_size * max_model_len // tile_size_kv,
        head_size,
        tiled_block_size,
        B_P_SIZE,
    )
    k_nki = k_nki.permute(0, 1, 4, 3, 2).contiguous()
    k_nki = k_nki.view(
        num_kv_heads,
        batch_size * max_block_per_request,
        block_size,
        head_size,
    )

    v_nki = v_nki.cpu()
    v_nki = v_nki.view(
        num_kv_heads,
        batch_size * max_block_per_request,
        block_size,
        head_size,
    )

    torch.testing.assert_close(v_nki, v_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(k_nki, k_ref, atol=1e-2, rtol=0)

