# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F
from neuronxcc import nki

from vllm.attention.ops.nki_flash_attn import (
    nki_load_and_transform_block_tables)


def ref_block_tables_transform(
    block_tables,
    num_tiles,
    num_blocks_per_tile,
    num_head,
    head_id,
    block_size_tiling_factor,
):
    assert block_tables.numel() == num_tiles * num_blocks_per_tile
    block_tables = block_tables.view(num_tiles, num_blocks_per_tile)
    B_F_SIZE = 128
    num_tiles_padded = (num_tiles + B_F_SIZE - 1) // B_F_SIZE * B_F_SIZE
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
    assert num_blocks_per_tile % B_F_SIZE == 0
    return block_tables_transposed.view(num_blocks_per_tile // B_F_SIZE,
                                        B_F_SIZE, num_tiles_padded)


@pytest.mark.parametrize(
    "q_head_per_kv_head,head_id",
    [
        # (1, 0),
        (3, 1),
    ],
)
@pytest.mark.parametrize(
    "num_tiles,num_blocks_per_tile",
    [
        # (1, 1),
        # (13, 16),
        # (17, 128),
        # (35, 512),
        # (128, 128),
        # (130, 64),
        (280, 256),
        # (315, 1),
    ],
)
@torch.inference_mode()
def test_load_and_transform_block_tables(
    monkeypatch: pytest.MonkeyPatch,
    num_tiles,
    num_blocks_per_tile,
    q_head_per_kv_head,
    head_id,
) -> None:
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()

    compiler_flags_str = " ".join([
        "-O1",
        "--retry_failed_compilation",
    ])
    with monkeypatch.context() as m:
        m.setenv("NEURON_CC_FLAGS", compiler_flags_str)

        torch.manual_seed(10000)
        torch.set_printoptions(sci_mode=False)

        # On Neuron, we need B_P_SIZE = 128 blocks to make DMA efficient
        B_P_SIZE = 128
        if num_blocks_per_tile < B_P_SIZE:
            assert B_P_SIZE % num_blocks_per_tile == 0
            block_size_tiling_factor = B_P_SIZE // num_blocks_per_tile
        else:
            block_size_tiling_factor = 1
        max_num_blocks = 100000
        block_tables = torch.randint(
            0,
            max_num_blocks,
            (num_tiles * num_blocks_per_tile, ),
            dtype=torch.int32,
        )
        nki_out = nki.jit(nki_load_and_transform_block_tables)[1, 1](
            block_tables.to(device=device),
            num_tiles,
            num_blocks_per_tile,
            q_head_per_kv_head,
            head_id,
            block_size_tiling_factor,
        ).cpu()
        ref_out = ref_block_tables_transform(
            block_tables,
            num_tiles,
            num_blocks_per_tile,
            q_head_per_kv_head,
            head_id,
            block_size_tiling_factor,
        )
        assert (nki_out.shape == ref_out.shape
                ), f"{nki_out.shape=} != {ref_out.shape=}"
        assert torch.all(nki_out == ref_out)
