import os
import numpy as np
import torch
import torch_xla.core.xla_model as xm
from torch_neuronx.xla_impl.ops import Argmax

import pytest
import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

def cdiv(a, b):
    return (a + b - 1) // b
def round_up(a, b):
    return cdiv(a, b) * b

import uuid
@nki.compiler.skip_middle_end_transformations
# @nki.trace
# @nki.jit(mode="simulation")
# @nki.baremetal(
#     artifacts_dir=f"./_artifact_{str(uuid.uuid4().hex)[:8]}",
#     additional_compile_opt=(
#         " -O1 --model-type=transformer --lnc=1 "
#         " --enable-internal-data-race-checker "
#         " --internal-compiler-debug-mode=penguin "),
#     debug_kernel=True,
#     show_compiler_tb=True,
# )
@nki.jit(debug_kernel=True)
def nki_argmax(
    v1,
):
    batch_size, vocab_size = v1.shape
    tile_size = 2048
    n_tiles = cdiv(vocab_size, tile_size)
    print(f"{n_tiles=}")
    dtype = v1.dtype

    v2 = nl.ndarray((batch_size, 1), dtype=np.int32, buffer=nl.shared_hbm)

    v3 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v3", buffer=nl.sbuf)
    v4 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=dtype, name="v4", buffer=nl.sbuf)
    v5 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="v5", buffer=nl.sbuf)
    v6 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="v6", buffer=nl.sbuf)
    v7 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="v7", buffer=nl.sbuf)
    v8 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="v8", buffer=nl.sbuf)
    v9 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.uint8, name="v9", buffer=nl.sbuf)
    v10 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="v10", buffer=nl.sbuf)
    v11 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v11", buffer=nl.sbuf)
    v12 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v12", buffer=nl.sbuf)
    v13 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v13", buffer=nl.sbuf)
    v14 = nl.ndarray((nl.par_dim(batch_size), 1, n_tiles), dtype=dtype, name="v14", buffer=nl.sbuf)
    v15 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="v15", buffer=nl.sbuf)

    v3_hbm = nl.ndarray((batch_size, n_tiles), dtype=np.int32, name="v3_hbm", buffer=nl.shared_hbm)
    v12_hbm = nl.ndarray((batch_size,), dtype=np.int32, name="v12_hbm", buffer=nl.shared_hbm)
    v14_hbm = nl.ndarray((batch_size, n_tiles), dtype=dtype, name="v14_hbm", buffer=nl.shared_hbm)
    v15_hbm = nl.ndarray((batch_size, n_tiles), dtype=np.int32, name="v15_hbm", buffer=nl.shared_hbm)

    i_p = nl.arange(batch_size)[:, None, None]
    i_f = nl.arange(tile_size)[None, None, :]
    iota = nisa.iota(i_f, dtype=np.int32, mask=None)

    for tile_idx in nl.affine_range(n_tiles):
        mask = (tile_idx*tile_size+i_f) < vocab_size
        v4[i_p, tile_idx, i_f] = nisa.memset(shape=(batch_size, 1, tile_size), value=-np.inf, dtype=dtype)
        v4[i_p, tile_idx, i_f] = nl.load(v1[i_p, tile_idx*tile_size+i_f], dtype=dtype, mask=mask)
        v5[i_p, tile_idx, 0] = nisa.tensor_reduce(nl.maximum, data=v4[i_p, tile_idx, i_f], mask=None, axis=[2], dtype=dtype, negate=False)
        v6[i_p, tile_idx, 0] = nisa.tensor_scalar(data=v5[i_p, tile_idx, 0], op0=nl.maximum, operand0=-np.inf, reverse0=False, dtype=dtype, mask=None)
        v7[i_p, tile_idx, i_f] = nl.broadcast_to(iota, shape=(batch_size, 1, tile_size))
        v8[i_p, tile_idx, i_f] = nisa.tensor_scalar(data=v7[i_p, tile_idx, i_f], op0=nl.subtract, operand0=tile_size, reverse0=True, dtype=np.int32, mask=None)
        v9[i_p, tile_idx, i_f] = nisa.tensor_tensor(data1=v4[i_p, tile_idx, i_f], data2=v6[i_p, tile_idx, 0], op=nl.equal)
        v10[i_p, tile_idx, i_f] = nl.multiply(v8[i_p, tile_idx, i_f], v9[i_p, tile_idx, i_f], mask=None, dtype=np.int32)
        v11[i_p, tile_idx, 0] = nisa.tensor_reduce(nl.maximum, data=v10[i_p, tile_idx, i_f], mask=None, axis=[2], dtype=np.int32, negate=False)
        v12[i_p, tile_idx, 0] = nisa.tensor_scalar(data=v11[i_p, tile_idx, 0],  op0=nl.maximum, operand0=-np.inf, reverse0=False, op1=nl.subtract, operand1=tile_size, reverse1=True, dtype=np.int32, mask=None)
        v3[i_p, tile_idx, 0] = nl.copy(v12[i_p, tile_idx, 0], dtype=np.int32, mask=None)
        nl.store(v3_hbm[i_p, tile_idx], value=v3[i_p, tile_idx, 0], mask=None)
        v13[i_p, tile_idx, 0] = nisa.tensor_scalar(data=v7[i_p, 0, tile_idx],  op0=nl.multiply, operand0=tile_size, dtype=np.int32, reverse0=False)
        v15[i_p, tile_idx, 0] = nisa.tensor_tensor(data1=v12[i_p, tile_idx, 0],  data2=v13[i_p, tile_idx, 0], op=nl.add)
        nl.store(v15_hbm[i_p, tile_idx], value=v15[i_p, tile_idx, 0], mask=None)

    for batch_idx in nl.affine_range(batch_size):
        for tile_idx in nl.affine_range(n_tiles):
            v14[batch_idx, 0, tile_idx] = nl.load(v1[batch_idx, v15[batch_idx, tile_idx, 0]], dtype=dtype, mask=None)

    i_b = nl.arange(n_tiles)[None, :, None]
    i_f = nl.arange(n_tiles)[None, None, :]
    nl.store(v14_hbm[i_p, i_f], value=v14[i_p, 0, i_f], mask=None)
    v5[i_p, 0, 0] = nisa.tensor_reduce(nl.maximum, data=v14[i_p, 0, i_f], mask=None, axis=[2], dtype=dtype, negate=False)
    v6[i_p, 0, 0] = nisa.tensor_scalar(data=v5[i_p, 0, 0], op0=nl.maximum, operand0=-np.inf, reverse0=False, dtype=dtype, mask=None)
    v7[i_p, 0, i_f] = nl.broadcast_to(nisa.iota(i_f, dtype=np.int32, mask=None), shape=(batch_size, 1, n_tiles))
    v8[i_p, 0, i_f] = nisa.tensor_scalar(data=v7[i_p, 0, i_f], op0=nl.subtract, operand0=n_tiles, reverse0=True, dtype=np.int32, mask=None)
    v9[i_p, 0, i_f] = nisa.tensor_tensor(data1=v14[i_p, 0, i_f], data2=v6[i_p, 0, 0], op=nl.equal)
    v10[i_p, 0, i_f] = nl.multiply(v8[i_p, 0, i_f], v9[i_p, 0, i_f], mask=None, dtype=np.int32)
    v11[i_p, 0, 0] = nisa.tensor_reduce(nl.maximum, data=v10[i_p, 0, i_f], mask=None, axis=[2], dtype=np.int32, negate=False)
    v12[i_p, 0, 0] = nisa.tensor_scalar(data=v11[i_p, 0, 0],  op0=nl.maximum, operand0=-np.inf, reverse0=False, op1=nl.subtract, operand1=n_tiles, reverse1=True, dtype=np.int32, mask=None)
    nl.store(v12_hbm[i_p], value=v12[i_p, 0, 0], mask=None)

    for batch_idx in nl.affine_range(batch_size):
        nl.store(v2[batch_idx, 0], nl.load(v15_hbm[batch_idx, v12[batch_idx, 0, 0]]))

    #return v2, v3_hbm, v12_hbm, v14_hbm, v15_hbm
    return v2


@pytest.mark.parametrize(
    "dtype,batch_size,vocab_size",
    [
        # (nl.float32, 3, 5001),
        (nl.float32, 8, 32000),
        (nl.bfloat16, 8, 32000),
    ])
# @torch.inference_mode()
def test_argmax(dtype, batch_size, vocab_size):
    # compiler_flags_str = (
    #     " -O1 "
    #     # " --tensorizer-options='--print-nki' "
    #     # " --tensorizer-options='--dump-after=All --dump-nki' "
    #     # " --internal-compiler-debug-mode=penguin "
    #     " --retry_failed_compilation "
    # )
    # os.environ["NEURON_CC_FLAGS"] = compiler_flags_str
    # torch.manual_seed(1000)
    np.random.seed(1000)

    #for tile_size in [101, 201, 401, 801, 1601, 3201, 6401, 12801, 32001]:
    tile_size = 2048

    vocab_size_padded = round_up(vocab_size, tile_size)
    logits_cpu = np.random.randn(batch_size, vocab_size).astype(dtype)
    ref_output = np.argmax(logits_cpu, axis=-1, keepdims=True).astype(np.int32)
    print(f"{ref_output=}, {batch_size=}, {tile_size=}")

    # device = xm.xla_device()
    # logits = logits_cpu.to(device=device)
    # nki_output = Argmax.apply(logits, dim=-1, keepdim=True).cpu()

    nki_output = nki_argmax(logits_cpu)
    print(f"{batch_size=}, {tile_size=}, {nki_output=}, {ref_output.flatten()=}")
    # print(f"{v3=}, {v12=}, {v14=}")
    np.testing.assert_allclose(nki_output.flatten(), ref_output.flatten())
    # import pdb
    # pdb.set_trace()
    # print()
