# SPDX-License-Identifier: Apache-2.0
"""Sampler layer implementing Neuron supported operations."""

import torch
import torch.nn as nn

import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

from vllm.v1.outputs import LogprobsTensors, SamplerOutput
from vllm.v1.sample.neuron.metadata import NeuronSupportedSamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler

_SAMPLING_EPS = 1e-5


def cdiv(a, b):
    return (a + b - 1) // b
def round_up(a, b):
    return cdiv(a, b) * b


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: NeuronSupportedSamplingMetadata,
    ) -> SamplerOutput:
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).

        # Use float32 for the logits.
        logits = logits.to(torch.float32)
        # Sample the next token.
        sampled = self.sample(logits, sampling_metadata)

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=None,
        )
        return sampler_output

    def apply_temperature(
        self,
        logits: torch.Tensor,
        temp: torch.Tensor,
    ) -> torch.Tensor:
        # Use in-place division to avoid creating a new tensor.
        return logits.div_(temp.unsqueeze(dim=1))

    def greedy_sample(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)
        # from torch_neuronx.xla_impl.ops import Argmax
        # return Argmax.apply(logits, dim=-1, keepdim=True)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: NeuronSupportedSamplingMetadata,
    ) -> torch.Tensor:
        greedy_sampled = self.greedy_sample(logits)

        assert sampling_metadata.temperature is not None

        # Apply temperature.
        logits = self.apply_temperature(logits, sampling_metadata.temperature)

        # Apply min_p.
        if sampling_metadata.min_p is not None:
            logits = self.apply_min_p(logits, sampling_metadata.min_p)

        # Apply top_k and/or top_p.
        random_sampled = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )

        sampled = torch.where(sampling_metadata.temperature < _SAMPLING_EPS,
                              greedy_sampled, random_sampled)
        return sampled

    def compute_logprobs(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    def gather_logprobs(
        self,
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logits: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs,
                                                 num_logprobs,
                                                 dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        min_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Filters logits using adaptive probability thresholding.
        """
        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values,
                                       dim=-1,
                                       keepdim=True)
        # Reshape min_p for broadcasting
        adjusted_min_p = min_p.unsqueeze(1) * max_probabilities
        # Identify valid tokens using threshold comparison
        valid_token_mask = probability_values >= adjusted_min_p
        # Apply mask using boolean indexing (xla friendly)
        logits.masked_fill_(~valid_token_mask, -float("inf"))
        return logits


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
@nki.jit
def nki_argmax(
    v1: torch.Tensor
) -> torch.Tensor:
    batch_size, vocab_size = v1.shape
    tile_size = 2048
    n_tiles = cdiv(vocab_size, tile_size)
    dtype = nl.float32

    v2 = nl.ndarray((batch_size, 1), dtype=np.int32, buffer=nl.shared_hbm)

    v3 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="", buffer=nl.sbuf)
    v4 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=dtype, name="custom-call.1.131", buffer=nl.sbuf)
    v5 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="custom-call.1.133", buffer=nl.sbuf)
    v6 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=dtype, name="custom-call.1.135", buffer=nl.sbuf)
    v7 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="custom-call.1.137", buffer=nl.sbuf)
    v8 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="custom-call.1.139", buffer=nl.sbuf)
    v9 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.uint8, name="custom-call.1.141", buffer=nl.sbuf)
    v10 = nl.ndarray((nl.par_dim(batch_size), n_tiles, tile_size), dtype=np.int32, name="custom-call.1.143", buffer=nl.sbuf)
    v11 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="custom-call.1.145", buffer=nl.sbuf)
    v12 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="custom-call.1.147", buffer=nl.sbuf)
    v13 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="custom-call.1.147", buffer=nl.sbuf)
    v14 = nl.ndarray((nl.par_dim(batch_size), 1, n_tiles), dtype=dtype, name="custom-call.1.131", buffer=nl.sbuf)
    v15 = nl.ndarray((nl.par_dim(batch_size), n_tiles, 1), dtype=np.int32, name="custom-call.1.147", buffer=nl.sbuf)

    v3_hbm = nl.ndarray((batch_size, n_tiles), dtype=np.int32, buffer=nl.shared_hbm)
    v12_hbm = nl.ndarray((batch_size,), dtype=np.int32, name="custom-call.1.147", buffer=nl.shared_hbm)
    v14_hbm = nl.ndarray((batch_size, n_tiles), dtype=dtype, name="custom-call.1.131", buffer=nl.shared_hbm)
    v15_hbm = nl.ndarray((batch_size, n_tiles), dtype=np.int32, buffer=nl.shared_hbm)

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
        v9[i_p, tile_idx, i_f] = nisa.tensor_scalar(data=v4[i_p, tile_idx, i_f], op0=nl.equal, operand0=v6[i_p, tile_idx, 0], reverse0=False, dtype=np.uint8, mask=None)
        v10[i_p, tile_idx, i_f] = nl.multiply(v8[i_p, tile_idx, i_f], v9[i_p, tile_idx, i_f], mask=None, dtype=np.int32)
        v11[i_p, tile_idx, 0] = nisa.tensor_reduce(nl.maximum, data=v10[i_p, tile_idx, i_f], mask=None, axis=[2], dtype=np.int32, negate=False)
        v12[i_p, tile_idx, 0] = nisa.tensor_scalar(data=v11[i_p, tile_idx, 0],  op0=nl.maximum, operand0=-np.inf, reverse0=False, op1=nl.subtract, operand1=tile_size, reverse1=True, dtype=np.int32, mask=None)
        v3[i_p, tile_idx, 0] = nl.copy(v12[i_p, tile_idx, 0], dtype=np.int32, mask=None)
        nl.store(v3_hbm[i_p, tile_idx], value=v3[i_p, tile_idx, 0], mask=None)
        v13[i_p, tile_idx, 0] = nisa.tensor_scalar(data=iota[0, 0, tile_idx],  op0=nl.multiply, operand0=tile_size, reverse0=False)
        v15[i_p, tile_idx, 0] = nisa.tensor_scalar(data=v12[i_p, tile_idx, 0],  op0=nl.add, operand0=v13[i_p, tile_idx, 0], reverse0=False)
        nl.store(v15_hbm[i_p, tile_idx], value=v15[i_p, tile_idx, 0], mask=None)

    for tile_idx in nl.affine_range(n_tiles):
        for batch_idx in nl.affine_range(batch_size):
            v14[batch_idx, 0, tile_idx] = nisa.tensor_copy_dynamic_src(v4[batch_idx, tile_idx, v3[batch_idx, tile_idx, 0]], dtype=dtype, mask=None)

    i_b = nl.arange(n_tiles)[None, :, None]
    i_f = nl.arange(n_tiles)[None, None, :]
    nl.store(v14_hbm[i_p, i_f], value=v14[i_p, 0, i_f], mask=None)
    v5[i_p, 0, 0] = nisa.tensor_reduce(nl.maximum, data=v14[i_p, 0, i_f], mask=None, axis=[2], dtype=dtype, negate=False)
    v6[i_p, 0, 0] = nisa.tensor_scalar(data=v5[i_p, 0, 0], op0=nl.maximum, operand0=-np.inf, reverse0=False, dtype=dtype, mask=None)
    v7[i_p, 0, i_f] = nl.broadcast_to(nisa.iota(i_f, dtype=np.int32, mask=None), shape=(batch_size, 1, n_tiles))
    v8[i_p, 0, i_f] = nisa.tensor_scalar(data=v7[i_p, 0, i_f], op0=nl.subtract, operand0=n_tiles, reverse0=True, dtype=np.int32, mask=None)
    v9[i_p, 0, i_f] = nisa.tensor_scalar(data=v14[i_p, 0, i_f], op0=nl.equal, operand0=v6[i_p, 0, 0], reverse0=False, dtype=np.uint8, mask=None)
    v10[i_p, 0, i_f] = nl.multiply(v8[i_p, 0, i_f], v9[i_p, 0, i_f], mask=None, dtype=np.int32)
    v11[i_p, 0, 0] = nisa.tensor_reduce(nl.maximum, data=v10[i_p, 0, i_f], mask=None, axis=[2], dtype=np.int32, negate=False)
    v12[i_p, 0, 0] = nisa.tensor_scalar(data=v11[i_p, 0, 0],  op0=nl.maximum, operand0=-np.inf, reverse0=False, op1=nl.subtract, operand1=n_tiles, reverse1=True, dtype=np.int32, mask=None)
    nl.store(v12_hbm[i_p], value=v12[i_p, 0, 0], mask=None)
    for batch_idx in nl.affine_range(batch_size):
        nl.store(v2[batch_idx, 0], nisa.tensor_copy_dynamic_src(v15[batch_idx, v12[batch_idx, 0, 0], 0]))

    return v2

@torch.library.custom_op("vllm::neuron_argmax", mutates_args=())
def neuron_argmax(input: torch.Tensor) -> torch.Tensor:
    return nki_argmax(input)


@neuron_argmax.register_fake
def _(
    input: torch.Tensor,
) -> torch.Tensor:
    batch_size, vocab_size = input.shape
    return torch.empty((batch_size, 1), dtype=input.dtype)

