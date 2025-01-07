from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_xla.experimental.custom_kernel  # Required to register custom ops.

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState

B_P_SIZE = 128
import torch.nn.functional as F

class NeuronAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "NEURON_ATTN_V1"

    @staticmethod
    def get_impl_cls() -> Type["NeuronAttentionBackendImpl"]:
        return NeuronAttentionBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["NeuronAttentionMetadata"]:
        return NeuronAttentionMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_kv_heads, num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        raise RuntimeError("swap_blocks is not used for the TPU backend.")

    @torch.compile(backend="openxla")
    @staticmethod
    def copy_blocks(
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        src_to_dists: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        src_indices, dst_indices = src_to_dists
        for k_cache, v_cache in kv_caches:
            torch.ops.xla.dynamo_set_buffer_donor_(k_cache, True)
            k_cache[:, dst_indices] = k_cache[:, src_indices]
            torch.ops.xla.dynamo_set_buffer_donor_(v_cache, True)
            v_cache[:, dst_indices] = v_cache[:, src_indices]
class BlockDiagonalCausalFromBottomRightMask:

    @staticmethod
    def _from_seqlens(query_lens, seq_lens, block_size=None):
        from torch import logical_and, logical_or

        contexted = block_size is None
        context_lens = torch.tensor(seq_lens) - torch.tensor(query_lens)
        n_queries = sum(query_lens)
        num_seqs = len(query_lens)
        if contexted:
            key_lens_blockaligned = seq_lens
        else:
            n_blocks_per_seq = (context_lens + block_size - 1) // block_size
            offset_per_seq = n_blocks_per_seq * block_size
            key_lens_blockaligned = offset_per_seq[:num_seqs].tolist()
        n_keys = sum(key_lens_blockaligned)

        a = (torch.arange(n_queries).reshape(n_queries,
                                             1).expand(n_queries, n_keys))
        b = torch.arange(n_keys).reshape(1, n_keys).expand(n_queries, n_keys)
        q_cumsum = torch.tensor([0] + query_lens).cumsum(dim=0)
        k_cumsum = torch.tensor([0] + key_lens_blockaligned).cumsum(dim=0)

        prior_mask = torch.zeros(n_queries, n_keys)
        new_masks: list[torch.Tensor] = []
        for seq_id in range(num_seqs):
            ri = q_cumsum[seq_id]
            ci = k_cumsum[seq_id]
            nr = query_lens[seq_id]

            if contexted:
                nc = seq_lens[seq_id]
                a_offset = ci + nc - ri - nr
                new_mask = (a + a_offset) >= b
            else:
                nc = context_lens[seq_id]
                a_offset = ci + nc - 1
                new_mask = a_offset >= b

            left_mask = b >= ci
            top_mask = a >= ri
            bottom_mask = a < (ri + nr)

            new_mask = logical_and(
                logical_and(logical_and(new_mask, left_mask), top_mask),
                bottom_mask,
            )
            prior_mask = logical_or(prior_mask, new_mask)
            new_masks = new_masks + [new_mask]
        return prior_mask

    @staticmethod
    def from_seqlens(query_lens, seq_lens, block_size=None):
        contexted = block_size is None
        if contexted:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens)
            active_mask = None
        else:
            prior_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, seq_lens, block_size)
            active_mask = BlockDiagonalCausalFromBottomRightMask._from_seqlens(
                query_lens, query_lens)
        return prior_mask, active_mask


@dataclass
class NeuronAttentionMetadata:

    # is_prompt: bool
    slot_mapping: torch.Tensor
    block_tables: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    active_block_table: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    # Currently, input sequences can only contain all prefills
    # or all decoding.
    # block_tables: Optional[torch.Tensor] = None
    # context_lens: Optional[torch.Tensor] = None
    # effective_query_lens: Optional[torch.Tensor] = None

    # @property
    # def prefill_metadata(self) -> Optional["NeuronAttentionMetadata"]:
    #     if self.num_prefills == 0:
    #         return None

    #     assert self.num_decode_tokens == 0
    #     return self

    # @property
    # def decode_metadata(self) -> Optional["NeuronAttentionMetadata"]:
    #     if self.num_decode_tokens == 0:
    #         return None

    #     assert self.num_prefills == 0
    #     assert self.num_prefill_tokens == 0
    #     assert self.block_tables is not None
    #     assert self.context_lens is not None
    #     return self


class NeuronAttentionBackendImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        # HACK AOYU Disable initial check in  NeuronAttentionBackendImpl(AttentionImpl)
        # if head_size % 128 != 0:
        #     raise NotImplementedError("Head size must be a multiple of 128.")
        # if alibi_slopes is not None:
        #     raise NotImplementedError("Alibi slopes is not supported.")
        # if sliding_window is not None:
        #     raise NotImplementedError("Sliding window is not supported.")
        # if kv_cache_dtype != "auto":
        #     raise NotImplementedError("FP8 KV cache dtype is not supported.")
        # if blocksparse_params is not None:
        #     raise NotImplementedError("Blocksparse is not supported.")
        # if logits_soft_cap is not None:
        #     raise NotImplementedError(
        #         "Attention logits soft-capping is not supported.")

        # if torch_xla.tpu.version() < 4:
        #     raise NotImplementedError("TPU version must be 4 or higher.")

        # self.megacore_mode = None
        # tpu_env = torch_xla.tpu.get_tpu_env()
        # tpu_type = (tpu_env.get("ACCELERATOR_TYPE", None)
        #             or tpu_env.get("TYPE", None)
        #             or tpu_env.get("TPU_ACCELERATOR_TYPE", None))
        # assert tpu_type is not None
        # tpu_type = tpu_type.lower()

        # if (("lite" not in tpu_type) and ("v6" not in tpu_type)):
        #     if self.num_kv_heads % 2 == 0:
        #         self.megacore_mode = "kv_head"
        #     else:
        #         # NOTE(woosuk): If the batch size is not a multiple of 2, the
        #         # megacore mode will be None.
        #         self.megacore_mode = "batch"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: NeuronAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: str = AttentionType.DECODER,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with Neuron attention.

        Args:
            query: shape = [batch_size, seq_len, num_heads * head_size]
            key: shape = [batch_size, seq_len, num_kv_heads * head_size]
            value: shape = [batch_size, seq_len, num_kv_heads * head_size]
            kv_cache[0] = [num_kv_heads, num_blocks, block_size, head_size]
            kv_cache[1] = [num_kv_heads, num_blocks, block_size, head_size]
                NOTE: kv_cache[0] and kv_cache[1] will be an empty tensor 
                with shape [0] for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "NeuronAttentionBackendImpl")
        batch_size, seq_len, hidden_size = query.shape
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(batch_size, seq_len, self.num_kv_heads,
                           self.head_size)
        

        if kv_cache[0].numel() > 0:
            slot_mapping = attn_metadata.slot_mapping
            key_cache, value_cache = kv_cache
            write_to_kv_cache(key, value, key_cache, value_cache, slot_mapping)

        query = query * self.scale
        pad_dims = (
            0,
            B_P_SIZE - query.shape[3],
            0,
            0,
            0,
            B_P_SIZE - query.shape[1],
            0,
            0
        )
        # output = torch.empty_like(query)
        output = []
        # padding
        query = F.pad(query, pad_dims, "constant", 0)
        key = F.pad(key, pad_dims, "constant", 0)
        value = F.pad(value, pad_dims, "constant", 0)
        key_cache = F.pad(key_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        value_cache = F.pad(value_cache, (0, B_P_SIZE - self.head_size), "constant", 0)
        key_cache=key_cache.permute(1,2,0,3).contiguous()
        value_cache=value_cache.permute(1,2,0,3).contiguous()
        from vllm.attention.ops.nki_flash_attn import context_attention_fwd, context_flash_attention_fwd
        for batch_idx in range(batch_size):
            q = query[batch_idx,:,:,:].unsqueeze(0)
            k = key[batch_idx,:,:,:].unsqueeze(0)
            v = value[batch_idx,:,:,:].unsqueeze(0)
            # debug tricks
            attn_mask = attn_metadata.attn_mask.permute(0,1)
            q = q.permute(0,2,3,1).contiguous()
            k = k.permute(0,2,3,1).contiguous()
            v = v.permute(0,2,1,3).contiguous()
            out = context_attention_fwd(
                q,
                k,
                v,
                key_cache,
                value_cache,
                block_table=attn_metadata.active_block_table,
                attn_mask=attn_mask,
                n_kv_head=self.num_kv_heads,
                head_size=self.head_size,
            )
            out = out[:,:,:seq_len,:self.head_size]
            out = out.permute(0, 2, 1, 3).contiguous()
            output.append(out)
            # output[batch_idx,:,:,:].unsqueeze(0) = out
        # Reshape the output tensor.
        output = torch.cat(output, dim=0)
        return output.reshape(batch_size, seq_len, hidden_size)


def write_to_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops.xla.dynamo_set_buffer_donor_(key_cache, True)
    torch.ops.xla.dynamo_set_buffer_donor_(value_cache, True)

    key = key.flatten(0, 2)
    value = value.flatten(0, 2)
    key_cache = key_cache.flatten(0, 2)
    value_cache = value_cache.flatten(0, 2)
    key_cache.index_copy_(0, slot_mapping, key)
    value_cache.index_copy_(0, slot_mapping, value)


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    pages_per_compute_block: int,
    megacore_mode: Optional[str],
) -> torch.Tensor:
    batch_size = query.shape[0]
    if megacore_mode == "batch" and batch_size % 2 != 0:
        megacore_mode = None
    else:
        megacore_mode = megacore_mode

    # NOTE(woosuk): A temporary workaround to avoid the error:
    # "xla::paged_attention() Expected a value of type 'str' for
    # argument 'megacore_mode' but instead found type 'NoneType'."
    if megacore_mode is not None:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
            megacore_mode=megacore_mode,
        )
    else:
        output = torch.ops.xla.paged_attention(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            pages_per_compute_block,
        )
    return output
