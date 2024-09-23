"""Utilities for selecting and loading neuron models."""
import importlib
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, CacheConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.cache_engine import CacheEngine

TORCH_DTYPE_TO_NEURON_AMP = {
    "auto": "f32",
    "half": "f16",
    "float16": "f16",
    "bfloat16": "bf16",
    "float": "f32",
    "float32": "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float32: "f32",
}

# Models supported by Neuron.
_NEURON_SUPPORTED_MODELS: Dict[str, Tuple[str, str, str]] = {
    "LlamaForCausalLM": ("transformers_neuronx.llama.model",
                         "LlamaForSampling", "LlamaForCausalLM"),
}


class NeuronCasualLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()

        # Lazy initialized
        self.model: nn.Module

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_metadata,
    ) -> torch.Tensor:
        # print(f"input_metadata={input_metadata}")
        # print(f"input_ids={input_ids.flatten()}, cache_ids={positions.flatten()}, slot_mapping={input_metadata.slot_mapping.flatten()}, prompt_lens={input_metadata.seq_lens_tensor}, block_tables={input_metadata.block_tables.flatten()}")
        import time
        tic = time.time()
        logits = self.model(input_ids.reshape(1, -1),
                            cache_ids=positions.reshape(1, -1),
                            start_ids=input_metadata.slot_mapping,
                            input_metadata=input_metadata)
        print(f"elapsed: {(time.time()-tic)*1000.:.2f} ms")
        return logits

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # print(logits.shape)
        # print(torch.min(logits, dim=1).values.shape)
        # print("logits", logits - torch.min(logits, dim=1, keepdim=True).values)
        logits_subtract = logits - torch.min(logits, dim=1, keepdim=True).values
        # print("logits max", torch.max(logits_subtract, dim=1))
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, **kwargs):
        arch = _get_model_architecture(self.config)
        neuronx_module_path, neuronx_model_cls_name, hf_model_cls_name = (
            _NEURON_SUPPORTED_MODELS[arch])
        neuronx_module = importlib.import_module(neuronx_module_path)
        neuronx_model_cls = getattr(neuronx_module, neuronx_model_cls_name)

        self.model = neuronx_model_cls.from_pretrained(model_name_or_path,
                                                       **kwargs)


def _get_model_architecture(config: PretrainedConfig) -> str:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _NEURON_SUPPORTED_MODELS:
            return arch
    raise ValueError(
        f"Model architectures {architectures} are not supported on Neuron "
        f"for now. Supported architectures: "
        f"{list(_NEURON_SUPPORTED_MODELS.keys())}")


def _get_buckets(env: str, default_value: List[int]) -> List[int]:
    env_value = os.getenv(env)
    if env_value is None:
        return default_value
    buckets_remove_empty = filter(
        lambda x: x is not None and len(x.strip()) > 0, env_value.split(","))
    buckets_int = map(int, buckets_remove_empty)
    buckets_list = list(buckets_int)
    return buckets_list

def determine_num_available_blocks(cache_config, model_config, parallel_config) -> Tuple[int, int]:
    """Determine the number of available KV blocks.

    Swapping is not yet supported, so always return num_cpu_blocks=0.
    """
    total_gpu_memory = 16 * 1e9 # 16GiB per NeuronCore
    cache_block_size = CacheEngine.get_cache_block_size(cache_config, model_config, parallel_config)
    num_gpu_blocks = int((total_gpu_memory * cache_config.gpu_memory_utilization) // cache_block_size)
    num_gpu_blocks = max(num_gpu_blocks, 0)
    assert num_gpu_blocks > 0, f"insufficient K/V cache space."

    # Swap not yet supported with Neuron backend.
    num_cpu_blocks = 0

    return num_gpu_blocks, num_cpu_blocks


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig, 
                     cache_config: CacheConfig) -> nn.Module:
    from transformers_neuronx import constants
    from transformers_neuronx.config import (ContinuousBatchingConfig, QuantizationConfig,
                                             NeuronConfig)

    # Create a model instance.
    amp = TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype]
    model = NeuronCasualLM(model_config.hf_config)

    continuous_batching_config = ContinuousBatchingConfig(
        max_model_len=model_config.max_model_len,
        max_num_seqs=scheduler_config.max_num_seqs,
        enable_chunked_prefill=True,
        optimized_paged_attention=True)
    neuron_config = NeuronConfig(
        fuse_qkv=True,
        quant = QuantizationConfig(quant_dtype='s8', dequant_dtype=amp),
        weight_tiling=True,
        cache_layout=constants.Layout.BSH,
        attention_layout=constants.Layout.BSH,
        continuous_batching=continuous_batching_config)

    # Need to init cache engine before load_weights for correct 
    # operation.

    # TODO: currently this happens here and also later. Ideally
    # we need to update tnx code to make sure we can call init 
    # after the model load.
    num_gpu_blocks, num_cpu_blocks = (determine_num_available_blocks(cache_config, model_config, parallel_config))

    if cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = cache_config.num_gpu_blocks_override
        num_gpu_blocks = num_gpu_blocks_override

    cache_config.num_gpu_blocks = num_gpu_blocks
    cache_config.num_cpu_blocks = num_cpu_blocks

    assert cache_config.num_gpu_blocks is not None
    neuron_config.continuous_batching.init_cache_engine(
        block_size=cache_config.block_size,
        num_blocks=cache_config.num_gpu_blocks
    )

    context_length_estimates = _get_buckets("NEURON_CONTEXT_LENGTH_BUCKETS",
                                            [scheduler_config.max_num_batched_tokens])
    n_positions = _get_buckets("NEURON_TOKEN_GEN_BUCKETS",
                               [scheduler_config.max_model_len])
    # [scheduler_config.max_model_len//2, scheduler_config.max_model_len])

    # Uncomment below to test bucketing for chunked prefill
    # n_positions = [n_positions[0]//4, n_positions[0]//2, n_positions[0]]
    # context_length_estimates = [context_length_estimates[0]//4, context_length_estimates[0]//2, context_length_estimates[0]]

    # Load the weights from the cached or downloaded files.
    model.load_weights(model_config.model,
                       tp_degree=parallel_config.tensor_parallel_size,
                       amp=amp,
                       neuron_config=neuron_config,
                       context_length_estimate=context_length_estimates,
                       n_positions=n_positions,
                       batch_size=scheduler_config.max_num_seqs)

    return model.eval()
