"""Utilities for selecting and loading neuron models."""
import importlib
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import transformers
from transformers import PretrainedConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import SamplerOutput

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
        logits = self.model(input_ids,
                            cache_ids=positions,
                            start_ids=input_metadata.slot_mapping,
                            input_metadata=input_metadata)
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


def get_neuron_model(model_config: ModelConfig,
                     parallel_config: ParallelConfig,
                     scheduler_config: SchedulerConfig) -> nn.Module:
    from transformers_neuronx import constants
    from transformers_neuronx.config import (ContinuousBatchingConfig, QuantizationConfig,
                                             NeuronConfig)

    # Create a model instance.
    amp = TORCH_DTYPE_TO_NEURON_AMP[model_config.dtype]
    model = NeuronCasualLM(model_config.hf_config)

    continuous_batching_config = ContinuousBatchingConfig(
        max_model_len=model_config.max_model_len,
        max_num_seqs=scheduler_config.max_num_seqs,
        optimized_paged_attention=True)
    neuron_config = NeuronConfig(
        fuse_qkv=True,
        quant = QuantizationConfig(quant_dtype='s8', dequant_dtype=amp),
        weight_tiling=True,
        cache_layout=constants.Layout.BSH,
        attention_layout=constants.Layout.BSH,
        continuous_batching=continuous_batching_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config.model,
        tp_degree=parallel_config.tensor_parallel_size,
        amp=amp,
        neuron_config=neuron_config,
        context_length_estimate=[scheduler_config.max_model_len],
        n_positions=[scheduler_config.max_model_len],
        batch_size=scheduler_config.max_num_seqs)

    return model.eval()
