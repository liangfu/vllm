"""A Neuron worker class."""
from typing import List, Tuple

import torch
import torch.distributed

from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.model_executor import set_random_seed
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.worker_base import LoraNotSupportedWorkerBase


class NeuronWorker(LoraNotSupportedWorkerBase):
    """A worker class that executes the model on a group of neuron cores.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.model_runner = NeuronModelRunner(model_config, parallel_config,
                                              scheduler_config, device_config)

    def init_device(self) -> None:
        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.
        """
        total_gpu_memory = self.parallel_config.tensor_parallel_size * 16 * 1e9 # 16GiB per NeuronCore
        cache_block_size = CacheEngine.get_cache_block_size(self.cache_config, self.model_config, self.parallel_config)
        num_gpu_blocks = int((total_gpu_memory * self.cache_config.gpu_memory_utilization) // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        assert num_gpu_blocks > 0, f"insufficient K/V cache space."

        # Swap not yet supported with Neuron backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        """Initialize the KV cache.
        """
        # Different values are not tested.
        assert num_cpu_blocks == 0

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        self._init_cache_engine()

    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None

        neuron_config = self.model_runner.model.model.neuron_config
        neuron_config.continuous_batching.init_cache_engine(
            block_size=self.cache_config.block_size,
            num_blocks=self.cache_config.num_gpu_blocks
        )

        # self.cache_engine = CacheEngine(self.cache_config, self.model_config,
        #                                 self.parallel_config)
        self.model_runner.set_block_size(self.cache_config.block_size)
        self.model_runner.compile_model()

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> List[SamplerOutput]:
        num_seq_groups = len(seq_group_metadata_list)

        # If there is no input, we don't need to execute the model.
        if num_seq_groups == 0:
            return []

        output = self.model_runner.execute_model(seq_group_metadata_list)

        # Neuron worker only supports single-step output. Wrap the output in a
        # list to conform to interface.
        return [output]

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError
