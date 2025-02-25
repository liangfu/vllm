# SPDX-License-Identifier: Apache-2.0
import os
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "1"

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams()

# Create an LLM.
llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
          max_num_seqs=4,
          max_model_len=512,
          max_num_batched_tokens=32,
          block_size=32,
          device="neuron",
          tensor_parallel_size=1,
          disable_async_output_proc=True,
          enable_chunked_prefill=True,
          worker_cls="vllm.v1.worker.neuron_worker.NeuronWorker"
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
