import os, torch
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
os.environ["NEURON_CC_FLAGS"]= " -O1 "

from vllm import LLM, SamplingParams

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
llm = LLM(model="openlm-research/open_llama_3b",
          tensor_parallel_size=2,
          max_num_seqs=4,

          max_model_len=256,
          max_num_batched_tokens=256,

          block_size=128,
          gpu_memory_utilization=0.2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
