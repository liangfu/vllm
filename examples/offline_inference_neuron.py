import os
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
os.environ["NEURON_CC_FLAGS"]= " -O1 "
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "The capital of France is",
    "Hello, my name is Liam and I am a 16 year old",
    "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better.",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "The future of AI is",
    "Hello, my name is Liam and I am a 16 year old boy who is currently in the 11th grade. I am a",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="openlm-research/open_llama_3b",
    tensor_parallel_size=2,
    max_num_seqs=4,
    max_model_len=1024,
    max_num_batched_tokens=1024,
    block_size=128,
    gpu_memory_utilization=0.2,

    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron",
    tensor_parallel_size=2)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
