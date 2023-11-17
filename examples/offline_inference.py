import os
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['PATH'] = "/home/ubuntu/workspace/vllm/venv/bin:/home/ubuntu/.local/bin:/opt/aws/neuron/bin:/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin/:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
os.environ['NEURONX_DUMP_TO'] = os.getcwd()

os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"
os.environ["NEURON_CC_FLAGS"]= " -O1 --tensorizer-options=' --no-run-pg-layout-and-tiling ' --internal-backend-options=' --enable-indirect-memcpy-bound-check=false ' "

# Create an LLM.
# llm = LLM(model="facebook/opt-125m")
llm = LLM(model="openlm-research/open_llama_3b", tensor_parallel_size=2, max_num_seqs=32, max_model_len=128, block_size=128)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
