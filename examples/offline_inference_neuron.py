import os, torch
os.environ['NEURONX_DUMP_TO'] = os.path.join(os.getcwd(),"_compile_cache")
# os.environ["NEURON_CC_FLAGS"]= " -O1 --internal-enable-dge-levels=vector_dynamic_offsets "
# os.environ["NEURON_CC_FLAGS"]= " -O3 --internal-enable-dge-levels=vector_dynamic_offsets --disable-internal-io-dge"
os.environ["NEURON_CC_FLAGS"]= " -O1 "
#  --internal-compiler-debug-mode=penguin --tensorizer-options='--enable-dge-on-indirect-dma' "
# os.environ["NEURON_CC_FLAGS"] += " --tensorizer-options='--dump-after=All' "
# os.environ["NEURON_CC_FLAGS"]= " --tensorizer-options='--enable-dge-on-indirect-dma' "
os.environ["NEURON_RT_DBG_EMBEDDING_UPDATE_BOUND_CHECK"] = "0"
os.environ["NEURON_RT_DBG_INDIRECT_MEMCPY_BOUND_CHECK"] = "0"
torch.set_printoptions(sci_mode=False)
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "The capital of France is",
    # "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows in the end the triumph of high achievement, and who at the worst, if he fails, at least fails while daring greatly, so that his place shall never be with those cold and timid souls who neither know victory nor",
    # "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows in the end",
    "It is not the critic who counts; not the man who points out how the strong man stumbles, or where the doer of deeds could have done them better. The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; who strives valiantly; who errs, who comes short again and again, because there is no effort without error and shortcoming; but who does actually strive to do the deeds; who knows great enthusiasms, the great devotions; who spends himself in a worthy cause; who at the best knows",
    "Hello, my name is Liam and I am a 16 year old",
    "The future of AI is",
    "Hello, my name is L",
    "Hello, my name is M",
    "Hello, my name is O",
    # "The future of AI is",
    # "Hello, my name is Liam and I am a 16 year old",
    # "The future of AI is",
]

# Create a sampling params object.
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_tokens=100, max_tokens=200)
sampling_params = SamplingParams(min_tokens=100, max_tokens=200)

# Create an LLM.
llm = LLM(
    # model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model="openlm-research/open_llama_3b",
    # model="openlm-research/open_llama_7b",
    # model="TheBloke/Llama-2-70B-fp16",

    model="openlm-research/open_llama_3b",
    tensor_parallel_size=2,
    max_num_seqs=4,
    max_model_len=256,
    max_num_batched_tokens=256,
    block_size=128,
    gpu_memory_utilization=0.2,

    # model="openlm-research/open_llama_7b",
    # tensor_parallel_size=8,
    # max_num_seqs=4,
    # max_model_len=2048,
    # max_num_batched_tokens=2048,
    # block_size=256,
    # gpu_memory_utilization=0.2,

    # model="TheBloke/Llama-2-70B-fp16",
    # tensor_parallel_size=32,
    # max_num_seqs=8,
    # max_model_len=1024,
    # max_num_batched_tokens=1024,
    # block_size=256,
    # gpu_memory_utilization=0.3,

    # The device can be automatically detected when AWS Neuron SDK is installed.
    # The device argument can be either unspecified for automated detection,
    # or explicitly assigned.
    device="neuron")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
