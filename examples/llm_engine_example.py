import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams


def main(args: argparse.Namespace):
    # Parse the CLI argument and initialize the engine.
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    # Test the following prompts.
    sampling_params = SamplingParams(temperature=0.0)
    test_prompts = [
        ("A robot may not injure a human being",
         SamplingParams(temperature=0.0)),
        ("To be or not to be,",
         sampling_params), # SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         sampling_params), # SamplingParams(n=2,
                        # best_of=5,
                        # temperature=0.8,
                        # top_p=0.95,
                        # frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         sampling_params), # SamplingParams(n=3, best_of=3, use_beam_search=True,
                        # temperature=0.0)),
    ]

    # Run the engine by calling `engine.step()` manually.
    request_id = 0
    while True:
        # To test continuous batching, we add one request at each step.
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                print(request_output)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    import os
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['PATH'] = "/home/ubuntu/workspace/vllm/venv/bin:/home/ubuntu/.local/bin:/opt/aws/neuron/bin:/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin/:/opt/aws/neuron/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
    os.environ['NEURONX_DUMP_TO'] = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.model = "openlm-research/open_llama_3b"
    args.tensor_parallel_size = 1 # tp_degree=32
    args.worker_use_ray = False
    args.max_num_seqs = 2
    args.max_model_len = 256
    main(args)
