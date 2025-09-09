import argparse
from swift.llm import (
    ModelType,
    Template,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
    register_template,
)
import torch
MODEL_DICT = {
    "qwen/Qwen2-1___5B-Instruct": ModelType.qwen2_1_5b_instruct,
    "qwen/Qwen2-7B-Instruct": ModelType.qwen2_7b_instruct,
    "qwen/Qwen2-72B-Instruct": ModelType.qwen2_72b_instruct,
    "meta-llama/Llama-2-7b-chat-hf": ModelType.llama2_7b_chat,
    "meta-llama/Llama-2-13b-chat-hf": ModelType.llama2_13b_chat,
    "meta-llama/Llama-2-70b-chat-hf": ModelType.llama2_70b_chat,
    "meta-llama/Meta-Llama-3-8B-Instruct": ModelType.llama3_8b_instruct,
    "meta-llama/Meta-Llama-3-70B-Instruct": ModelType.llama3_70b_instruct,
    "meta-llama/Meta-Llama-3___1-8B-Instruct": ModelType.llama3_1_8b_instruct,
    "meta-llama/Meta-Llama-3___1-70B-Instruct": ModelType.llama3_1_70b_instruct,
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="model path",
)

parser.add_argument(
    "--model_type",
    type=str,
    required=True,
    help="qwen/Qwen2-1___5B-Instruct, qwen/Qwen2-7B-Instruct, qwen/Qwen2-72B-Instruct, meta-llama/Llama-2-7b-chat-hf, meta-llama/Llama-2-13b-chat-hf, meta-llama/Llama-2-70b-chat-hf, meta-llama/Meta-Llama-3-8B-Instruct, meta-llama/Meta-Llama-3-70B-Instruct, meta-llama/Meta-Llama-3___1-8B-Instruct, meta-llama/Meta-Llama-3___1-70B-Instruct",
)

parser.add_argument(
    "--benchmark",
    type=str,
    default="",
    help="Specific Benchmark",
)

parser.add_argument(
    "--gpu_num",
    type=int,
    default=1,
    help="Num of GPUs",
)

parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="temperature",
)

parser.add_argument(
    "--nowtime",
    required=True,
    help="nowtime",
)

args = parser.parse_args()
print(args)

model_type = MODEL_DICT[args.model_type]
llm_engine = get_vllm_engine(
    model_type,
    torch.bfloat16,
    model_id_or_path=args.model_path,
    gpu_memory_utilization=0.8,
    tensor_parallel_size=args.gpu_num,
    engine_kwargs={"distributed_executor_backend": "ray"},
)

if args.benchmark == "humaneval":

    class HumanevalTemplateType:
        humaneval = "humaneval"

    if "Llama-2" in args.model_type:
        register_template(
            HumanevalTemplateType.humaneval,
            Template(
                ["<s>[INST] "],
                ["{{QUERY}} [/INST]\n```python\n"],
                ["</s><s>[INST] "],
                ["</s>"],
                "You are an intelligent programming assistant to produce Python algorithmic solutions",
                ["<s>[INST] <<SYS>>\n{{SYSTEM}}\n<</SYS>>\n\n"],
            ),
        )
    elif "Llama-3" in args.model_type:
        register_template(
            HumanevalTemplateType.humaneval,
            Template(
                ["<|begin_of_text|>"],
                [
                    "<|start_header_id|>user<|end_header_id|>\n\n{{QUERY}}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n```python\n"
                ],
                ["<|eot_id|>"],
                ["<|eot_id|>"],
                "You are an intelligent programming assistant to produce Python algorithmic solutions",
                [
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{SYSTEM}}<|eot_id|>"
                ],
            ),
        )

    elif "Qwen" in args.model_type:
        register_template(
            HumanevalTemplateType.humaneval,
            Template(
                [],
                [
                    "<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n```python\n"
                ],
                ["<|im_end|>\n"],
                ["<|im_end|>"],
                "You are an intelligent programming assistant to produce Python algorithmic solutions",
                ["<|im_start|>system\n{{SYSTEM}}<|im_end|>\n"],
                auto_add_bos=False,
            ),
        )

    template = get_template(HumanevalTemplateType.humaneval, llm_engine.hf_tokenizer)
else:
    template = get_template(
        get_default_template_type(model_type), llm_engine.hf_tokenizer
    )

# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 512
llm_engine.generation_config.repetition_penalty = 1.0
llm_engine.generation_config.temperature = args.temperature
print(llm_engine.generation_config)


def inference(data_infer):
    return inference_vllm(
        # llm_engine, template, data_infer, use_tqdm=False, verbose=True,
        llm_engine,
        template,
        data_infer,
        use_tqdm=True,
        verbose=False,
    )
