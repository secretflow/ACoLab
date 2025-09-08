import os
import sys
import json
import pdb
import random
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_path = your_model_path
#/mnt/data1/model/meta-llama/Meta-Llama-3-8B-Instruct
#/mnt/data1/model/qwen/Qwen2-7B-Instruct
# print(data_type, model_path)

from swift.llm import (
    ModelType,
    get_vllm_engine,
    get_default_template_type,
    get_template,
    inference_vllm,
)

# model_type = ModelType.llama3_8b_instruct
model_type = ModelType.qwen2_7b_instruct

llm_engine = get_vllm_engine(
    model_type,
    torch.float16,
    model_id_or_path=model_path,
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,
)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_tokens = 512
# llm_engine.generation_config.repetition_penalty = 1.1
llm_engine.generation_config.temperature = 0.0
llm_engine.generation_config.top_k = 20
# llm_engine.generation_config.top_p = 1.0
# llm_engine.generation_config.seed = 4
print(llm_engine.generation_config)


with open("/LLaMA-Factory-main/data_winogrande/winogrande_1.1/winogrande_validation_qwen.json", "r", encoding="utf-8") as f:
    data = json.load(f)


data_infer = []
label_list = []

for idx,element in enumerate(data):
    conversation = element['conversations']
    dict1 = {}
    dict1['query'] = conversation[0]['value']
    data_infer.append(dict1)

    label_list.append(conversation[1]['value'])


resp_list = inference_vllm(llm_engine, template, data_infer, use_tqdm=True)

result = []
for index, response in enumerate(resp_list):
    # res = response["response"].strip().lower()
    res = response["response"].strip()
    result.append(res)
pdb.set_trace()
if len(result) != len(label_list):
        print("error")
else:
    Accuracy = accuracy_score(label_list, result)
    print("The accuracy of data is {:.3f}".format(Accuracy))
