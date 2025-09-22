# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List
import pdb
import json
from bert_score import score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
from tqdm import tqdm
import argparse

result = []
def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=2048, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        dict = {}
        res = resp_list[index].choices[0].message.content
        dict['text'] = res
        result.append(dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--model_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/Chinese-medical-dialogue/train_medical_sample.jsonl")

    args = parser.parse_args()
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats
    from swift.llm import VllmEngine

    model_path = args.model_path
    model_type = 'qwen2_5'

    model = model_path

    infer_backend = 'vllm'

    if infer_backend == 'pt':
        engine = PtEngine(model, model_type=model_type, max_batch_size=64)
    elif infer_backend == 'vllm':
        engine = VllmEngine(model, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=1)
    

    dataset = load_dataset(['/mnt/data3/nianke_multi_agent/agentscope/train_data/huatuo-o1-sft/test_medical.jsonl'], strict=False, shuffle=False)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    path = 'generate_test_data/train_medical_sample_clean.jsonl'
    with open(path, 'w', encoding='utf-8') as f:
        for item in result:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
  
    