# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List
import pdb
import json
import re
from bert_score import score
from rouge_score import rouge_scorer
from fraction import Fraction
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
from tqdm import tqdm
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
from swift.plugin import InferStats
from swift.llm import VllmEngine
import argparse
import pickle


result = []

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=4096, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        res = resp_list[index].choices[0].message.content
        result.append(res)


    # def process(self):

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/train_data/Chinese-medical-dialogue/train_medical.jsonl")

    args = parser.parse_args()

    model_path = "binary_screening_model"
    model_type = "qwen2_5"

    engine = VllmEngine(model_path, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=4)
    
    dataset = load_dataset([args.data_path], strict=False, shuffle=False)[0]

    print(f'dataset: {dataset}')
    res = []
    for data in tqdm(dataset):
        input = data['messages'][0]['content']
        response = data['messages'][1]['content']
        text = "患者的问题：\n" + input + '\n医生的回答：\n' + response
        # text = input + response
        prompt = f"你是一个智能医疗文本优化的评价助手。请仔细阅读并分析以下医患对话文本，它由患者的问题和医生的回答组成。文本可能会存在如问题或回答不正确，语言不规范或不流畅，内容不合规等问题。如果存在问题则需要对文本进行优化，否则不需要优化。你的任务是判断是否需要对文本进行优化。 \n\n \
        【文本】：{text}"

        data['messages'][0]['content'] = prompt
        res.append(InferRequest(**data))

    infer_requests = res
    infer_batch(engine, infer_requests)

    noise = [1] * len(result)

    # 文本需要优化:0 文本不需要优化:1
    for idx in range(len(result)):
        if result[idx] == '文本需要优化':
            noise[idx] = 0       
        
    with open('/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/data/clean_or_dirty.pkl', 'wb') as file:
        pickle.dump(noise, file)