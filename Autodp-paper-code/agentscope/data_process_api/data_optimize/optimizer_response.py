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
import random
import subprocess

class Optimizeresponse:
    def __init__(self,
                data = None,
                ):
        self.data = data
        self.result = []

    def infer_batch(self, engine: 'InferEngine', infer_requests: List['InferRequest']):
        request_config = RequestConfig(max_tokens=8192, temperature=0.7, top_p=0.8, top_k=20)
        metric = InferStats()
        resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
        for index, response in enumerate(resp_list):
            res = resp_list[index].choices[0].message.content
            self.result.append(res)


    def process(self, have_response_idx):

        model_path = "/mnt/data3/nianke_multi_agent/model/Qwen3-32B"
        model_type = "qwen3"

        engine = VllmEngine(model_path, model_type=model_type,gpu_memory_utilization=0.90,tensor_parallel_size=4)

        res = []
        for idx, data in tqdm(enumerate(self.data)):
            if idx in have_response_idx:
                input = data['conversations'][0]['value']
                response = data['conversations'][1]['value']

                prompt = f"你是一款用于优化医疗对话文本的智能助手。请仔细阅读并分析以下医患对话内容。您的任务是根据五类医生的回复优化要求及其定义，对文本中的医生的回复部分进行优化。\n\n \
                **患者的问题**：\n \
                {input}\n\n \
                **医生的回复**：\n \
                {response}\n\n \
                **医生回复的优化要求**：\n \
                1. **回复的准确性**：医生应能准确理解患者的问题，并提供科学、准确的回答。\n \
                2. **安全性**：\n 医生的回答需遵循法律法规、伦理和职业道德。需尊重患者的选择。\n \
                3. **流畅性**：\n 确保语义连贯，无逻辑错误或无关信息。保持友好、热情的回答态度。\n \
                4. **简洁性**：\n 清晰简洁地解释复杂医学知识。避免回答内容过于冗余。\n \
                5. **优化后医生的回复的长度**：\n 将优化后的回复长度控制在原回复长度的上下10%范围内。\n\n \
                你需要特别注意，如果患者的问题的内容只有'无'则你只需关注优化的后四点要求。\n \
                你的输出必须严格按照以下格式：\n \
                **优化后的医生的回复**：\n \
                此处只能输出优化后的医生的回复。\n \
                **理由**：\n \
                这里只能输出理由。/no_think"
                
                data_new = {}
                data_new['messages'] = []
                dict = {}
                dict['role'] = 'user'
                dict['content'] = prompt
                data_new['messages'].append(dict)
                res.append(InferRequest(**data_new))

        infer_requests = res
        self.infer_batch(engine, infer_requests)

        return self.result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer_question.jsonl")

    args = parser.parse_args()

    # subprocess.run(['python', 'clean_or_dirty.py', '--data_path', args.data_path])

    with open('/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/data/clean_or_dirty.pkl', 'rb') as file:
        clean_or_dirty = pickle.load(file)

    dataset = []

    with open(args.data_path,'r',encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    optimizer_data = []
    optimizer_data_index = []
    for i in range(len(clean_or_dirty)):
        if clean_or_dirty[i] == 0:
            optimizer_data.append(dataset[i])
            optimizer_data_index.append(i)

    if len(optimizer_data) > 0:

        no_response_idx,have_response_idx = [],[]
        no_response = []

        #############只对有回复的文本进行优化###########
        for i in range(len(optimizer_data)):
            if optimizer_data[i]['conversations'][1]['value'] == '无':
                no_response_idx.append(i)
                no_response.append(optimizer_data[i])
            else:
                have_response_idx.append(i)

        gq = Optimizeresponse(optimizer_data)
        have_response = gq.process(have_response_idx)

        result = ['']*len(optimizer_data)

        for index, value in zip(no_response_idx, no_response):
            result[index] = value

        for index, value in zip(have_response_idx, have_response):
            result[index] = value 

        i = 0
        for idx in optimizer_data_index:
            if '**优化后的医生的回复**：' in result[i] and '**理由**：' in result[i]:
                response = result[i].split('**优化后的医生的回复**：')[1].split('**理由**：')[0].strip()
            else:
                print("................模型输出格式有问题...............")
                response = optimizer_data[i]['conversations'][1]['value']
            dataset[idx]['conversations'][1]['value'] = response
            i += 1

    
    with open('/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer.jsonl', 'w', encoding='utf-8') as file:
        for entry in dataset:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')