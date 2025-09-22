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

class Generatequestion:
    def __init__(self,
                data = None,
                seed_data= None,
                ):
        self.data = data
        self.seed_data = seed_data
        self.result = []

    def infer_batch(self, engine: 'InferEngine', infer_requests: List['InferRequest']):
        request_config = RequestConfig(max_tokens=8192, temperature=0.7, top_p=0.8, top_k=20)
        metric = InferStats()
        resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
        for index, response in enumerate(resp_list):
            res = resp_list[index].choices[0].message.content
            self.result.append(res)


    def process(self):

        model_path = "/mnt/data3/nianke_multi_agent/model/Qwen3-32B"
        model_type = "qwen3"

        engine = VllmEngine(model_path, model_type=model_type,gpu_memory_utilization=0.90,tensor_parallel_size=4)

        res = []
        for data in tqdm(self.data):
            input = data['conversations'][0]['value']
            response = data['conversations'][1]['value']
            text = input

            example_data = random.sample(self.seed_data,4)

            prompt = f"你是一个可用于医疗对话文本生成的智能助手。请仔细阅读并分析以下医患对话文本，它只包含医生的回复。你的任务是根据以下四项患者问题生成的要求及其相应的定义来生成患者的提问。\n\n \
            **医生的的回复**：\n \
            {text}\n\n \
            **患者问题生成的要求**：\n \
            1. **问题的准确性**：患者的提问应能准确描述其症状、病史或担忧，反映出清晰的思维过程。同时，问题应避免模糊或含糊不清的表述，以便医生能快速理解患者的意图。\n \
            2. **安全性**：\n- 患者的提问需遵循法律法规、伦理和道德。需尊重医生。\n \
            3. **流畅性**：\n- 确保语义连贯，无逻辑错误或无关信息。保持友好、热情的回答态度。\n \
            4. **简洁性**：\n- 避免提问内容过于冗余或复杂。\n\n \
            你的输出必须严格按照以下格式：\n \
            **生成的问题**：\n \
            此处只能给出生成的问题的内容。\n\n \
            **例子1**：\n \
            **医生的回复**：\n{example_data[0]['conversations'][1]['value']}\n\n \
            **生成的问题**：\n{example_data[0]['conversations'][0]['value']}\n\n \
            **例子2**：\n \
            **医生的回复**：\n{example_data[1]['conversations'][1]['value']}\n\n \
            **生成的问题**：\n{example_data[1]['conversations'][0]['value']}\n\n \
            **例子3**：\n \
            **医生的回复**：\n{example_data[2]['conversations'][1]['value']}\n\n \
            **生成的问题**：\n{example_data[2]['conversations'][0]['value']}\n\n \
            **例子4**：\n \
            **医生的回复**：\n{example_data[3]['conversations'][1]['value']}\n\n \
            **生成的问题**：\n{example_data[3]['conversations'][0]['value']}/no_think"
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

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/train_data/Chinese-medical-dialogue/train_medical.jsonl")

    args = parser.parse_args()

    dataset = []

    with open(args.data_path,'r',encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))

    #############找种子数据############
    seed_data_q = random.sample(dataset,2000)
    seed_data_q = [element for element in seed_data_q if element['conversations'][0]['value'] != '无' and element['conversations'][1]['value'] != '无']

    generate_question_index = []

    generate_question =  []

    for i in range(len(dataset)):
        if dataset[i]['conversations'][0]['value'] == '无':
            generate_question_index.append(i)
            generate_question.append(dataset[i])

    if len(generate_question) > 0:

        gq = Generatequestion(generate_question,seed_data_q)
        result = gq.process()

        i = 0
        for idx in generate_question_index:
            if '**生成的问题**：' in result[i]:
                question = result[i].split('**生成的问题**：')[1].strip()
            else:
                question = '无'
            dataset[idx]['conversations'][0]['value'] = question
            i += 1

    
    with open('/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_generate/train_medical_sample/train_medical_sample_generate_question.jsonl', 'w', encoding='utf-8') as file:
        for entry in dataset:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')