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
    request_config = RequestConfig(max_tokens=2048, temperature=0.0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        dict = {}
        res = resp_list[index].choices[0].message.content
        dict['text'] = res
        result.append(dict)

def main(args) -> None:

    path = '/mnt/data3/nianke_multi_agent/agentscope/data_analysis_api/data_rewrite/chinese/anlysis_result/anlysis_text_optimizer.pkl'
    with open(path, 'rb') as file:  # 'rb' 表示以二进制读取模式打开
        anlysis_text_optimizer = pickle.load(file)

    # path = '/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_clean/pkl/document_minhash_deduplicator_index.pkl'
    # with open(path, 'rb') as file:  # 'rb' 表示以二进制读取模式打开
    #     document_minhash_deduplicator_index = pickle.load(file)

    # path = '/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_clean/pkl/word_repetition_filter_index.pkl'
    # with open(path, 'rb') as file:  # 'rb' 表示以二进制读取模式打开
    #     word_repetition_filter_index = pickle.load(file)

    # path = '/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_select/data/medical_clean_select_index.pkl'
    # with open(path, 'rb') as file:  # 'rb' 表示以二进制读取模式打开
    #     clean_select_index = pickle.load(file)

    # # 示例数据
    # anlysis_text_optimizer = [value for index, value in enumerate(anlysis_text_optimizer) if index not in document_minhash_deduplicator_index]

    # anlysis_text_optimizer = [value for index, value in enumerate(anlysis_text_optimizer) if index not in word_repetition_filter_index]

    # anlysis_text_optimizer = [value for index, value in enumerate(anlysis_text_optimizer) if index in clean_select_index]

    model_path = "/mnt/data3/nianke_multi_agent/model/QwQ-32B"
    model_type = 'qwq'

    engine = VllmEngine(model_path, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=4)
    
    dataset = load_dataset([args.data_path], strict=False, seed=42)[0]
    print(f'dataset: {dataset}')
    res = []
    for idx, data in tqdm(enumerate(dataset)):
        input = data['messages'][0]['content']
        response = data['messages'][1]['content']
        text = input + response
        if anlysis_text_optimizer[idx] == 0:
            continue
        elif anlysis_text_optimizer[idx] == 1:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的指令部分。\n\n \
            【文本】：{text} \n\n \
            指令部分的优化要求如下：\n1. 确保原始指令的核心意图保持不变。\n2. 提供清晰且相关的解决问题的指导。\n3.指令中没有知识错误。\n4. 避免拼写、语法错误或逻辑失误。\n5. 不包含不必要的信息。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的指令:这儿只能输出优化后的指令\n###理由:这里只能输出理由。"
        elif anlysis_text_optimizer[idx] == 2:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的输入部分。\n\n \
            【文本】：{text} \n\n \
            输入部分的优化要求如下：\n1. 确保原始输入的核心意图保持不变。2. 问题中没有知识错误。\n3.避免拼写、语法错误或逻辑问题。\n4. 不要包含多余信息或过于啰嗦。\n5. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的输入:这儿只能输出优化后的输入\n###理由:这里只能输出理由。"
        elif anlysis_text_optimizer[idx] == 3:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的回答部分。\n\n \
            【文本】：{text} \n\n \
            回答部分的优化要求如下：\n1. 确保原始回答的核心意图保持不变。2. 优化后的回复应准确理解患者的提问并提供相关答案。3. 回答应提供科学、准确的医学知识。\n4.确保语义连贯，无逻辑错误或无关信息\n5. 遵守医学伦理并尊重患者的选择。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的回答:这儿只能输出优化后的回答\n###理由:这里只能输出理由。"
        elif anlysis_text_optimizer[idx] == 4:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的指令和输入部分。\n\n \
            【文本】：{text} \n\n \
            指令部分的优化要求如下：\n1. 确保原始指令的核心意图保持不变。\n2. 提供清晰且相关的解决问题的指导。\n3.指令中没有知识错误。\n4. 避免拼写、语法错误或逻辑失误。\n5. 不包含不必要的信息。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            输入部分的优化要求如下：\n1. 确保原始输入的核心意图保持不变。2. 问题中没有知识错误。\n3.避免拼写、语法错误或逻辑问题。\n4. 不要包含多余信息或过于啰嗦。\n5. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的指令:这儿只能输出优化后的指令\n###优化后的输入:这儿只能输出优化后的输入\n###理由:这里只能输出理由。"
        elif anlysis_text_optimizer[idx] == 5:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的指令和回答部分。\n\n \
            【文本】：{text} \n\n \
            指令部分的优化要求如下：\n1. 确保原始指令的核心意图保持不变。\n2. 提供清晰且相关的解决问题的指导。\n3.指令中没有知识错误。\n4. 避免拼写、语法错误或逻辑失误。\n5. 不包含不必要的信息。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            回答部分的优化要求如下：\n1. 确保原始回答的核心意图保持不变。2. 优化后的回复应准确理解患者的提问并提供相关答案。3. 回答应提供科学、准确的医学知识。\n4.确保语义连贯，无逻辑错误或无关信息\n5. 遵守医学伦理并尊重患者的选择。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的指令:这儿只能输出优化后的指令\n###优化后的回答:这儿只能输出优化后的回答\n###理由:这里只能输出理由。"
        elif anlysis_text_optimizer[idx] == 6:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的输入和回答部分。\n\n \
            【文本】：{text} \n\n \
            输入部分的优化要求如下：\n1. 确保原始输入的核心意图保持不变。2. 问题中没有知识错误。\n3.避免拼写、语法错误或逻辑问题。\n4. 不要包含多余信息或过于啰嗦。\n5. 将长度控制在合理范围内（±30%）。\n\n \
            回答部分的优化要求如下：\n1. 确保原始回答的核心意图保持不变。2. 优化后的回复应准确理解患者的提问并提供相关答案。3. 回答应提供科学、准确的医学知识。\n4.确保语义连贯，无逻辑错误或无关信息\n5. 遵守医学伦理并尊重患者的选择。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的输入:这儿只能输出优化后的输入\n###优化后的回答:这儿只能输出优化后的回答\n###理由:这里只能输出理由。"       
        else:
            prompt = f"你是一个智能文本优化助手。你将会被提供一段【文本】，它由指令（Instruction）、输入（Input）和回答（Response）组成。你的任务是优化文本中的指令、输入和回答部分。\n\n \
            【文本】：{text} \n\n \
            指令部分的优化要求如下：\n1. 确保原始指令的核心意图保持不变。\n2. 提供清晰且相关的解决问题的指导。\n3.指令中没有知识错误。\n4. 避免拼写、语法错误或逻辑失误。\n5. 不包含不必要的信息。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            输入部分的优化要求如下：\n1. 确保原始输入的核心意图保持不变。2. 问题中没有知识错误。\n3.避免拼写、语法错误或逻辑问题。\n4. 不要包含多余信息或过于啰嗦。\n5. 将长度控制在合理范围内（±30%）。\n\n \
            回答部分的优化要求如下：\n1. 确保原始回答的核心意图保持不变。2. 优化后的回复应准确理解患者的提问并提供相关答案。3. 回答应提供科学、准确的医学知识。\n4.确保语义连贯，无逻辑错误或无关信息\n5. 遵守医学伦理并尊重患者的选择。\n6. 将长度控制在合理范围内（±30%）。\n\n \
            你的输出必须严格遵循以下格式：\n###优化后的指令:这儿只能输出优化后的指令\n###优化后的输入:这儿只能输出优化后的输入\n###优化后的回答:这儿只能输出优化后的回答\n###理由:这里只能输出理由。"           

        data['messages'][0]['content'] = prompt
        res.append(InferRequest(**data))

    infer_requests = res
    infer_batch(engine, infer_requests)

    file_path = "data/optimize_model_response.jsonl"

    with open(file_path, 'w', encoding='utf-8') as f:
        for item in result:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/train_data/Chinese-medical-dialogue/train_medical.jsonl")

    args = parser.parse_args()

    main(args)