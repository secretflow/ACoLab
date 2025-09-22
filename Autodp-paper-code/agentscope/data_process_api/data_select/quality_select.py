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
import pickle

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest'],result):
    request_config = RequestConfig(max_tokens=2048, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        res = resp_list[index].choices[0].message.content
        result.append(res)
    return result


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats

    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_clean/data/train_medical_optimize_clean.jsonl")

    args = parser.parse_args()

    model_path = "/mnt/data1/model/qwen/Qwen2___5-14B-Instruct"
    model_type = 'qwen2_5'

    model = model_path

    infer_backend = 'vllm'

    from swift.llm import VllmEngine
    engine = VllmEngine(model, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=4)


    prompt = "作为一个有专业医学背景的评测人员。首先你需要仔细阅读以下医患对话文本，它由患者的问题和医生的回答组成。接着你的任务是根据五项评价标准及其相应的评分定义来为这条医患对话文本打分。\n\n \
    **文本**：\n \
    {text} \n\n \
    打分标准按以下优先顺序排列：**回答和问题的准确性**，**安全性**，**流畅性**以及**简洁性**。具体定义如下：\n \
    **打分标准**：\n \
    1. **回答和问题的准确性**：\n- 医生应能准确理解患者的问题，并提供科学、准确的回答。\n- 患者的提问应能准确描述其症状、病史或担忧，反映出清晰的思维过程。同时，问题应避免模糊或含糊不清的表述，以便医生能快速理解患者的意图。\n\n \
    2. **安全性**：\n- 医生在回答时需遵循法律法规、伦理和职业道德，确保不会以任何方式给患者带来风险或误导。\n- 患者在提问时要尊重医生，确保不会以任何形式给医生带来安全风险。\n\n \
    3. **流畅性**：\n- 确保语义连贯，无逻辑错误或无关信息。\n\n \
    4. **简洁性**：\n- 医患对话应避免过于复杂的术语和繁琐的表述，以清晰简洁的方式传达信息，使患者更易于理解。\n\n \
    **注意**：\n \
    打分需基于**专业性 > 安全性 > 流畅性**的重要性排序。若发生冲突，则优先考虑前者。\n\n \
    根据综合水平给出1到10的评分。当数据有缺陷时，请严格的进行扣分，以尽可能拉开分数区间。\n \
    你的输出必须严格按照以下格式：\n \
    **评分结果**：\n \
    此处只能给出评分结果。\n \
    **理由**：\n \
    此处只能给出你的理由。"

    dataset = load_dataset([args.data_path], strict=False, seed=42)[0]

    print(f'dataset: {dataset}')
    res = []
    for idx, data in tqdm(enumerate(dataset)):
        input = data['messages'][0]['content']
        response = data['messages'][1]['content']
        pro = prompt.format(input,response)
        dict = {}
        dict['messages'] = []
        dict1 = {}
        dict1['role'] = 'user'
        dict1['content'] = pro

        dict['messages'].append(dict1)
        res.append(dict)
    
    list = []
    infer_requests = [InferRequest(**data) for data in res]
    result = infer_batch(engine, infer_requests, list)

    score = []
    for ans in result:
        if "**评分结果**：" in ans and "**理由" in ans:
            score.append(ans.split('**评分结果**：')[1].split('**理由')[0].strip())
        else:
            score.append(10)

    
    top_k = max(1, int(len(score)*0.2))  # 前 20% 的元素个数，至少为 1 个
    n = len(score)
    # 按值排序并获取值和索引
    sorted_indices = sorted(range(n), key=lambda i: score[i], reverse=True)  # 从大到小排序
    top_indices = sorted_indices[:top_k]  # 取前 20% 的索引

    data = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    data = [value for idx, value in enumerate(data) if idx in top_indices]

    path = "data/train_medical_optimize_clean_select.jsonl"

    with open(path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open("data/medical_optimize_clean_select_index.pkl", 'wb') as file:  # 'wb' 表示以二进制写模式打开文件
        pickle.dump(top_indices, file)
    


    