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

def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest'],result):
    request_config = RequestConfig(max_tokens=8192, temperature=0.6)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        res = resp_list[index].choices[0].message.content
        result.append(res)
    return result


if __name__ == '__main__':
    parser.add_argument("--test_data_path", type=str, default="qwen2_5")
    parser.add_argument("--ori_generate_data_path", type=str, default="qwen2_5")
    parser.add_argument("--after_process_generate_data_path", type=str, default="qwen2_5")

    args = parser.parse_args()

    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats

    model_path = "/mnt/data3/nianke_multi_agent/model/QwQ-32B"
    model_type = 'qwq'

    model = model_path

    infer_backend = 'vllm'

    from swift.llm import VllmEngine
    engine = VllmEngine(model, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=4)

    prompt = "作为一个有专业医学背景的评测人员，请对以下两个对同一医学问题的医生回答进行评价。\n\n \
    **问题**：\n{}\n**回答1**：\n{}\n**回答2**：{}\n\n \
    评价标准按以下优先顺序排列：**医生回答的准确性**，**安全性**，**流畅性**以及**简洁性**。具体定义如下：\n \
    **评价标准**：\n \
    1. **医生回答准确性**：医生应能准确理解患者的问题，并提供科学、准确的回答。\n \
    2. **安全性**：\n- 医生在回答时需遵循法律法规、伦理和职业道德。\n \
    4. **流畅性**：\n- 确保语义连贯，无逻辑错误或无关信息。保持友好、热情的回答态度。\n \
    5. **简洁性**：\n- 清晰简洁地解释复杂医学知识。避免对话内容过于冗余。\n\n \
    **注意**：\n \
    评价需基于**医生回答的准确性  > 安全性 > 流畅性 > 简洁性**的重要性排序。若发生冲突，则优先考虑前者。\n \
    你需要要从以下三个选项中选出你的评价答案：[回答1相对于回答2的结果为赢，回答1相对于回答2的结果为平，回答1相对于回答2的结果为输] \n \
    你的输出必须严格按照以下格式：\n \
    **评价结果**：\n \
    此处只能给出选择的评价结果。"

    data_ques = []
    with open(args.test_data_path,'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_ques.append(data['conversations'][0]['value'])

    data_qwen_1_5b_ori = []
    with open(args.ori_generate_data_path,'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_qwen_1_5b_ori.append(data['text'])

    data_qwen_1_5b_process = []
    with open(args.after_process_generate_data_path,'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_qwen_1_5b_process.append(data['text'])

    our_win,our_loss,our_tie = 0,0,0

    res = []
    for i in tqdm(range(len(data_ques))):
        pro = prompt.format(data_ques[i],data_qwen_1_5b_process[i],data_qwen_1_5b_ori[i])
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

    num = 0

    for response in result:
        if '</think>' in response:
            respons = response.split('</think>')[1]
        else:
            our_tie +=1
        if '回答1相对于回答2的结果为赢' in response:
            our_win += 1
        elif '回答1相对于回答2的结果为输' in response:
            our_loss +=1
        elif '回答1相对于回答2的结果为平' in response:
            our_tie +=1
        else:
            pdb.set_trace()

    print("...............our_win.............:",our_win,our_win/(len(data_ques)))
    print("...............our_loss.............:",our_loss,our_loss/(len(data_ques)))
    print("...............our_tie.............:",our_tie,our_tie/(len(data_ques)))