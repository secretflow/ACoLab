# Copyright (c) Alibaba, Inc. and its affiliates.
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdb
import json
import sys
from tqdm import tqdm

# class Evaluation:
#     """返回清洗后的数据"""

#     def __init__(self,
#                 data = None,
#                 ):
#         self.data = data
    
#     def process(self): 

if __name__ == "__main__":

    model_name = "/mnt/data3/nianke_multi_agent/model/Baichuan-M1-14B-Instruct"  
    tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True,torch_dtype = torch.bfloat16).cuda()


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
    with open("/mnt/data3/nianke_multi_agent/agentscope/train_data/huatuo-o1-sft/test_medical.jsonl",'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_ques.append(data['conversations'][0]['value'])

    data_qwen_1_5b_ori = []
    with open("/mnt/data3/nianke_multi_agent/agentscope/generate_test_data/train_medical_sample_dirty.jsonl",'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_qwen_1_5b_ori.append(data['text'])

    data_qwen_1_5b_process = []
    with open("/mnt/data3/nianke_multi_agent/agentscope/generate_test_data/train_medical_sample_clean.jsonl",'r',encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_qwen_1_5b_process.append(data['text'])

    our_win,our_loss,our_tie = 0,0,0

    for i in tqdm(range(len(data_ques))):
        pro = prompt.format(data_ques[i],data_qwen_1_5b_process[i],data_qwen_1_5b_ori[i])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pro}
        ]
    
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 4. Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 5. Decode the generated text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if '赢' in response:
            our_win += 1
        elif '输' in response:
            our_loss +=1
        elif '平' in response:
            our_tie +=1
        else:
            pdb.set_trace()

    for i in tqdm(range(len(data_ques))):
        pro = prompt.format(data_ques[i],data_qwen_1_5b_ori[i],data_qwen_1_5b_process[i])

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": pro}
        ]
    
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 4. Generate text
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # 5. Decode the generated text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if '赢' in response:
            our_loss += 1
        elif '输' in response:
            our_win +=1
        elif '平' in response:
            our_tie +=1
        else:
            pdb.set_trace()

    # print("...............our_win.............:",our_win,our_win/(len(data_ques)*2))
    # print("...............our_loss.............:",our_loss,our_loss/(len(data_ques)*2))
    # print("...............our_tie.............:",our_tie,our_tie/(len(data_ques)*2))

    score = our_win/(len(data_ques)*2) - our_loss/(len(data_ques)*2)

    print("反馈得分：",round(score, 4))
