from transformers import AutoModelForCausalLM,AutoModel
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import Trainer
from torch import nn
from safetensors.torch import load_file
import heapq
import torch
import random
import os
import pdb
import json
import pandas as pd
from tqdm import tqdm
from torch.nn.functional import cosine_similarity

features_ln_output1 = []
features_ln_output2 = []

def spars(tensor):

    k = int(tensor.numel() * 0.2)

    indices = torch.argsort(tensor.abs().view(-1), descending=True)[:k]

    mask = torch.zeros_like(tensor)
    mask.view(-1)[indices] = 1
    return mask

# path_model = "/mnt2/nianke/model/Mistral-7B-Instruct-v0.3"

path_model = "/model/llama3_1_8b_instruct"

tokenizer = AutoTokenizer.from_pretrained(path_model, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(path_model, torch_dtype=torch.bfloat16).cuda()

config = AutoConfig.from_pretrained(path_model)

idx = [torch.tensor(18), torch.tensor(19), torch.tensor(20), torch.tensor(21), torch.tensor(22), torch.tensor(23), torch.tensor(24), torch.tensor(25), torch.tensor(26), torch.tensor(27), torch.tensor(28), torch.tensor(29),torch.tensor(30)]

# idx = [torch.tensor(19), torch.tensor(20), torch.tensor(21), torch.tensor(22), torch.tensor(23), torch.tensor(24), torch.tensor(25), torch.tensor(26), torch.tensor(27), torch.tensor(28), torch.tensor(29),torch.tensor(30)]


layers = model.model.layers

lora_dict = {}
for step in tqdm(range(40)):

    lora_a_grads = torch.load(f"prune/lora_grad_mistral/lora_grade_arc/lora_grad_lora_a_step{step}_llama.pth")
    lora_b_grads = torch.load(f"prune/lora_grad_mistral/lora_grade_arc/lora_grad_lora_b_step{step}_llama.pth")
    for k,v in lora_a_grads.items():
        score = torch.square(torch.mm(lora_b_grads[k.replace('lora_A','lora_B')],lora_a_grads[k]))
        if k.replace('.lora_A','') not in lora_dict:
            lora_dict[k.replace('.lora_A','')] = score
        else:
            lora_dict[k.replace('.lora_A','')] += score
    del lora_a_grads
    del lora_b_grads

################### 13 层 ##############
for (name1, param1), (name2, param2), (name3, param3), (name4, param4)  in zip(layers[idx[0]-1].named_parameters(), layers[idx[0]].named_parameters(), layers[idx[1]].named_parameters(), layers[idx[12]].named_parameters()):
    if 'input_layernorm' in name1 or  'post_attention_layernorm' in name1:
        continue
    if name1 == name2:

        # name = name2.replace('weight','default.weight')
        # marix = lora_dict[f'base_model.model.model.layers.18.{name}']

        # mask = spars(marix).cuda()

        name = name2.replace('weight','default.weight')
        marix = lora_dict[f'base_model.model.model.layers.18.{name}']

        mask_18 = spars(marix).cuda()

        name = name3.replace('weight','default.weight')
        marix = lora_dict[f'base_model.model.model.layers.19.{name}']

        mask_19 = spars(marix).cuda()

        name = name3.replace('weight','default.weight')
        marix = lora_dict[f'base_model.model.model.layers.30.{name}']

        mask_30 = spars(marix).cuda()

        # same_sign = (param1 * param2 * param3 * param4) > 0

        # same_sign = (param1 * param2) > 0
        same_sign_18 = (param1 * param2) > 0
        same_sign_19 = (param1 * param3) > 0
        same_sign_30 = (param1 * param4) > 0


        # merge_mask = mask.bool() & same_sign
        merge_mask_18 = mask_18.bool() & same_sign_18
        merge_mask_19 = mask_19.bool() & same_sign_19
        merge_mask_30 = mask_30.bool() & same_sign_30

        # merge_mask = merge_mask.float()
        merge_mask_18 = merge_mask_18.float()
        merge_mask_19 = merge_mask_19.float()
        merge_mask_30 = merge_mask_30.float()

        param2.requires_grad = False
        param3.requires_grad = False
        param4.requires_grad = False
        # param4.requires_grad = False

        new_param = param1 + param2.mul_(merge_mask_18) + param3.mul_(merge_mask_19) + param4.mul_(merge_mask_30)

        with torch.no_grad():
            param1.copy_(new_param)


layers_bottom = layers[:idx[0]]

layers_top = layers[idx[12]+1:]
student = nn.ModuleList()
for ans in layers_bottom:
    student.append(ans)

for ans in layers_top:
    student.append(ans)
model.model.layers = student

total_params = sum(p.numel() for p in model.parameters())

print(f"Model has {total_params} parameters.")

new_config = AutoConfig.from_pretrained(path_model)
new_config.num_hidden_layers = len(model.model.layers)

model.config = new_config

print(".............success...........")

model.save_pretrained('llama_nomerge_10/layer_prune_llama_important_nomerge_10', safe_serialization=False)
tokenizer.save_pretrained('llama_nomerge_10/layer_prune_llama_important_nomerge_10')
