import torch
import pdb
from tqdm import tqdm

aa = torch.zeros(32).cuda()

for step in tqdm(range(40)):


    lora_a_grads = torch.load(f"prune/lora_grad_llama/lora_grade_piqa/lora_grad_lora_a_step{step}_llama.pth")
    lora_a_grad = []
    for k,v in lora_a_grads.items():
        lora_a_grad.append(v)

    lora_b_grads = torch.load(f"prune/lora_grad_llama/lora_grade_piqa/lora_grad_lora_b_step{step}_llama.pth")
    lora_b_grad = []
    for k,v in lora_b_grads.items():
        lora_b_grad.append(v)

    for i in range(0,len(lora_a_grad),7):
        for j in range(i,i+7):
            exp_avg_a = lora_a_grad[j]
            exp_avg_b = lora_b_grad[j]
            ten = torch.norm(torch.mm(exp_avg_b,exp_avg_a))
            # ten = torch.norm(torch.mm(exp_avg_b,exp_avg_a))
            aa[int(i/7)] += ten
            # bb[int(i/14)] += exp_avg_b

    del lora_a_grads
    del lora_b_grads

print(aa)

