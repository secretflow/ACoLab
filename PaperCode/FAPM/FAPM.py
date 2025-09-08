from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdb
import json
path = "/mnt/data1/model/meta-llama/Meta-Llama-3-8B-Instruct"
# #/mnt/data1/model/qwen/Qwen2-7B-Instruct
#/mnt/data1/model/meta-llama/Meta-Llama-3-8B-Instruct
path_round1_lr1e_5 = "*/Llama3-MRPC-5e-6"


model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()
model_round1_lr1e_5 = AutoModelForCausalLM.from_pretrained(path_round1_lr1e_5, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().cuda()


weights = model.state_dict()
weights_round1_lr1e_5 = model_round1_lr1e_5.state_dict()

for key,value in weights.items():
    print(key)
    # if 'embed' in key or 'norm' in key:
    #     continue

    tensor = weights_round1_lr1e_5[key] - weights[key]

    k = int(tensor.numel() * 0.1)

    lamda = 1.0 * weights[key].abs().mean()

    t = tensor.abs() - lamda*(tensor.abs() / weights[key].abs())

    indices = torch.argsort(t.view(-1), descending=True)[:k]

    # indices = torch.argsort(tensor.abs().view(-1), descending=True)[:k]

    mask = torch.zeros_like(tensor)
    mask.view(-1)[indices] = 1
    tensor.mul_(mask)

    weights[key] = weights[key] + tensor
    

torch.save(weights, "*/pytorch_model.bin")


# def FAPM(W_pre, W_tv, key):
#     ### W_pre denotes the parameters of the pre-trained model.
#     ### W_tv epresents the task vector, which is the difference between the parameters of the fine-tuned model and the pre-trained model.
#     ### key represents one of the parameter matrices in the model.
#     S = W_tv[key].abs() - W_pre[key].abs().mean() * (W_tv[key].abs() / W_pre[key].abs())
#     return S



