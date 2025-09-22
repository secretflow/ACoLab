import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
import pdb
import torch
import copy
from clean_or_dirty import AnlysisDataset
from function import GraphCut
from optimizer import NaiveGreedy
import random
import json

random.seed(42)


def dataset_quantization(data, ratio=0.2, embed=None):
    
    budget_n = int(len(data) * ratio)

    embeddings_original = embed
    embeddings_original = torch.from_numpy(embeddings_original).cuda()
    embeddings = copy.deepcopy(embeddings_original)

    indices_original = np.arange(len(data))
    indices_original = torch.from_numpy(indices_original).cuda()
    indices = copy.deepcopy(indices_original)

    sim_matrix = lambda a, b: embeddings[a] @ embeddings[b].T


    submod_f = GraphCut(index=indices, similarity_kernel=sim_matrix)
    submod_opt = NaiveGreedy(args=None, index=indices, budget=budget_n)
    result_indices = submod_opt.select(
        gain_function=submod_f.calc_gain,
        update_state=submod_f.update_state,
    )

    data = [data[i] for i in result_indices]

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/train_data/huatuo-o1-sft/train_medical.jsonl")
    parser.add_argument("--ratio", type=float, default=0.2)
    args = parser.parse_args()

    data = []
    with open(args.data_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    anlysis = AnlysisDataset(args.data_path)

    noise = anlysis.process()

    data_clean,data_dirty = [], []

    for i in range(len(noise)):
        if noise[i] == 0:
            data_dirty.append(data[i])
        else:
            data_clean.append(data[i])

    model = SentenceTransformer("/mnt/data1/model/bge-large-zh-v1.5")


    sentences = []
    for d in data_dirty:
        conversations = d['conversations']
        inputs = conversations[0]['value']
        outputs = conversations[1]['value']
        text = "患者的问题：\n" + inputs + '\n医生的回答：\n' + outputs
        sentences.append(text)

    embeddings = model.encode(sentences, show_progress_bar=True)

    ratio = args.ratio
    data_dirty_sample = dataset_quantization(data_dirty, ratio=ratio, embed=embeddings)


    sentences = []
    for d in data_clean:
        conversations = d['conversations']
        inputs = conversations[0]['value']
        outputs = conversations[1]['value']
        text = "患者的问题：\n" + inputs + '\n医生的回答：\n' + outputs
        sentences.append(text)

    embeddings = model.encode(sentences, show_progress_bar=True)

    ratio = args.ratio
    data_clean_sample = dataset_quantization(data_clean, ratio=ratio, embed=embeddings)

    datas = data_dirty_sample + data_clean_sample

    random.shuffle(datas)

    with open('/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/train_medical_sample.jsonl', 'w', encoding='utf-8') as file:
        for entry in datas:
            json_line = json.dumps(entry, ensure_ascii=False)
            file.write(json_line + '\n')
