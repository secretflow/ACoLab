from deduplicator import Deduplicator
from clean_noise import CleanNoiseMapper
from special_characters_filter import SpecialCharactersFilter
from token_num_filter import TokenNumFilter
from word_repetition_filter import Wordrepetition
import argparse
import json
import pdb


parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/Chinese-medical-dialogue/train_medical_sample.jsonl")

args = parser.parse_args()

tokenizer_model_path = "/mnt/data1/model/qwen/Qwen2___5-7B-Instruct"

data = []
with open(args.data_path, 'r', encoding='utf-8') as file:
    for line in file:
        conversations = json.loads(line.strip())['conversations']
        text = '患者的问题：' + conversations[0]['value'] + '医生的回复：' + conversations[1]['value']
        data.append(text)

judger = Deduplicator(data)
data_clean = judger.compute_stats()

judger = CleanNoiseMapper(data_clean)
data_clean = judger.compute_stats()

judger = SpecialCharactersFilter(data_clean)
data_clean = judger.compute_stats()

judger = TokenNumFilter(data_clean, tokenizer_model_path, 20, 2560)
data_clean = judger.compute_stats()

judger = Wordrepetition(data_clean)
data_clean = judger.compute_stats()

with open('/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_clean/train_medical_sample/train_medical_sample_clean.jsonl', 'w', encoding='utf-8') as file:
    for entry in data_clean:
        json_line = json.dumps(entry, ensure_ascii=False)
        file.write(json_line + '\n')


