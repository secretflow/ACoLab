import sys
from transformers import AutoTokenizer
import argparse
import pdb
import json
from tqdm import tqdm

class TokenNumFilter:
    """Filter to keep samples with total token number within a specific
    range."""

    def __init__(self,
                data = None,
                model_path = None,
                min_num: int = 10,
                max_num: int = sys.maxsize):
        """
        :param min_num: The min filter token number in this op, samples
            will be filtered if their token number is below this
            parameter.
        :param max_num: The max filter token number in this op, samples
            will be filtered if their token number exceeds this
            parameter.
        """
        self.model_path = model_path
        self.min_num = min_num
        self.max_num = max_num
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.data = data
        # with open(self.data_path, 'r', encoding='utf-8') as file:
        #     for line in file:
        #         conversations = json.loads(line.strip())['conversations']
        #         text = conversations[0]['value'] + conversations[1]['value']
        #         self.data.append(text)

    def compute_stats(self):
        # check if it's computed already
        index = []

        for idx, stat in enumerate(self.data):
            if len(stat) == 0:
                num = 0.0
            else:
                num = len(self.tokenizer.tokenize(stat))

            if num >= self.min_num and num <= self.max_num:
                continue
            else:
                index.append(idx)

        
        value = [value for i, value in enumerate(self.data) if i not in index]

        print(len(value))
        
        return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/data/medqa/medqa_multi_agent_dirty_train.jsonl")
    parser.add_argument("--model_path", type=str, default="/mnt/data1/model/meta-llama/Meta-Llama-3___1-8B-Instruct")

    args = parser.parse_args()

