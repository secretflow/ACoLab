from collections import Counter
from typing import List, Tuple
import argparse
import pdb
import json
from tqdm import tqdm
import pickle

class Wordrepetition:

    def __init__(self,
                data = None):

        self.data = data

    def generate_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str]]:
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def calculate_ngram_repetition_ratio(self, tokens, n: int) -> float:
        ngrams = self.generate_ngrams(tokens, n)
        total_ngrams = len(ngrams)
        if total_ngrams == 0:
            return 0.0
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(count for count in ngram_counts.values() if count > 1)
        return repeated_ngrams / total_ngrams

    def filter_samples_by_ngram_ratio(self,
        sample: None,
        n: int,
        min_ratio: float,
        max_ratio: float
    ) -> List[str]:
        # tokens = sample.split()  # 将句子切分为单词
        tokens = []
        for element in sample:
            tokens.append(element)

        repetition_ratio = self.calculate_ngram_repetition_ratio(tokens, n)
        if min_ratio <= repetition_ratio <= max_ratio:
            return True
        else:
            return False


    def compute_stats(self):
        # check if it's computed already
        n = 2  # 使用 2-gram
        min_ratio = 0.0
        max_ratio = 0.6

        index = []

        for idx, stat in tqdm(enumerate(self.data)):
            filtered_samples = self.filter_samples_by_ngram_ratio(stat, n, min_ratio, max_ratio)
            if filtered_samples:
                continue
            else:
                index.append(idx)
        
        value = [value for i, value in enumerate(self.data) if i not in index]

        result = []
        for d in value:
            dict = {}
            dict['conversations'] = []
            dict1 = {}
            value1 = d.split('医生的回复：')[0].split('患者的问题：')[1]
            dict1['from'] = 'human'
            dict1['value'] = value1
            dict2 = {}
            value2 = d.split('医生的回复：')[1]
            dict2['from'] = 'gpt'
            dict2['value'] = value2
            dict['conversations'].append(dict1)
            dict['conversations'].append(dict2)
            result.append(dict)
        print(len(result))

        return result

    

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="data/train_medical_optimize_clean.jsonl")

    args = parser.parse_args()