from typing import Optional

import regex as re

import argparse
import pdb
import json
from tqdm import tqdm

class CleanNoiseMapper:
    """返回清洗后的数据"""

    def __init__(self,
                data = None,
                pattern: Optional[str] = None,
                repl: str = '',
                ):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        """
        # self.data_path = data_path
        # self.data = []
        # with open(self.data_path, 'r', encoding='utf-8') as file:
        #     for line in file:
        #         conversations = json.loads(line.strip())['conversations']
        #         text = conversations[0]['value'] + conversations[1]['value']
        #         self.data.append(text)
        self.data = data

        if pattern is None:
            self.pattern_email = r'[A-Za-z0-9.\-+_]+@[a-z0-9.\-+_]+\.[a-z]+'

            self.pattern_ip = r'(?:(?:1[0-9][0-9]\.)|(?:2[0-4][0-9]\.)|'
            self.pattern_ip += r'(?:25[0-5]\.)|(?:[1-9][0-9]\.)|(?:[0-9]\.))'
            self.pattern_ip += r'{3}(?:(?:1[0-9][0-9])|(?:2[0-4][0-9])|'
            self.pattern_ip += r'(?:25[0-5])|(?:[1-9][0-9])|(?:[0-9]))|'
            self.pattern_ip += r'([\da-fA-F]{1,4}:){7}[\da-fA-F]{1,4}'  # ipv6

            self.pattern_link = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\$ \ $,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


            self.patterns_to_remove = r"\s{2,}"  


        self.repl = repl

    def compute_stats(self):
        # check if it's computed already

        for idx, stat in tqdm(enumerate(self.data)):

            if re.search(self.pattern_email, stat, flags=re.DOTALL):
                stat = re.sub(pattern=self.pattern_email, repl=self.repl, string=stat, flags=re.DOTALL)

            if re.search(self.pattern_ip, stat, flags=re.DOTALL):
                stat = re.sub(pattern=self.pattern_ip, repl=self.repl, string=stat, flags=re.DOTALL)

            if re.search(self.pattern_link, stat, flags=re.DOTALL):
                stat = re.sub(pattern=self.pattern_link, repl=self.repl, string=stat, flags=re.DOTALL)

            if re.search(self.patterns_to_remove, stat):
                stat = re.sub(pattern=self.patterns_to_remove, repl=' ', string=stat, flags=re.DOTALL)

            self.data[idx] = stat

        result = []
        for idx,d in tqdm(enumerate(self.data)):
            result.append(d)

        return result

        
        # path = "data/train_medical_optimize_clean.jsonl"
        # with open(path, 'w', encoding='utf-8') as file:
        #     for item in result:
        #         file.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/data/train_medical_optimize.jsonl")

    args = parser.parse_args()


