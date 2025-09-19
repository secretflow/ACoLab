from typing import Optional

import regex as re

import argparse
import pdb
import json
from tqdm import tqdm

class RemoveNonEnglishChar:
    """返回清洗后的数据"""

    def __init__(self,
                data_path = None,
                pattern: Optional[str] = None,
                repl: str = '',
                ):
        """
        Initialization method.

        :param pattern: regular expression pattern to search for within text.
        :param repl: replacement string, default is empty string.
        """
        self.data_path = data_path
        self.data = []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                conversations = json.loads(line.strip())['conversations']
                text = conversations[0]['value'] + conversations[1]['value']
                self.data.append(text)

    def compute_stats(self):
        # check if it's computed already

        for idx, stat in tqdm(enumerate(self.data)):
            
            stat = re.sub(r'[^\x00-\x7F0-9.， ,\\-。%《*》/•、&＆(—)（+）：？!！“”·°]+]', '', stat)
            self.data[idx] = stat

        return self.data



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/data/medqa/medqa_multi_agent_dirty_train.jsonl")

    args = parser.parse_args()


    judger = RemoveNonEnglishChar(args.data_path)
    samples_stats = judger.compute_stats()


