# Some code here has been modified from:
# https://huggingface.co/spaces/huggingface/text-data-filtering
# -------------------------------------------------------
import string
import pdb
import emoji
import json
import pickle
import argparse
import re
from tqdm import tqdm

class SpecialCharactersFilter:
    """Filter to keep samples with special-char ratio within a specific
    range."""

    _batched_op = True

    def __init__(self,
                data = None):
        """
        :param min_ratio: The min filter ratio in this op, samples will
            be filtered if their special-char ratio is below this
            parameter.
        :param max_ratio: The max filter ratio in this op, samples will
            be filtered if their special-char ratio exceeds this
            parameter.
        """

        self.other_special_characters = (
        "▬…✦�­£​•€«»"
        "˘⇓↓↑←→§¿∈﻿¢ø½¼¾¹²³⁃ʺ¦‐⠀"
        "◆●■►▼▲▴∆▻★☆✱ːº¯˜ɪ†ン♡✓⊕‟ाাी्े◦˚"
        "゜ʼ≖ʼ¤ッツシ‿∞➤πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿▷Г♫∟™ª₪®「❖"
        "」﴾》"
        )

        self.emoji = list(emoji.EMOJI_DATA.keys())
        self.special_characters = set(self.other_special_characters)
        self.special_characters.update(self.emoji)
        self.data = data
        # with open(self.data_path, 'r', encoding='utf-8') as file:
        #     for line in file:
        #         conversations = json.loads(line.strip())['conversations']
        #         text = conversations[0]['value'] + conversations[1]['value']
        #         self.data.append(text)

    def compute_stats(self):

        for idx, stat in enumerate(self.data):
            for element in stat:
                if element in self.special_characters:
                    self.data[idx].replace(element,'')

        result = []
        for d in tqdm(self.data):
            result.append(d)

        print(len(result))
        
        return result

        
        # path = "data/train_medical_optimize_clean.jsonl"
        # with open(path, 'w', encoding='utf-8') as file:
        #     for item in result:
        #         file.write(json.dumps(item, ensure_ascii=False) + '\n')

        # completed = "Execution completed."
        # return completed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="data/train_medical_optimize_clean.jsonl")

    args = parser.parse_args()


