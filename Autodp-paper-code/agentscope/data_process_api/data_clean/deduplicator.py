from datasketch import MinHash, MinHashLSH
import argparse
import pdb
import json
from tqdm import tqdm
import pickle

class Deduplicator:
    def __init__(self,
                data = None,
                ):
        self.data = data
    
    def get_minhash(self, text, num_perm=128):
        """
        将文本转换为 MinHash 对象。
        :param text: 输入文本
        :param num_perm: MinHash 的排列数（影响精度和性能）
        :return: MinHash 对象
        """
        m = MinHash(num_perm=num_perm)
        # 将文本分割为单词（或使用 n-gram）
        words = text.split()
        for word in words:
            m.update(word.encode('utf8'))  # MinHash 需要字节输入
        return m 
    
    def compute_stats(self):

        lsh = MinHashLSH(threshold=0.7, num_perm=128)  # threshold 是相似度阈值
        # 去重后的结果
        # unique_texts = []
        index = []

        for idx, text in tqdm(enumerate(self.data)):
            # 为当前文本生成 MinHash
            m = self.get_minhash(text)
            
            # 检查是否与 LSH 中的其他文本相似
            if len(lsh.query(m)) == 0:  # 如果没有相似的文本
                # unique_texts.append(text)  # 添加到去重结果
                lsh.insert(f"text_{idx}", m)  # 插入 LSH 索引
            else:
                index.append(idx)

        ############对数据进行裁剪############
        
        value = [value for i, value in enumerate(self.data) if i not in index]
        
        return value



# 定义一个函数，将文本转换为 MinHash
# def get_minhash(text, num_perm=128):
#     """
#     将文本转换为 MinHash 对象。
#     :param text: 输入文本
#     :param num_perm: MinHash 的排列数（影响精度和性能）
#     :return: MinHash 对象
#     """
#     m = MinHash(num_perm=num_perm)
#     # 将文本分割为单词（或使用 n-gram）
#     words = text.split()
#     for word in words:
#         m.update(word.encode('utf8'))  # MinHash 需要字节输入
#     return m


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="data/train_medical_optimize_clean.jsonl")

    args = parser.parse_args()

    # data = []
    # with open(args.data_path, 'r', encoding='utf-8') as file:
    #     for line in file:
    #         conversations = json.loads(line.strip())['conversations']
    #         text = conversations[0]['value'] + conversations[1]['value']
    #         data.append(text)

    # # 初始化 LSH 索引
    # lsh = MinHashLSH(threshold=0.7, num_perm=128)  # threshold 是相似度阈值
    # # 去重后的结果
    # # unique_texts = []
    # index = []

    # for idx, text in tqdm(enumerate(data)):
    #     # 为当前文本生成 MinHash
    #     m = get_minhash(text)
        
    #     # 检查是否与 LSH 中的其他文本相似
    #     if len(lsh.query(m)) == 0:  # 如果没有相似的文本
    #         # unique_texts.append(text)  # 添加到去重结果
    #         lsh.insert(f"text_{idx}", m)  # 插入 LSH 索引
    #     else:
    #         index.append(idx)

    # ############对数据进行裁剪############
    
    # value = [value for i, value in enumerate(data) if i not in index]

    # value_sim = [i for i, value in enumerate(data) if i in index]

    # with open("pkl/deduplicator_index_optimize_clean.pkl", 'wb') as file:  # 'wb' 表示以二进制写模式打开文件
    #     pickle.dump(value_sim, file)

    # result = []
    # for d in tqdm(value):
    #     dict = {}
    #     dict['conversations'] = []
    #     dict1 = {}
    #     value1 = d.split('Response:')[0] + 'Response:'
    #     dict1['from'] = 'human'
    #     dict1['value'] = value1
    #     dict2 = {}
    #     value2 = d.split('Response:')[1]
    #     dict2['from'] = 'gpt'
    #     dict2['value'] = value2
    #     dict['conversations'].append(dict1)
    #     dict['conversations'].append(dict2)
    #     result.append(dict)

    
    # path = "data/train_medical_optimize_clean.jsonl"
    # with open(path, 'w', encoding='utf-8') as file:
    #     for item in result:
    #         file.write(json.dumps(item, ensure_ascii=False) + '\n')

