import json
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb

def load_jsonl(file_path, text_key="text"):
    """
    从 jsonl 文件中加载文本数据，该文件每行是一个 json 对象，
    默认假设文本存储在 "text" 键中。
    """
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 可根据实际情况调整键名
            text = data['conversations'][0]['value'] 
            # + data['conversations'][1]['value']
            texts.append(text)
    return texts

#加载两批数据
texts_a = load_jsonl("/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/train_medical_sample.jsonl") # 第一批数据
texts_b = load_jsonl("/mnt/data3/nianke_multi_agent/agentscope/train_data/huatuo-26m-lite/train_medical.jsonl") # 第二批数据


all_texts = texts_a + texts_b

model = SentenceTransformer("/mnt/data1/model/bge-large-zh-v1.5")
embeddings = model.encode(all_texts, show_progress_bar=True)

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)


plt.figure(figsize=(8, 6))
n_a = len(texts_a)
plt.scatter(embeddings_2d[n_a:, 0], embeddings_2d[n_a:, 1], color="red", label="original dataset", alpha=0.8, s=1.5, marker='o')
plt.scatter(embeddings_2d[:n_a, 0], embeddings_2d[:n_a, 1], color="blue", label="sample dataset", alpha=0.8, s=1.5, marker='o')

plt.legend(fontsize=14)
plt.title("t-SNE distribution of two batches of text data.", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlabel("t-SNE 维度1")
# plt.ylabel("t-SNE 维度2")
plt.show()
plt.savefig("tsne_plot_huatuo_26m_lite.pdf", dpi=300)