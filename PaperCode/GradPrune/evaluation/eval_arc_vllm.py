import os
from typing import List
import pdb
import json
from bert_score import score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calculate_rouge_l(reference_text, candidate_text):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(reference_text, candidate_text)
    return scores['rougeL']

result = []
def infer_batch(engine: 'InferEngine', infer_requests: List['InferRequest']):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    metric = InferStats()
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
    for index, response in enumerate(resp_list):
        res = resp_list[index].choices[0].message.content
        result.append(res)


if __name__ == '__main__':
    from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, load_dataset
    from swift.plugin import InferStats

    model_path = sys.argv[1]
    model_type = sys.argv[2]

    with open("/mnt2/name_prune/data/arc_test_acl.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    label_list = []

    for idx,element in enumerate(data):
        conversation = element['conversations']
        label_list.append(conversation[1]['value'])

    # llama3_1, gemma2, glm4, internlm2, mistral_nemo, phi3_small, qwen2_5, yi

    name = model_path.split("/")[-3]
    model = model_path

    infer_backend = 'pt'

    if infer_backend == 'pt':
        engine = PtEngine(model, model_type=model_type, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, model_type=model_type,gpu_memory_utilization=0.95,tensor_parallel_size=1)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    dataset = load_dataset(['/mnt2/name_prune/data/arc_test_acl.json'], strict=False, seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    rouge_pre = []
    rouge_f1 = []

    if len(result) != len(label_list):
            print("error")

    accuracy = accuracy_score(label_list, result)
    print("*********Model:{} Method:{}**********".format(model_type,name))
    print("The Accuracy of arc is {:.3f}".format(accuracy))  
    print("**********************************************************")
    


