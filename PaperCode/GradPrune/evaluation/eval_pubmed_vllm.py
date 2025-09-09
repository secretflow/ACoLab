import os
from typing import List
import pdb
import json
from bert_score import score
from rouge_score import rouge_scorer
import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

    with open("/mnt2/name_prune/data/pubmed_test_acl.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    label_list = []

    for idx,element in enumerate(data):
        conversation = element['conversations']
        label_list.append(conversation[1]['value'])

    # llama3_1, gemma2, glm4, internlm2, mistral_nemo, phi3_small, qwen2_5, yi
    infer_backend = 'pt'

    name = model_path.split("/")[-3]
    
    model = model_path

    if infer_backend == 'pt':
        engine = PtEngine(model, model_type=model_type, max_batch_size=64)
    elif infer_backend == 'vllm':
        from swift.llm import VllmEngine
        engine = VllmEngine(model, model_type=model_type)
    elif infer_backend == 'lmdeploy':
        from swift.llm import LmdeployEngine
        engine = LmdeployEngine(model)

    dataset = load_dataset(['/mnt2/name_prune/data/pubmed_test_acl.json'], strict=False, seed=42)[0]
    print(f'dataset: {dataset}')
    infer_requests = [InferRequest(**data) for data in dataset]
    infer_batch(engine, infer_requests)

    rouge_pre = []
    rouge_f1 = []

    if len(result) != len(label_list):
            print("error")
    else:
        for i in range(len(result)):
            candidate_text = result[i]
            reference_text = label_list[i]
            rouge_l_score = calculate_rouge_l(reference_text, candidate_text)
            rouge_pre.append(rouge_l_score[0])
            rouge_f1.append(rouge_l_score[2])      
 
    P, R, F1 = score(result, label_list, model_type='roberta-large', lang="en", verbose=True)  

    print("*********Model:{} Method:{}**********".format(model_type,name))
    print("#############rouge-f1 score#############",sum(rouge_f1)/len(rouge_f1))
    print("#############bertscore_F1 score#############",sum(F1)/len(F1))
    print("#############all score pubmed#############",(sum(F1)/len(F1)+sum(rouge_f1)/len(rouge_f1))/2)

