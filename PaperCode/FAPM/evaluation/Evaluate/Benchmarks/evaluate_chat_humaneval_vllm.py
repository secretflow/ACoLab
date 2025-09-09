import os
import sys
import json
import jsonlines
from data.human_eval.human_eval.evaluation import evaluate_functional_correctness

from vllm_helper import inference


if __name__ == "__main__":

    NOW_TIME = sys.argv[2]

    os.makedirs("result/" + NOW_TIME, exist_ok=True)

    problem_file = "data/human_eval/data_file/HumanEval.jsonl"

    prompt_list = []
    task_id_list = []
    with jsonlines.open(problem_file) as f:
        for jobj in f:
            prompt_list.append(
                {
                    "query": f"Can you complete the following Python function?\n```python\n"
                    + jobj["prompt"].strip()
                    + "\n```"
                }
            )
            task_id_list.append(jobj["task_id"])
    response_list = inference(prompt_list)
    final_list = []
    for i, response in enumerate(response_list):
        answer = response["response"]
        if "\n```" in answer:
            answer = response["response"].split("\n```")[0]
        final_list.append(
            {"task_id": task_id_list[i], "completion": answer, "response": response}
        )

    temp_result_file = "result/" + NOW_TIME + "/humaneval.jsonl"
    with open(temp_result_file, "w", encoding="utf-8") as f:
        for d in final_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    n_workers = 4
    timeout = 3.0
    k = [1]
    results = evaluate_functional_correctness(
        temp_result_file, k, n_workers, timeout, problem_file
    )
    print(results)
    results["detailed"] = final_list

    with open("result/" + NOW_TIME + "/humaneval.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    os.remove(temp_result_file)
