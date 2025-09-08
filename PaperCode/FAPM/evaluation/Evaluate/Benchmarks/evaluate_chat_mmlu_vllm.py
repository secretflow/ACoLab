import os
import re
import sys
import json
import pandas as pd
from thefuzz import process

from vllm_helper import inference

def format_example(line):
    example = (
        "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
        + line["question"]
        + "\n"
    )
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(
        response, {choice: row[choice] for choice in choices}
    )
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred

def eval_subject(
    subject_name_list,
    test_df_list,
):


    json_result_dict = {}
    correct_dict = {}
    question_list = []
    for test_df in test_df_list:
        for _, row in test_df.iterrows():
            question = format_example(row)
            question_list.append({"query": question})

    response_list = inference(question_list)
    global_index = 0
    for i, test_df in enumerate(test_df_list):
        subject_name = subject_name_list[i]
        score = []
        responses = []
        result = []
        for _, row in test_df.iterrows():
            response = response_list[global_index]["response"]
            global_index += 1
            pred = extract_answer(response, row)

            if "answer" in row:
                correct = 1 if pred == row["answer"] else 0
                score.append(correct)
            responses.append(response)
            result.append(pred)

        test_df["model_output"] = result
        test_df["model_response"] = response
        if score:
            test_df["correctness"] = score

        test_json_str = test_df.to_json(orient="records")
        test_json = json.loads(test_json_str)
        json_result_dict[subject_name] = test_json
        correct_dict[subject_name] = score

    return json_result_dict, correct_dict


def cal_mmlu(res):
    final_result = {}

    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])

    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            final_result[k] = acc_sum_dict[k] * 100 / cnt_dict[k]
            # print("%s ACC: %.2f " % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
    final_result["AVERAGE ACC"] = acc_sum * 100 / cnt
    # print("AVERAGE ACC:%.2f " % (acc_sum * 100 / cnt))
    return final_result

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]

if __name__ == "__main__":

    NOW_TIME = sys.argv[2]

    os.makedirs("result/" + NOW_TIME, exist_ok=True)

    test_df_list = []
    subject_name_list = []
    for subject_name in SUBJECTS:
        test_file_path = os.path.join(
            "data/mmlu/data", "test", f"{subject_name}_test.csv"
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        ).astype(str)
        test_df_list.append(test_df)
        subject_name_list.append(subject_name)
    json_result_dict, dev_result = eval_subject(
        subject_name_list,
        test_df_list,
    )
    final_result = cal_mmlu(dev_result)
    print(final_result)
    final_json = {}
    final_json["result"] = final_result
    final_json["detailed"] = json_result_dict
    with open("result/" + NOW_TIME + "/mmlu.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)