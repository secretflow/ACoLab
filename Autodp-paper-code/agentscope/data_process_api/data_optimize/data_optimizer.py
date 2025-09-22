import subprocess
import time
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/Chinese-medical-dialogue/train_medical_sample.jsonl")

    args = parser.parse_args()

    ################获取需要优化的文本#############

    subprocess.run(['python', 'data_process_api/data_optimize/clean_or_dirty.py', '--data_path', args.data_path])


    data_path_pre = "/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_optimize/train_medical_sample/train_medical_sample_optimizer_question.jsonl"

    subprocess.run(['python', 'data_process_api/data_optimize/optimizer_question.py', '--data_path', args.data_path])

    time.sleep(40)

    subprocess.run(['python', 'data_process_api/data_optimize/optimizer_response.py', '--data_path', data_path_pre])





