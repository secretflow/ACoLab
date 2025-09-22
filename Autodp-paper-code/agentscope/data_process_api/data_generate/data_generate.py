
import subprocess
import time
import argparse
import pdb

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Example script to pass hyperparameters.")

    parser.add_argument("--data_path", type=str, default="/mnt/data3/nianke_multi_agent/agentscope/dq/data_sample/Chinese-medical-dialogue/train_medical_sample.jsonl")

    args = parser.parse_args()


    data_path_question = "/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_generate/train_medical_sample/train_medical_sample_generate_question.jsonl"

    data_path_qa = "/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_generate/train_medical_sample/train_medical_sample_generate_qa.jsonl"

    result = subprocess.run(f"cat {args.data_path} | wc -l", shell=True, capture_output=True, text=True)

    line_count = result.stdout.strip()

    line_count = int(line_count)

    if line_count <= 1000:
        subprocess.run(['python', 'data_process_api/data_generate/generate_qa.py', '--data_path', args.data_path])

        time.sleep(40)

        subprocess.run(['python', 'data_process_api/data_generate/generate_question.py', '--data_path', data_path_qa])

        time.sleep(40)

        subprocess.run(['python', 'data_process_api/data_generate/generate_response.py', '--data_path', data_path_question])

    else:

        subprocess.run(['python', 'data_process_api/data_generate/generate_question.py', '--data_path', args.data_path])

        time.sleep(40)

        subprocess.run(['python', 'data_process_api/data_generate/generate_response.py', '--data_path', data_path_question])




