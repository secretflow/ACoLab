#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name_or_path="../model/Qwen3-32B"
port=8000

while getopts "m:p:" flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        m) model_name_or_path=${OPTARG};;
        p) port=${OPTARG};;
    esac
done

python -m vllm.entrypoints.openai.api_server --model "${model_name_or_path}" \
  --port "${port}" --tensor-parallel-size 4 --gpu-memory-utilization 0.9 --enforce-eager
