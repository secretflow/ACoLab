#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name_or_path="/mnt/data3/nianke_multi_agent/model/QwQ-32B"
vllm server ${model_name_or_path} --tensor-parallel-size 4
# port=8006

# while getopts "m:p:" flag
# do
#     # shellcheck disable=SC2220
#     case "${flag}" in
#         m) model_name_or_path=${OPTARG};;
#         p) port=${OPTARG};;
#     esac
# done

# python -m vllm.entrypoints.openai.api_server --model "${model_name_or_path}" \
#   --port "${port}" --tensor-parallel-size 4 --gpu-memory-utilization 0.95 --enforce-eager
