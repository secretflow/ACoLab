TIMESTAMP=$(date +%Y_%m_%d_%H_%M_%S) # 时间戳
echo ${TIMESTAMP}

CUDA_DEVICES=6
gpu_num=1

# CUDA_DEVICES=0,1,2,3
model_path=your_model_path
model_type=qwen/Qwen2-7B-Instruct
# /mnt/data1/model/qwen/Qwen2-7B-Instruct meta-llama/Meta-Llama-3-8B-Instruct
# gpu_num=4


echo "ceval"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python evaluate_chat_ceval_vllm.py --nowtime ${TIMESTAMP} --model_path ${model_path} --model_type ${model_type} --gpu_num ${gpu_num}
echo "gsm8k"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python evaluate_chat_gsm8k_vllm.py --nowtime ${TIMESTAMP} --model_path ${model_path} --model_type ${model_type} --gpu_num ${gpu_num}
echo "mmlu"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python evaluate_chat_mmlu_vllm.py --nowtime ${TIMESTAMP} --model_path ${model_path} --model_type ${model_type} --gpu_num ${gpu_num}
echo "humaneval"
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python evaluate_chat_humaneval_vllm.py --nowtime ${TIMESTAMP} --model_path ${model_path} --model_type ${model_type} --benchmark humaneval --gpu_num ${gpu_num}
