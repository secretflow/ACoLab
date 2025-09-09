export MODELSCOPE_CACHE=/mnt2/cache
export HF_DATASETS_CACHE=/mnt2/cache
MASTER_PORT=29508 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /model/llama3_1_8b_instruct \
    --train_type lora \
    --model_type mistral \
    --dataset /data/billsum_train_acl.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --save_strategy steps \
    --save_steps 40 \
    --eval_strategy no \
    --logging_steps 1 \
    --torch_dtype bfloat16 \
    --save_total_limit 20 \
    --output_dir /saves/mistral-7b-instruct/lora/billsum_merge \
    --gradient_checkpointing true \
    --max_length 2048 \
    --lora_rank 8 \
    --target_modules all-linear \
    --lora_alpha 32
