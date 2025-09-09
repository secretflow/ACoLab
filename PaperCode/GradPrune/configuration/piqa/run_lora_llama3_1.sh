export MODELSCOPE_CACHE=/mnt2/cache
export HF_DATASETS_CACHE=/mnt2/cache
nproc_per_node=4 \
MASTER_PORT=29506 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /model/llama3_1_8b_instruct \
    --train_type lora \
    --model_type llama3_1 \
    --dataset /data/piqa_train_acl.json \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --save_strategy epoch \
    --eval_strategy no \
    --model_author swift \
    --model_name swift-robot \
    --deepspeed zero2 \
    --logging_steps 5 \
    --torch_dtype bfloat16 \
    --save_total_limit 1 \
    --output_dir /saves/llama3-1-8b-instruct/lora/piqa_acl_swift \
    --gradient_checkpointing true \
    --max_length 2048 \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32
