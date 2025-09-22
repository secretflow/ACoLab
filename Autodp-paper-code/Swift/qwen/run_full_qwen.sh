nproc_per_node=4 \
MASTER_PORT=29501 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model /mnt/data1/model/Qwen2.5-1.5B-Instruct \
    --train_type full \
    --model_type qwen2_5 \
    --dataset $1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --save_strategy epoch \
    --eval_strategy no \
    --deepspeed zero2 \
    --logging_steps 5 \
    --torch_dtype bfloat16 \
    --save_total_limit 1 \
    --output_dir $2 \
    --gradient_checkpointing true \
    --max_length 2560
