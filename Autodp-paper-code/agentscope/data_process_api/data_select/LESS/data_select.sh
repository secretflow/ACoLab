
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nianke_cf
# ##############步骤1，用少量数据训练lora模型###########
DATA_DIR=$1
Llama_MODEL_PATH=/mnt/data1/model/meta-llama/Llama-2-7b-chat-hf
PERCENTAGE=0.05
DATA_SEED=3
JOB_NAME=llama2-7b-train_medical_sample

rm -rf data_process_api/data_select/out/$JOB_NAME

./data_process_api/data_select/LESS/less/scripts/train/warmup_lora_train.sh "$DATA_DIR" "$Llama_MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"

# ##############步骤2，获取训练数据的梯度###########

max_checkpoint=$(ls -d data_process_api/data_select/out/$JOB_NAME/checkpoint-* 2>/dev/null \
  | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2}' \
  | sort -n \
  | tail -1)

CKPT=$max_checkpoint
TRAINING_DATA_NAME=train_medical_sample
TRAINING_DATA_FILE=/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_select/LESS/data/cosmos.json # when changing data name, change the data path accordingly
GRADIENT_TYPE="adam"
MODEL_PATH=data_process_api/data_select/out/$JOB_NAME/checkpoint-$CKPT
Train_Grad_OUTPUT_PATH=data_process_api/data_select/grads/llama2-7b-Chinese-medical-dialogue/${TRAINING_DATA_NAME}
DIMS="8192"

rm -rf $Train_Grad_OUTPUT_PATH

./data_process_api/data_select/LESS/less/scripts/get_info/grad/get_train_lora_grads.sh "$DATA_DIR" "$MODEL_PATH" "$Train_Grad_OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE"

# ##############步骤3，获取验证集数据的梯度###########

TASK=valid
Valid_Grad_OUTPUT_PATH=data_process_api/data_select/grads/llama2-7b-Chinese-medical-dialogue/less-valid-data-sgd # for validation data, we always 
Valid_DATA_DIR=/mnt/data3/nianke_multi_agent/agentscope/data_process_api/data_select/LESS/less_valid_data.jsonl
DIMS="4096 8192" # We use 8192 as our default projection dimension 

rm -rf $Valid_Grad_OUTPUT_PATH

./data_process_api/data_select/LESS/less/scripts/get_info/grad/get_eval_lora_grads.sh "$TASK" "$Valid_DATA_DIR" "$MODEL_PATH" $Valid_Grad_OUTPUT_PATH "$DIMS"

# ##############步骤4，训练集和验证集的梯度匹配###########

checkpoints=$(ls -d data_process_api/data_select/out/$JOB_NAME/checkpoint-* 2>/dev/null \
  | awk -F'checkpoint-' '/checkpoint-[0-9]+$/ {print $2}' \
  | sort -n)

DIM=8192 # decide which dimension to use
GRADIENT_PATH=$Train_Grad_OUTPUT_PATH/dim${DIM}
TRAIN_FILE_NAMES="train_medical_sample"
CKPTS=$checkpoints # checkpoing index
CHECKPOINT_WEIGHTS="1.6877e-05 1.2859e-05 7.7030e-06 2.5616e-06" # average lr of the epoch

VALIDATION_GRADIENT_PATH=$Valid_Grad_OUTPUT_PATH/dim${DIM}
TARGET_TASK_NAMES="train_medical_sample"
SELECTED_DATA_OUTPUT_PATH="data_process_api/data_select/selected_data"

rm -rf $SELECTED_DATA_OUTPUT_PATH

./data_process_api/data_select/LESS/less/scripts/data_selection/matching.sh "$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$SELECTED_DATA_OUTPUT_PATH"

# ##############步骤5，从训练集中挑选高质量数据###########

SELECTED_DATA_OUTPUT_PATH="data_process_api/data_select/selected_data"
python3 -m less.data_selection.write_selected_data \
--target_task_names ${TARGET_TASK_NAMES} \
--train_file_names ${TRAIN_FILE_NAMES} \
--train_files $DATA_DIR \
--output_path $SELECTED_DATA_OUTPUT_PATH \
--percentage 0.5

conda deactivate 