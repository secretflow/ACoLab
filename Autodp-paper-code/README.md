
This GitHub repository contains the code for the paper "LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning."

## Environment
ms-swift == 3.0.0
torch == 2.4.0
transformers == 4.47.0.dev0

## How to Use

### Step 1

The to-be-processed data undergoes data selection and is then trained to obtain scores on dirty data.

、、、
cd agentcode/dataset_sample
python dataset_sample --data_path XX --ratio XX 
cd Swift
./run.sh ##The corresponding data and the running scripts need to be modified.
cd evaluation
python infer_vllm.py --model_path XX --model_type XX --test_data_path XX --out_path agentscope/generate_test_data/XX
、、、

### Step 2

Using a LLM as an agent to automatically acquire the optimal data processing strategy.

、、、
cd agentscope
./script/vllm/vllm_setup_qwen.sh ##启动大模型服务
python meta_agent.py
、、、

Note: Since the data selection employs the LESS algorithm, which requires an older version of the Touch environment, a separate Conda environment needs to be configured. For detailed setup instructions, please refer to https://github.com/princeton-nlp/LESS.

The models used in the paper, such as Baichuan-M1 and Qwen, should be downloaded manually. Some of the training data can be accessed from agentscope/train_data.

### Step 3

Once the data strategy is obtained, apply it to process the full dataset once.

## License
This framework is licensed under the Apache License 2.0. For models and datasets, please refer to the original resource page and follow the corresponding License.

## Legal Disclaimer
The Legal Disclaimer of this framework is under the LEGAL