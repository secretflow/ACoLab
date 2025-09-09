# FAPM

This GitHub repository contains the code for the paper "Mitigating Catastrophic Forgetting in Large Language Models with Forgetting-aware Pruning."

## Environment
llamafactory == 0.8.2.dev0
peft == 0.11.1
torch == 2.3.0
transformers == 4.47.0.dev0

## How to Use

1. Download the data from Hugging Face and process it into JSON files as illustrated in the examples under the "data" directory.

2. Train the model using Llama Factory. The configuration file used is as follows: llama3_full_sft_winogrande.yaml

3. Execute the FAPM.py file.
You need to input the base model path, the fine-tuned model path, and the path for saving the model.

5. Use the scripts in the evaluation directory for assessment. 

## License
This framework is licensed under the [Apache License 2.0](https://github.com/secretflow/secretflow/blob/main/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## Legal Disclaimer
The Legal Disclaimer of this framework is under the [LEGAL](https://github.com/secretflow/secretflow/blob/main/LEGAL.md)
