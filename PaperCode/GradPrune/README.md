# GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs
This repo contains the code for our submitted paper.

Download the dataset from Hugging Face.

## Requirements
**•** ms-swift == 3.0.0

**•** torch == 2.4.0

**•** transformers == 4.45.2

**•** vllm == 0.6.3.post1

**•** peft == 0.12.0

**•** python == 3.11.9

## Layer Prune Pipeline
### Step1: Lora Training
The experiments showed that a significant drop in loss occurs during the initial phase of model training (approximately within the first 40 steps). This phase represents the period during which the model rapidly learns downstream tasks. Accompanying this phenomenon, layers of the model exhibit varying degrees of learning capability. Gradient updates, as a method for adjusting model parameters, effectively reflect the extent to which the model adapts to and responds to downstream tasks. Consequently, the gradient updates during the early stages of training were utilized to assess the importance of different layers within the model.

```
run_lora_llama3_1_layer_important.sh #Add the paths for path_model, data_path and save_path in the file.
```

*When obtaining the gradients in this step, you need to replace the trainer.py file in the Transformers package with the file we provided, and modify the gradient-saving paths in lines 2460 and 2461.*

### Step2: Obtaining layer importance scores
```
layer_important.py
```
## Layer Prune and Merge Pipeline
Prune the model layers based on their layer importance scores. The pruned layers are not directly discarded but are instead added to the remaining layers using Merged.
```
layer_prune/process_important.py #Add the paths for path_model and save_path in the file.
```

## Training model
Use the .py files in the configuration folder to conduct dense model training.

Use the .py files in the configuration_layer_prune folder to conduct prune model training.

## Evaluation
Use the .py files in the evaluation folder to conduct testing.

## License
This framework is licensed under the [Apache License 2.0](https://github.com/secretflow/secretflow/blob/main/LICENSE). For models and datasets, please refer to the original resource page and follow the corresponding License.

## Legal Disclaimer
The Legal Disclaimer of this framework is under the [LEGAL](https://github.com/secretflow/secretflow/blob/main/LEGAL.md)
