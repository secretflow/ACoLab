from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import pandas as pd
import torch
import argparse
import os
from tqdm import tqdm
from model.ella import load_ella,load_ella_for_pipe,generate_image_with_flexible_max_length, T5TextEmbedder


def main(args):
    # Load the Stable Diffusion pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    ella = load_ella(args.ella_path, pipe.device, pipe.dtype)
    # Load the T5 model(default is flan-t5-xl or after DPO)
    t5_encoder = T5TextEmbedder(pretrained_path=args.t5_path).to(pipe.device, dtype=torch.float16)
    
    load_ella_for_pipe(pipe, ella)

    
    # Make folders
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if args.input_file.endswith('.csv'):
        data = []
        data = pd.read_csv(args.input_file, lineterminator='\n')

        for i in tqdm(range(len(data))):
            _batch_size = 1
            prompt = [data['prompt'][i]] * _batch_size
            image_flexible = generate_image_with_flexible_max_length(
                pipe,
                t5_encoder,
                prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                height=args.image_size,
                width=args.image_size,
                generator=[
                    torch.Generator(device="cuda").manual_seed(args.seed + i)
                    for i in range(_batch_size)
                ],
            )
            image_flexible.save(os.path.join(args.output_folder, data["image"][i]))
    else: 
        print('Invalid input file format')



if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'generateImages',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--ella_path', help='path of ella', type=str, required=True)
    parser.add_argument('--model_path', help='path of model', type=str, required=True)
    parser.add_argument('--t5_path', help='path of LLM', type=str, required=False, default="google/flan-t5-xl")
    parser.add_argument('--input_file', help='path to csv file with prompts', type=str, required=True)
    parser.add_argument('--output_folder', help='folder where to save images', type=str, required=True)
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=12)
    parser.add_argument('--num_inference_steps', help='number of diffusion steps during inference', type=float, required=False, default=50)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--seed', help='seed for gneration', type=int, required=False, default=1001)
    args = parser.parse_args()
    main(args)