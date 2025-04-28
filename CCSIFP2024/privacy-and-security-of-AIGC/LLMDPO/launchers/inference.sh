python inference.py \
  --ella_path "/private/task/zhuangshuhan/checkpoints/ELLA" \
  --model_path "/private/task/zhuangshuhan/checkpoints/stable-diffusion-v1-5" \
  --t5_path "/private/task/zhuangshuhan/checkpoints/ELLA/models--google--flan-t5-xl--text_encoder" \
  --input_file "/private/task/zhuangshuhan/AIGC_Sec/LLMDPO/data/CoProv2_test.csv" \
  --output_folder "results/output_ella" \
  --guidance_scale 12 \
  --num_inference_steps 50 \
  --image_size 512 \
  --seed 1001
