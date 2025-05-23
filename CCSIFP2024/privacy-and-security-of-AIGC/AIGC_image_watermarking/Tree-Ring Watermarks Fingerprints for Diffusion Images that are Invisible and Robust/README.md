# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust
This repository contains the official implementation of Tree-Ring Watermarks, a watermarking method for diffusion model outputs.

## About
Tree-Ring Watermarking is a novel technique for embedding invisible and robust watermarks in images generated by diffusion models. It modifies the initial noise array used in the generation process to include a specific pattern, or "key," in its Fourier transform. The standard diffusion pipeline remains unchanged, ensuring compatibility. To detect the watermark, the generation process is inverted to retrieve the original noise array, which is then analyzed for the presence of the embedded key.

## Dependencies
The following dependencies are required:

- `PyTorch == 1.13.0`
- `transformers == 4.23.1`
- `diffusers == 0.11.1`
- `datasets`

**Note**: Higher versions of the `diffusers` library may not be compatible with the DDIM inversion code.

## Usage

### Main Experiments and CLIP Score Calculation
**Non-Adversarial Case:**
```
python genereate_watermark.py --run_name no_attack --watermark_channel 3 --watermark_pattern ring --start_index 0 --end_index 1000 --enable_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
```

**Adversarial Case (e.g., Rotation of 75 Degrees):**
```
python genereate_watermark.py --run_name rotation --watermark_channel 3 --watermark_pattern ring --r_degree 75 --start_index 0 --end_index 1000 --enable_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
```

## Parameters
**Crucial hyperparameters for Tree-Ring:**

- `watermark_channel`: the index of the watermarked channel. If set as -1, watermark all channels.
- `watermark_pattern`: watermark type: zeros, rand, ring.
- `watermark_radius`: watermark radius.


## Enhancement: Single-Layer Ring Watermark with Multi-Bit Embedding
This version introduces a single-layer ring watermarking scheme that allows multi-bit embedding.

The new implementation includes additional parameters:
- `msg`: the index of the watermarked channel. If set as -1, watermark all channels.
- `sync_marker`: A synchronization mechanism to improve robustness and alignment during extraction.
- `msg_scaler`: A scaling factor that adjusts the strength of the embedded watermark information.
- `msg_redundant`: The redundancy level to enhance robustness against distortions.

### Usage
**Multi-Bit Embedding with Single-Layer Ring Watermark**
```
python genereate_watermark_msg.py --run_name multi_bit_embedding --watermark_channel 3 --watermark_pattern message --watermark_radius 10 --msg "10101011110010111110101111001011" --sync_marker "10101011" --msg_scaler 100 --msg_redundant 2 --start_index 0 --end_index 1000 --enable_tracking
```
