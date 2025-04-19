#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import os
import sys
import time
import random
import pathlib
import numpy as np
import pandas as pd
from collections import OrderedDict
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.transforms as T
from torchvision.io import read_video, write_video, read_image, write_png, ImageReadMode

import nibabel as nib
from einops import rearrange
import tqdm

from omegaconf import OmegaConf
from modeling.magvit_model import VisionTokenizer
from src.utils import get_config, preprocess_vision_input, calculate_resize_shape
from PIL import Image
import tqdm
from modeling.lfq_quantizer import LFQ

import torch.nn as nn


# -------------------------------
# Utility functions
# -------------------------------

def find_max(k):
    if k <= 1:
        return None  # No such number exists if k <= 1
    n = (k - 1) // 8  # Solve for n such that 1 + 8n < k
    return 1 + 8 * n

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.
    """
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array using trilinear interpolation
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array
def read_nii_preprocess(path):
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    df = pd.read_csv("path_to_valid_metadata.csv") #select the metadata
    file_name = path.split("/")[-1]
    row = df[df['VolumeName'] == file_name]
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"].iloc[0])

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data= np.transpose(img_data, (1, 2, 0))

    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = img_data
    img_data = (((img_data ) / 1000)).astype(np.float32)
    slices=[]

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    a, b,c = tensor.shape
    if c > 241:
        c=241
    c = 241
    target_shape = (512,512,c)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    #print(tensor.shape)
    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 0, 1)
    #print(tensor.shape)
    tensor = tensor.unsqueeze(0)


    return tensor



def read_nii(path):
    """
    Reads a NIfTI file, applies rescaling and spacing adjustments,
    and returns a tensor formatted for the model.
    """
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    # Load metadata from CSV (assumed to be available at the given location)
    df = pd.read_csv("path_to_valid_metadata.csv")
    file_name = os.path.basename(path)
    row = df[df['VolumeName'] == file_name]
    slope = float(row["RescaleSlope"].iloc[0])
    intercept = float(row["RescaleIntercept"].iloc[0])
    xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
    z_spacing = float(row["ZSpacing"].iloc[0])

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    # Apply intensity scaling and spacing resampling
    img_data = slope * img_data + intercept
    img_data = img_data.transpose(2, 0, 1)
    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    #img_data = resize_array(tensor, current, target)
    img_data = tensor[0][0].cpu().numpy()
    img_data = np.transpose(img_data, (1, 2, 0))
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)
    img_data = (((img_data) / 1000)).astype(np.float32)

    # Prepare for cropping/padding
    tensor = torch.tensor(img_data)
    a, b, c = tensor.shape
    if a != b:
        if b>a:
            b = a
        else:
            a = b
    if a % 16 != 0:
        a = (a // 16) * 16  # Round down to nearest multiple of 16
        b = a  # Set b equal to the adjusted value of a
    #if c > 201:
    #    c = 201
    c = c
    target_shape = (a, b, c)
    h, w, d = tensor.shape
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before
    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before
    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before
    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
    tensor = tensor.permute(2, 0, 1)  # Now shape is (slices, H, W)
    tensor = tensor.unsqueeze(0)      # Add batch dimension
    return tensor

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--modal", type=str, required=True,
                        help="Modality: 'image', 'video', or 'nii'")
    parser.add_argument("--ckpt-path", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--target-pixels", type=int, default=384,
                        help="Target pixels for the short edge")
    parser.add_argument("--max-num-frames", required=False, default=None, type=int)
    parser.add_argument("--extract-frame-interval", required=False, default=1, type=int)
    parser.add_argument("-i", type=str, required=True,
                        help="Input file or folder path. For NIfTI mode, provide the main folder containing the nested .nii/.nii.gz files.")
    parser.add_argument("-o", type=str, required=True,
                        help="Output directory")
    return parser.parse_args()

def setup_distributed():
    """
    Initializes the distributed process group if environment variables indicate a DDP run.
    Returns the local rank, global rank, and world size.
    """
    if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"DDP initialized: rank {rank}/{world_size}, using GPU {local_rank}")
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        print("Running in single-process mode.")
    return local_rank, rank, world_size
def center_crop_axis2(tensor):
    # Get the size along axis 2.
    D = tensor.shape[2]
    # Compute n such that new_size = 1 + 4*n <= D
    n = (D - 1) // 4
    new_size = 1 + 4 * n
    # Calculate the starting index for the central crop
    start = (D - new_size) // 2

    # Create a slice for axis 2
    # For a tensor with shape (a, b, c, ...), we slice as follows:
    slices = [slice(None)] * tensor.ndim  # create a slice(None) for each dimension
    slices[2] = slice(start, start + new_size)

    # Return the cropped tensor
    return tensor[tuple(slices)]

# -------------------------------
# Main function
# -------------------------------
def main():
    args = parse_args()
    local_rank, rank, world_size = setup_distributed()

    # Ensure the output directory exists
    output_dir = args.o
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir.replace("quantized", "encoded"), exist_ok=True)

    # Load the model checkpoint and configuration
    states = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model_config = get_config(args.config_path)
    model = VisionTokenizer(config=model_config, commitment_cost=0, diversity_gamma=0,
                            use_gan=False, use_lecam_ema=False, use_perceptual=False)
    model.tokenizer.load_state_dict(states, strict=True)
    tokenizer = model.tokenizer

    model.eval()

    model.to("cuda").to(torch.bfloat16)  # each process puts the model on its assigned GPU
    model.tokenizer.to(torch.bfloat16)
    # Calculate the spatial downsample ratio based on the decoder configuration.
    spatial_downsample_ratio = 2 ** (len(model_config.model.decoder.channel_multipliers))

    if args.modal == "nii":
        input_path = args.i
        input_path = pathlib.Path(input_path)
        if input_path.is_dir():
            # Use rglob to recursively find all .nii and .nii.gz files
            nii_files = sorted(list(input_path.rglob("*.nii*")))
        else:
            nii_files = [input_path]
        if len(nii_files) == 0:
            print("No NIfTI files found in the input directory.")
            return

        # Partition the file list across processes
        total_files = len(nii_files)
        nii_files = [f for i, f in enumerate(nii_files) if i % world_size == rank]
        print(f"Rank {rank} processing {len(nii_files)} files out of total {total_files}.")

        # Process each NIfTI file independently.
        for nii_file in tqdm.tqdm(nii_files):

            output_file_enc = os.path.join(output_dir.replace("quantized", "encoded"), f"{nii_file.stem}_embedded.npz")
            output_file = os.path.join(output_dir, f"{nii_file.stem}_embedded.npz")
            if os.path.exists(output_file_enc) and os.path.exists(output_file):
                print(f"Skipping {nii_file.stem} as outputs already exist.")
                continue


            print(f"Rank {rank} processing file: {nii_file}")
            frames_torch = read_nii_preprocess(str(nii_file))
            slice_num = frames_torch.shape[1]

            #corrected_num = find_max(slice_num)
            #corrected_num = 41
            #frames_torch = frames_torch[:,:corrected_num]
            # For the NIfTI case, the original script permutes dimensions.
            #frames = frames_torch.permute(1, 0, 2, 3)
            frames = frames_torch.unsqueeze(0)
            print("Input tensor shapes:", frames_torch.shape, frames.shape)

            # Process the volume patch-by-patch along the first dimension.
            patch_size = 101
            outs = []
            #for i in range(frames.shape[0]):


            with torch.no_grad():
                #if True:
                #for i in range(1):
                #patch = frames[i:i+1]
                patch = frames.to(torch.bfloat16)
                #patch = patch[:,:,:200]
                patch = center_crop_axis2(patch)
                print("Processing patch shape:", patch.shape)
                h, w = frames.shape[-2:]

                # Adjust height and width by the spatial downsample ratio
                h = h // spatial_downsample_ratio
                w = w // spatial_downsample_ratio

                patch = patch.to("cuda")
                s = time.time()
                # Encode the patch
                #with torch.cuda.amp.autocast():
                _, encoded_output, _, encoded_really= tokenizer.encode(patch, entropy_loss_weight=0.0)
                e = time.time()
                print(f"=== encode cost: {e - s:.3f} s ===")
                token_ids = encoded_output.indices
                print(f"num tokens: {token_ids.size()}, uniques: {token_ids.unique().numel()}")
                s = time.time()
                output_file_enc = os.path.join(output_dir.replace("quantized","encoded"), f"{nii_file.stem}_embedded.npz")
                #print(encoded_output)
                np.savez(output_file_enc, arr = encoded_really.float().cpu().detach().numpy())

                # Quantize the tokens and rearrange into spatial layout
                quantized = tokenizer.quantize.indices_to_codes(indices=token_ids, project_out=True)
                print("quantized shape1:", quantized.shape)
                quantized = rearrange(quantized, "b (t h w) c -> b c t h w", h=h, w=w)
                print("quantized shape2:", quantized.shape)

                """
                quantized = F.interpolate(
                    quantized,
                    scale_factor=(1.0, 0.5, 0.5),   # (T-scale, H-scale, W-scale)
                    mode='trilinear',              # or 'nearest', 'area', etc.
                    align_corners=False
                )
                """
                B, C, T, H, W = quantized.shape
                #pool2d = nn.AvgPool2d(kernel_size=2, stride=2)

                #quantized = quantized.contiguous().view(B * C * T, 1, H, W)
                #quantized = pool2d(quantized)
                #quantized = quantized.view(B, C, T, quantized.shape[-2], quantized.shape[-1])


                output_file = os.path.join(output_dir, f"{nii_file.stem}_embedded.npz")
                np.savez(output_file, arr = quantized.float().cpu().detach().numpy())

                # Decode the quantized tokens to reconstruct the patch.
                """
                decoded_output_patch = tokenizer.decode(quantized)
                print("Decoded patch shape:", decoded_output_patch.shape)
                e = time.time()
                #print(f"=== decode cost: {e - s:.3f} s ===")
                decoded_patch = decoded_output_patch.detach()
                outs.append(decoded_patch)
                """
            """
            # Concatenate all decoded patches along the patch (time/slice) dimension.
            #decoded_output = torch.cat(outs, dim=1)
            decoded_output = decoded_patch
            print("Concatenated decoded output shape:", decoded_output.shape)
            #decoded_output = decoded_output.unsqueeze(0)
            decoded_output = decoded_output.squeeze(0)
            #decoded_output = decoded_output.permute(0, 3, 1, 2)
            print("Resized decoded output shape:", decoded_output.shape)

            frames = frames.squeeze(0)
            #decoded_output = decoded_output.flip(2)
            #frames = frames.permute(0, 1, 3, 2)

            # Save the result. Here we write a video file; the output filename is based on the input file stem.
            output_file = os.path.join(output_dir, f"{nii_file.stem}_recon.mp4")
            video_tensor = (rearrange(decoded_output.permute(0,1,3,2).repeat(3, 1, 1, 1), "c t h w -> t h w c")
                            .clamp(-1, 1) + 1) * 127.5
            video_tensor = video_tensor.to(torch.uint8)
            write_video(output_file, video_tensor, fps=10)
            output_file = os.path.join(output_dir, f"{nii_file.stem}_gt.mp4")
            video_tensor = (rearrange(frames.permute(0,1,3,2).repeat(3, 1, 1, 1), "c t h w -> t h w c")
                            .clamp(-1, 1) + 1) * 127.5
            video_tensor = video_tensor.to(torch.uint8)
            write_video(output_file, video_tensor, fps=10)

            print(f"Rank {rank} finished writing: {output_file}")
            """


    elif args.modal == "image":
        # For single image processing (kept from your original code)
        input_path = args.i
        img = Image.open(input_path).convert("RGB")
        resize_shape = calculate_resize_shape(source_image_size=img.size,
                                              target_pixels=args.target_pixels,
                                              min_square_size=spatial_downsample_ratio)
        print(f"original image size: {img.size[::-1]} | resize shape: {resize_shape}")
        frames_torch = T.functional.pil_to_tensor(img).unsqueeze(dim=0)
        frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
        frames = frames.unsqueeze(dim=2)

        s = time.time()
        _, encoded_output, *_ = tokenizer.encode(frames.to("cuda"), entropy_loss_weight=0.0)
        e = time.time()
        print(f"=== encode cost: {e - s:.3f} s ===")
        token_ids = encoded_output.indices
        print(f"num tokens: {token_ids.size()}, uniques: {token_ids.unique().numel()}")
        s = time.time()
        quantized = tokenizer.quantize.indices_to_codes(indices=token_ids, project_out=True)
        quantized = rearrange(quantized, "b (t h w) c -> b c t h w",
                              h=frames.shape[-2], w=frames.shape[-1])
        decoded_output_patch = tokenizer.decode(quantized)
        e = time.time()
        print(f"=== decode cost: {e - s:.3f} s ===")
        # Save input and reconstructed images
        write_png(((frames.squeeze(dim=0).squeeze(dim=1).detach().cpu()+1)*127.5).to(torch.uint8),
                  os.path.join(output_dir, "input.png"), compression_level=0)
        write_png(((decoded_output_patch.squeeze(dim=1).detach().cpu().clamp(-1,1)+1)*127.5).to(torch.uint8),
                  os.path.join(output_dir, "recon_output_image.png"), compression_level=0)

    elif args.modal == "video":
        # For video processing (kept from your original code)
        frames_torch, _ = read_video(args.i, output_format="TCHW", pts_unit="sec")
        original_shape = frames_torch.shape
        orig_img_h, orig_img_w = frames_torch.shape[-2:]
        frames_torch = frames_torch[::args.extract_frame_interval][:args.max_num_frames]
        resize_shape = calculate_resize_shape(source_image_size=(orig_img_w, orig_img_h),
                                              target_pixels=args.target_pixels,
                                              min_square_size=spatial_downsample_ratio)
        frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
        print(f"original video shape: {original_shape} | resize shape: {resize_shape}")
        frames = frames.unsqueeze(dim=0).permute(0, 2, 1, 3, 4)  # n c t h w

        s = time.time()
        _, encoded_output, *_ = tokenizer.encode(frames.to("cuda"), entropy_loss_weight=0.0)
        e = time.time()
        print(f"=== encode cost: {e - s:.3f} s ===")
        token_ids = encoded_output.indices
        print(f"num tokens: {token_ids.size()}, uniques: {token_ids.unique().numel()}")
        s = time.time()
        quantized = tokenizer.quantize.indices_to_codes(indices=token_ids, project_out=True)
        quantized = rearrange(quantized, "b (t h w) c -> b c t h w", h=frames.shape[-2], w=frames.shape[-1])
        decoded_output_patch = tokenizer.decode(quantized)
        e = time.time()
        print(f"=== decode cost: {e - s:.3f} s ===")
        decoded_output = decoded_output_patch.detach()
        write_video(os.path.join(output_dir, "recon_2d_coronal.mp4"),
                    ((rearrange(decoded_output.repeat(3, 1, 1, 1), "c t h w -> t h w c")
                      .clamp(-1,1)+1)*127.5).to(torch.uint8), fps=10)
    else:
        print("Modal type not recognized. Please choose 'nii', 'image', or 'video'.")

    # Finalize the distributed process group (if initialized)
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
