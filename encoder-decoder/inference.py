# -*- coding:UTF-8 -*-
import torch
import sys
import os
from omegaconf import OmegaConf
from modeling.magvit_model import VisionTokenizer
from torchvision.io import read_video, write_video, read_image, write_png
from torchvision.io import ImageReadMode
import torchvision.transforms as T
from einops import rearrange
import time
import pathlib
import random
from collections import OrderedDict
from src.utils import get_config, preprocess_vision_input, calculate_resize_shape
from PIL import Image
from argparse import ArgumentParser
import nibabel as nib
import numpy as np
import pandas as pd
import torch.nn.functional as F
import tqdm

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array
def read_nii(path):
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    df = pd.read_csv("/path/to/your/valid_metadata.csv") #select the metadata
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--modal", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--target-pixels", type=int, default=384, help="target pixels for the short edge")
    parser.add_argument("--max-num-frames", required=False, default=None, type=int)
    parser.add_argument("--extract-frame-interval", required=False, default=1, type=int)

    parser.add_argument("-i", type=str, required=True, help="input image/video path")
    parser.add_argument("-o", type=str, required=True, help="output dir")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input_path = args.i
    output_dir = args.o
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    states = torch.load(args.ckpt_path, map_location="cpu", weights_only=True)
    model_config = get_config(args.config_path)
    model = VisionTokenizer(config=model_config, commitment_cost=0, diversity_gamma=0, use_gan=False, use_lecam_ema=False, use_perceptual=False)
    model.tokenizer.load_state_dict(states, strict=True)
    tokenizer = model.tokenizer
    modal = args.modal
    model.eval()
    model.to("cuda")
    spatial_downsample_ratio = 2 ** (len(model_config.model.decoder.channel_multipliers) - 1)
    #temporal_downsample_ratio = 2 ** (sum(model_config.model.decoder.temporal_downsample))

    with torch.no_grad():
        if modal == "image":
            img = Image.open(input_path).convert("RGB")
            resize_shape = calculate_resize_shape(source_image_size=img.size, target_pixels=args.target_pixels, min_square_size=spatial_downsample_ratio)
            print(f"original image size: {img.size[::-1]} | resize shape: {resize_shape}")
            frames_torch = T.functional.pil_to_tensor(img).unsqueeze(dim=0)
            frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
            frames = frames.unsqueeze(dim=2)
        elif modal == "video":
            frames_torch = read_video(input_path, output_format="TCHW", pts_unit="sec")[0]
            original_shape = frames_torch.shape
            orig_img_h, orig_img_w = frames_torch.shape[-2:]
            frames_torch = frames_torch[::args.extract_frame_interval][:args.max_num_frames]
            resize_shape = calculate_resize_shape(source_image_size=(orig_img_w, orig_img_h), target_pixels=args.target_pixels, min_square_size=spatial_downsample_ratio)
            frames = preprocess_vision_input(frames_torch, resize_shape=resize_shape)
            print(f"original video shape: {original_shape} | resize shape: {resize_shape}")
            frames = frames.unsqueeze(dim=0).permute(0, 2, 1, 3, 4) # n c t h w
        elif modal == "nii":
            frames_torch = read_nii(input_path)
            frames = frames_torch.permute(1,0,2,3)
            print(frames_torch.shape, "shape here")
            #frames = frames_torch[:,:]
            #frames=frames.unsqueeze(1)
            print(frames.shape)

        patch_size = 101  # Each patch will span 17 elements along dim=1
        #num_patches = frames.size(2) // patch_size  # Number of patches (170 // 17 = 10)
        #patch_size = 17
        #overlap_size = 8
        #num_patches = (frames.size(2) - overlap_size) // (patch_size - overlap_size)

        #num_patches = frames.shape[0]
        outs = []
        #for i in tqdm.tqdm(range(num_patches)):
        #for i in range(frames.shape[0]):
        with torch.no_grad():
            #start = i * patch_size
            #end = start + patch_size
            #patch = frames[:,:,start:end]
            patch=frames[i:i+1]
            print(patch.shape)
            #start = i * (patch_size - overlap_size)
            #end = start + patch_size


            #patch = frames[ i:i+1]

            print(patch.size())
            h, w = frames.shape[-2:]
            h = h // spatial_downsample_ratio
            w = w // spatial_downsample_ratio
            print(h, w)
            tokenizer = tokenizer
            #frames = frames

            patch = patch.to("cuda")
            s = time.time()
            _, encoded_output, *_ = tokenizer.encode(patch, entropy_loss_weight=0.0)
            e = time.time()
            print(f"=== encode cost: {e -s} s ===")
            token_ids = encoded_output.indices

            print(f"num tokens: {token_ids.size()}, uniques: {token_ids.unique().numel()}")
            s = time.time()
            quantized = tokenizer.quantize.indices_to_codes(indices=token_ids, project_out=True)
            print("quantized shape1:", quantized.shape)
            #quantized = rearrange(quantized, "b (t h w) c -> b c t h w", h=h, w=w)
            quantized = rearrange(quantized, "b (h w) c -> b c h w", h=h, w=w)
            print("quantized_shape2:", quantized.shape)


            decoded_output_patch = tokenizer.decode(quantized)
            print(decoded_output_patch.shape)
            e = time.time()
            print(f"=== decode cost: {e -s} s ===")
            decoded_patch = decoded_output_patch.detach()
            outs.append(decoded_patch)

        decoded_output = torch.cat(outs,dim=1)
        print(decoded_output.shape)
        decoded_output = decoded_output[:,:].unsqueeze(0)
        decoded_output = F.interpolate(decoded_output, size=(474, 512, 512), mode='trilinear', align_corners=False)
        decoded_output = decoded_output.squeeze(0)
        decoded_output = decoded_output.permute(0,3,1,2)
        print(decoded_output.shape)
        frames = frames.unsqueeze(0)
        #frames = F.interpolate(frames, size=(474, 512, 512), mode='trilinear', align_corners=False)
        frames = frames.squeeze(0)
        #frames = frames.flip(1)
        decoded_output = decoded_output.flip(2)
        frames = frames.permute(0,1,3,2)
        if modal == "image":
            print(frames.squeeze(dim=0).squeeze(dim=1).shape, decoded_output.squeeze(dim=0).squeeze(dim=1).shape)
            write_png(((frames.squeeze(dim=0).squeeze(dim=1).detach().cpu()+1) * 127.5).to(torch.uint8), os.path.join(output_dir, "input.png"), compression_level=0)
            write_png(((decoded_output.squeeze(dim=1).detach().cpu().clamp(-1,1)+1) * 127.5).to(torch.uint8), os.path.join(output_dir, f"recon_output_image.png"), compression_level=0)
        elif modal == "video" or modal == "nii":
            #write_video(os.path.join(output_dir, f"input_axial.mp4"), ((rearrange(frames.repeat(3,1,1,1), "c t h w -> t h w c")+1) * 127.5).to(torch.uint8), fps=10)
            write_video(os.path.join(output_dir, f"recon_2d_coronal.mp4"), ((rearrange(decoded_output.repeat(3,1,1,1), "c t h w -> t h w c").clamp(-1,1)+1) * 127.5).to(torch.uint8), fps=10)
if __name__ == "__main__":
    main()