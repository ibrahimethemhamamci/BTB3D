import argparse
import torch
import json
import os
import torch.nn.functional as F

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
import numpy as np
import tqdm
import nibabel as nib
import pandas as pd
import multiprocessing
from functools import partial

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


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def process_subset(args, data_subset, gpu_index, output_list):
    """
    Process a subset of data on a specific GPU.

    Args:
        args: Parsed command-line arguments.
        data_subset (list): Subset of data to process.
        gpu_index (int): GPU index to use.
        output_list (multiprocessing.Manager().list): Shared list to store outputs.
    """
    # Assign the specific GPU
    device = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(device)
    disable_torch_init()

    # Load Model on the assigned GPU
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        args.load_8bit,
        args.load_4bit,
        device=device
    )

    model.to(device)

    # Prepare to collect outputs
    output_save = []

    for element in tqdm.tqdm(data_subset, desc=f"GPU {gpu_index} Processing"):
        # Determine conversation mode based on model name
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llama3"

        # Override with command-line argument if provided
        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] Auto-inferred conversation mode is {}, but `--conv-mode` is {}, using {}'.format(
                conv_mode, args.conv_mode, args.conv_mode))
            conv_mode = args.conv_mode
        print(conv_mode)
        conv = conv_templates[conv_mode].copy()
        roles = ('user', 'assistant') if "mpt" in model_name.lower() else conv.roles

        image_file = element["image"].replace(".nii.gz",".npz").replace(".npz", ".nii_embedded.npz")
        #image_path = "/shares/menze.dqbm.uzh/ihamam/CT_CLIP_final_experiments_after_submission_low_patch_size/external_val_encodings/image/"+image_file
        image_path = args.embedding_path +image_file
        print(image_path)
        image = np.load(image_path)["arr"].transpose(0,2,3,4,1)
        image_size = image.size
        # Similar operation in model_worker.py
        #image_tensor = process_images([image], image_processor, model.config)
        image_tensor = torch.tensor(image)

        if isinstance(image_tensor, list):
            image_tensor = [img.to(device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)  # Add batch dimension if needed
        conversations_save = []

        for conversation in element["conversations"]:
            i = 0
            id = element["id"]
            if conversation["from"] == "human":
                inp = conversation["value"]
                #inp = "<image>\n<report_generation>"
                #inp = inp.replace("<report_generation>", "")
                print(inp)
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                print(image_file)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                #stop_str = stop_str + conv.roles[0]
                #stop_str = conv.sep + conv.roles[1]
                print(stop_str)

                keywords = [stop_str]
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        image_sizes=[image_size],
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=True)

                outputs = tokenizer.decode(output_ids[0]).strip()
                conv.messages[-1][-1] = outputs

                conversations_save.append({"id":id, "question": inp, "answer":outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        output_save.append({"image": image_file, "conversations_out": conversations_save})

    # Append the results to the shared list
    output_list += output_save

    #except Exception as e:
    #    print(f"An error occurred on GPU {gpu_index}: {e}")

def main(args):
    # Load and Split Data
    save_it = args.model_path.split("/")[-2].split("-")[-1]
    print(save_it)

    with open(args.json_path, 'r') as file:
        data_val = json.load(file)

    total_nodes = args.total_nodes
    node_index = args.node_index

    # Calculate the subset of data for this node
    per_node = len(data_val) // total_nodes
    start_idx = node_index * per_node
    # Ensure the last node picks up any remaining data
    end_idx = start_idx + per_node if node_index != total_nodes - 1 else len(data_val)
    #start_idx = 0
    #end_idx = 12
    data_subset = data_val[start_idx:end_idx]

    # Number of GPUs per node
    num_gpus = args.num_gpus  # Adjust if needed

    # Split data_subset into num_gpus parts
    split_size = len(data_subset) // num_gpus
    data_splits = [data_subset[i*split_size : (i+1)*split_size] for i in range(num_gpus)]
    # Handle any remaining data
    for i in range(len(data_subset) % num_gpus):
        data_splits[i].append(data_subset[-(i+1)])

    # Initialize multiprocessing Manager
    manager = multiprocessing.Manager()
    output_list = manager.list()

    # Create a list to hold processes
    processes = []

    for gpu_idx in range(num_gpus):
        p = multiprocessing.Process(target=process_subset, args=(args, data_splits[gpu_idx], gpu_idx, output_list))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Convert manager.list() to a regular list
    final_output = list(output_list)

    # Save the results to a JSON file specific to this node
    output_filename = f"16_preprocessed_encoded_attnpool_1node_{save_it}it_node{node_index}_vqa.json"
    with open(output_filename, "w") as json_file:
        json.dump(final_output, json_file, indent=4)

    print(f"Node {node_index} has completed processing. Results saved to {output_filename}.")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' to avoid CUDA re-initialization issues
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Distributed LLaVA Processing Script")

    # Model and Processing Arguments
    parser.add_argument("--model-path", type=str, default="path_to_checkpoint")
    parser.add_argument("--model-base", type=str, default="path_to_checkpoint")
    parser.add_argument("--embedding_path", type=str, default="path_to_encoded_volumes")
    parser.add_argument("--json_path", type=str, default="path_to_vqa_reports.json")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda').")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode override.")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit precision.")
    parser.add_argument("--load-4bit", action="store_true", help="Load model in 4-bit precision.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")

    # Distributed Arguments
    parser.add_argument("--node_index", type=int, default=0, help="Index of the current node (0).")
    parser.add_argument("--total_nodes", type=int, default=1, help="Total number of nodes (default: 1).")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of gpus (default: 1).")

    args = parser.parse_args()
    main(args)
