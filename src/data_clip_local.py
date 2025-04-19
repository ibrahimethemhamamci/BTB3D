import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import nibabel as nib
import tqdm
import random
import torch.nn.functional as F
from torch.utils.data import Dataset

from itertools import cycle, islice


class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, batch_size=10):
        self.data_folder = data_folder
        self.batch_size=batch_size

        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.df = pd.read_csv("/anvme/workspace/b180dc42-ct-rate/data_volumes/train_metadata.csv")  # select the metadata

        self.samples = self.prepare_samples()
        percent = 100
        #num_files = int((len(self.samples) * percent) / 100)
        #self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0
        random.seed(43)
        # We no longer use a resizing transform.
        # Instead, cropping to 512x512 will be done manually.
        self.nii_to_tensor = self.nii_img_to_tensor

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"], row['Impressions_EN']
        return accession_to_text



    def prepare_samples(self):
        # Initialize dictionaries for each node (keys 0 to 15)
        samples_2d = {}
        samples_3d = {}
        num_nodes = int(os.environ["SLURM_NNODES"])

        # Process each node's file (only first 100 lines)
        for node_number in range(num_nodes):
            formatted_node_number = f"{node_number:02d}"

            file_list = f"/anvme/workspace/b180dc42-ct-rate/clip_maisi/O2-MAGVIT2/slurm/node_{formatted_node_number}.txt.txt"
            samples_2d[node_number] = []
            samples_3d[node_number] = []

            with open(file_list, "r") as f:
                # Process only the first 100 lines from the file.

                for nii_file in tqdm.tqdm(f, desc=f"Processing {file_list}"):
                #for nii_file in tqdm.tqdm(islice(f, 40), desc=f"Processing {file_list}"):

                    nii_file = nii_file.strip()
                    accession_number = os.path.basename(nii_file)
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    if impression_text == "Not given.":
                        impression_text = ""
                    # Concatenate text pieces (if impression_text is iterable)
                    input_text_concat = ""
                    for text in impression_text:
                        input_text_concat += str(text)
                    input_text = f'{impression_text}'

                    # Retrieve metadata row for the accession.
                    row_meta = self.df[self.df['VolumeName'] == accession_number]
                    NumberofSlices = int(row_meta["NumberofSlices"].iloc[0])
                    z_spacing = float(row_meta["ZSpacing"].iloc[0])
                    tmpdir = os.environ["TMPDIR"]
                    nii_file_send = tmpdir + "/train/" + accession_number

                    if NumberofSlices >= 201:
                        meta_rows = int(row_meta["Rows"].iloc[0])
                        meta_cols = int(row_meta["Columns"].iloc[0])
                        for i in range(NumberofSlices - 200):
                            # For square data larger than 512x512, generate 4 crops.
                            if meta_rows == meta_cols and meta_rows > 512:
                                for crop_id in range(4):
                                    samples_3d[node_number].append((nii_file_send, input_text_concat, i, "3d", crop_id))
                                    samples_2d[node_number].append((nii_file_send, input_text_concat, i, "2d", crop_id))
                            else:
                                if meta_rows == meta_cols:
                                    # Otherwise, use a single (center) crop.
                                    samples_3d[node_number].append((nii_file_send, input_text_concat, i, "3d", None))
                                    samples_2d[node_number].append((nii_file_send, input_text_concat, i, "2d", None))

        # -----------------------------------------------
        # Post-processing across nodes (steps 1–4)
        # -----------------------------------------------

        batch_group = self.batch_size * 4  # desired group size per GPU

        for node in range(num_nodes):
            random.shuffle(samples_2d[node])
            random.shuffle(samples_3d[node])

        # (2) Equalize the number of samples across nodes.
        # Here we assume for each node, samples_2d[node] and samples_3d[node] have equal length.
        max_length = max(len(samples_2d[node]) for node in range(num_nodes))
        for node in range(num_nodes):
            current_length = len(samples_2d[node])
            if current_length < max_length:
                additional_needed = max_length - current_length
                samples_2d[node].extend(islice(cycle(samples_2d[node]), additional_needed))
                samples_3d[node].extend(islice(cycle(samples_3d[node]), additional_needed))

        # (3) Ensure each node's list length is divisible by batch_group.
        for node in range(num_nodes):
            remainder = len(samples_2d[node]) % batch_group
            if remainder != 0:
                additional = batch_group - remainder
                samples_2d[node].extend(islice(cycle(samples_2d[node]), additional))
                samples_3d[node].extend(islice(cycle(samples_3d[node]), additional))

        # (4) Divide each node's list into chunks of size batch_group,
        # then reassemble global final lists by round-robin interleaving across nodes.
        node_chunks_2d = {}
        node_chunks_3d = {}
        for node in range(num_nodes):
            node_chunks_2d[node] = [samples_2d[node][i:i+batch_group]
                                    for i in range(0, len(samples_2d[node]), batch_group)]
            node_chunks_3d[node] = [samples_3d[node][i:i+batch_group]
                                    for i in range(0, len(samples_3d[node]), batch_group)]

        # Assume every node now has the same number of chunks.
        num_chunks = len(node_chunks_2d[0])
        final_chunks_2d = []
        final_chunks_3d = []
        for chunk_idx in range(num_chunks):
            # Round-robin: first chunk from node 0, then node 1, ..., node 15, then next round.
            for node in range(num_nodes):
                final_chunks_2d.append(node_chunks_2d[node][chunk_idx])
                final_chunks_3d.append(node_chunks_3d[node][chunk_idx])

        # Flatten the chunks into a single list for 2d and 3d.
        final_samples_2d = [sample for chunk in final_chunks_2d for sample in chunk]
        final_samples_3d = [sample for chunk in final_chunks_3d for sample in chunk]

        # Zip the 2d and 3d samples together (interleaving them)
        samples = [item for pair in zip(final_samples_2d, final_samples_3d) for item in pair]
        print(len(samples), "total samples")
        print(len(final_samples_2d), "total 2d samples")
        return samples



    def nii_img_to_tensor(self, path, i, append, crop=None):
        # Load image with nibabel.
        #print(path)
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        file_name = os.path.basename(path)
        row_meta = self.df[self.df['VolumeName'] == file_name]
        slope = float(row_meta["RescaleSlope"].iloc[0])
        intercept = float(row_meta["RescaleIntercept"].iloc[0])

        # Apply slope/intercept without resizing/interpolation.
        img_data = slope * img_data + intercept
        img_data = img_data.astype(np.float32)  # ensure float32
        # Assume img_data is in shape (Rows, Columns, Slices).
        tensor = torch.tensor(img_data)  # shape: (H, W, D)

        # Slice selection for the slice dimension.
        if append == "3d":
            # Take 9 contiguous slices.
            tensor = tensor[:, :, i:i+201]
        else:
            # For 2d, randomly choose 4 slices.
            num_slices = tensor.shape[2]
            indices = np.random.choice(num_slices, size=4, replace=False)
            tensor = tensor[:, :, indices]

        # Clip Hounsfield units and normalize.
        hu_min, hu_max = -1000, 1000
        tensor = torch.clamp(tensor, min=hu_min, max=hu_max)
        tensor = tensor / 1000.0

        # Perform spatial cropping to 512x512.
        # Get current spatial dimensions.
        H, W, D = tensor.shape
        # Get original metadata dimensions.
        meta_rows = int(row_meta["Rows"].iloc[0])
        meta_cols = int(row_meta["Columns"].iloc[0])
        desired_size = 512

        if meta_rows == meta_cols and meta_rows > desired_size and crop is not None:
            # Use corner crops.
            if crop == 0:       # top-left
                start_h, start_w = 0, 0
            elif crop == 1:     # top-right
                start_h, start_w = 0, meta_cols - desired_size
            elif crop == 2:     # bottom-left
                start_h, start_w = meta_rows - desired_size, 0
            elif crop == 3:     # bottom-right
                start_h, start_w = meta_rows - desired_size, meta_cols - desired_size
            else:
                start_h = (H - desired_size) // 2
                start_w = (W - desired_size) // 2
        else:
            # Use a center crop.
            start_h = max((H - desired_size) // 2, 0)
            start_w = max((W - desired_size) // 2, 0)

        # Crop the spatial region.
        cropped = tensor[start_h:start_h+desired_size, start_w:start_w+desired_size, :]

        # If the crop is smaller than desired, pad accordingly.
        cH, cW, _ = cropped.shape
        pad_h = max(desired_size - cH, 0)
        pad_w = max(desired_size - cW, 0)
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            # Pad format for 3D tensor (pad last dim remains unchanged): (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            cropped = F.pad(cropped, (0, 0, pad_left, pad_right, pad_top, pad_bottom), value=-1)


        # Permute dimensions to (slices, H, W).
        cropped = cropped.permute(2, 0, 1)


        # Add a batch dimension and convert to bfloat16.
        cropped = cropped.unsqueeze(0).to(torch.bfloat16)
        return cropped

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nii_file, input_text, i, append, crop = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file, i, append, crop)
        # Clean input_text if necessary.
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        return video_tensor, str(index), append
