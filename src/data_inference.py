import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
import nibabel as nib
import random

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

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels="labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths = []
        self.samples = self.prepare_samples()
        print(len(self.samples))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim, resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform=self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"], row['Impressions_EN']
        return accession_to_text

    def prepare_samples(self):
        # First, collect samples as before
        samples = []
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        #for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
        #    for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
        for nii_file in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*.nii.gz'))):
            accession_number = nii_file.split("/")[-1]
            if accession_number not in self.accession_to_text:
                continue

            impression_text = self.accession_to_text[accession_number]
            if impression_text == "Not given.":
                impression_text = ""

            # (The following concatenation is kept as in the original code.)
            input_text_concat = ""
            for text in impression_text:
                input_text_concat = input_text_concat + str(text)
            input_text_concat = impression_text[0]
            input_text = f'{impression_text}'

            onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
            if len(onehotlabels) > 0:
                samples.append((nii_file, input_text, onehotlabels[0]))
                self.paths.append(nii_file)

        # --- NEW GROUPING LOGIC BASED ON X/Y DIMENSIONS ---
        # Load metadata (make sure the path is correct)
        meta_df = pd.read_csv("/anvme/workspace/b180dc42-ct-rate/data_volumes/valid_metadata.csv")
        groups = {}
        for sample in samples:
            nii_file, input_text, onehotlabels = sample
            file_name = nii_file.split("/")[-1]
            # Look up the metadata row by file name (VolumeName)
            meta_row = meta_df[meta_df["VolumeName"] == file_name]
            if meta_row.empty:
                continue  # Skip samples with no matching metadata
            # Extract x and y dimensions from the "Rows" and "Columns" columns
            rows = meta_row["Rows"].iloc[0]
            cols = meta_row["Columns"].iloc[0]
            key = (rows, cols)
            if key not in groups:
                groups[key] = []
            groups[key].append(sample)

        # For each group, pad with repeated samples until the count is divisible by 8
        for key, group in groups.items():
            while len(groups[key])% 64 != 0:
                n = len(group)
                remainder = n % 64
                if remainder != 0:
                    needed = 64 - remainder
                    # Pad by repeating samples from the group (cycling through the group)
                    for i in range(needed):
                        group.append(group[i % len(group)])
                groups[key] = group
            print(len(groups[key]))
        # Split each group into "words" (chunks) of 8 samples
        words = []
        for key, group in groups.items():
            # Create chunks of size 8 from the group
            for i in range(0, len(group), 64):
                word = group[i:i+64]
                words.append(word)

        # Shuffle the list of words.
        # (This interleaves chunks from groups so that in the final flattened list,
        # each contiguous block of 8 comes from the same (Rows,Columns) grouping.)
        random.shuffle(words)

        # Flatten the list of words back into a single list of samples
        new_samples = [sample for word in words for sample in word]

        return new_samples

    def npz_to_tensor(self, path):
        npz_img = np.load(path)["arr_0"][0]
        tensor = torch.tensor(npz_img)
        target_shape = (120, 120, 60)

        # Extract dimensions
        z, h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[:, h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(1)) // 2
        pad_h_after = dh - tensor.size(1) - pad_h_before

        pad_w_before = (dw - tensor.size(2)) // 2
        pad_w_after = dw - tensor.size(2) - pad_w_before

        pad_d_before = (dd - tensor.size(3)) // 2
        pad_d_after = dd - tensor.size(3) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(0, 3, 1, 2)

        #print(tensor.shape)
        # tensor = tensor[0:1]

        return tensor

    def nii_img_to_tensor(self, path, transform):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        df = pd.read_csv("/anvme/workspace/b180dc42-ct-rate/data_volumes/valid_metadata.csv")  # select the metadata
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

        # Prepare image data
        img_data = np.transpose(img_data, (1, 2, 0))
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data) / 1000)).astype(np.float32)
        slices = []

        tensor = torch.tensor(img_data)
        h, w, d = tensor.shape

        if d <49:
            new_d = 49
        else:
            new_d = d

        min_dim = min(h, w)
        target_dim = (min_dim // 16) * 16

        target_shape = (target_dim, target_dim, new_d)

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

        tensor = tensor.permute(2, 0, 1)
        #print(tensor.shape)

        tensor = tensor.unsqueeze(0).to(torch.bfloat16)

        return tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        nii_file, input_text, onehotlabels = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        name_acc = nii_file.split("/")[-2]
        return video_tensor, str(index)
