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
import nibabel as nib
import tqdm
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

class CTReportDataset(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.df = pd.read_csv("/anvme/workspace/b180dc42-ct-rate/data_volumes/train_metadata.csv") #select the metadata

        self.samples = self.prepare_samples()
        percent = 100
        num_files = int((len(self.samples) * percent) / 100)
        #num_files = 2286
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0



        #self.resize_dim = resize_dim
        #self.resize_transform = transforms.Resize((resize_dim, resize_dim))
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    def prepare_samples(self):
        samples_2d = []
        samples_3d = []
        samples=[]
        count = 0
        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    accession_number = nii_file.split("/")[-1]
                    #accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]

                    if impression_text == "Not given.":
                        impression_text=""

                    input_text_concat = ""
                    for text in impression_text:
                        input_text_concat = input_text_concat + str(text)
                    input_text_concat = impression_text[0]
                    input_text = f'{impression_text}'


                    row = self.df[self.df['VolumeName'] == accession_number]
                    NumberofSlices = float(row["NumberofSlices"].iloc[0])
                    z_spacing = float(row["ZSpacing"].iloc[0])
                    NumberofSlices_spacing = int((NumberofSlices * z_spacing) / 1.5)

                    samples.append((nii_file,input_text))
                    """
                    if NumberofSlices_spacing >=9:

                        for i in range(NumberofSlices_spacing-8):
                            #samples.append((nii_file, input_text_concat, i, mode))
                            for k in range(2):
                                if k == 0:
                                    append = "3d"
                                    samples_3d.append((nii_file, input_text_concat, i, append))

                                else:
                                    append = "2d"
                                    samples_2d.append((nii_file, input_text_concat, i, append))

                    """
                    """
                    samples.append((nii_file, input_text_concat, 0))
                    self.paths.append((nii_file, 0))
                    """

        #random.shuffle(samples_2d)  # Shuffle 2D samples
        #random.shuffle(samples_3d)  # Shuffle 3D samples
        random.shuffle(samples)
        length = len(samples)
        #if length % 20 != 0:
        #    samples_2d = samples_2d[:length - (length % 20)]

        #length = len(samples_3d)
        #if length % 20 != 0:
        #    samples_3d = samples_3d[:length - (length % 20)]
        # Merge back into the dataset while maintaining 2D-3D sequence
        #samples = [item for pair in zip(samples_2d, samples_3d) for item in pair]
        #print(len(samples), "len samples")
        #print(len(samples_2d), "len samples2d")

        return samples

    def __len__(self):
        return len(self.samples)


    def npz_to_tensor(self,path):
        npz_img = np.load(path)["arr_0"][0]
        tensor = torch.tensor(npz_img)
        target_shape = (512,512,33)

        # Extract dimensions
        z, h, w, d= tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[:,h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(1)) // 2
        pad_h_after = dh - tensor.size(1) - pad_h_before

        pad_w_before = (dw - tensor.size(2)) // 2
        pad_w_after = dw - tensor.size(2) - pad_w_before

        pad_d_before = (dd - tensor.size(3)) // 2
        pad_d_after = dd - tensor.size(3) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(0,3, 1, 2)
        tensor = tensor[0:1]
        #print(tensor.shape)


        return tensor
    def nii_img_to_tensor(self, path):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        row = self.df[self.df['VolumeName'] == file_name]
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

        # Extract dimensions
        #if append == "3d":
        #tensor = tensor[:,:,i:i+9]
        #else:
        #    indices = np.random.choice(tensor.shape[2], size=4, replace=False)

         #   tensor = tensor[:,:,indices]


        h, w, d = tensor.shape

        """
        if d < 17:
            new_d = 17
        """
        target_shape = (512,512,241)

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
        tensor = tensor.unsqueeze(0).to(torch.bfloat16)

    
        return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        #print(nii_file, i)
        video_tensor = self.nii_img_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, str(index), "3d"