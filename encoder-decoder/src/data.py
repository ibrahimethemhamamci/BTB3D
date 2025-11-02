from pathlib import Path
from PIL import Image
import torch.distributed
from torch.utils.data import Dataset
import torchvision.transforms as T
from functools import partial
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from beartype.door import is_bearable
from typing import Tuple, List
from torch.utils.data import DataLoader as PytorchDataLoader
from torchvision.io import read_video

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")


def identity(t, *args, **kwargs):
    return t

def get_resized_shape(w, h, target_size):
    min_edge = min(w, h)
    scale = target_size / min_edge
    if min_edge == w:
        scaled_w = target_size
        scaled_h = int(h * scale)
    elif min_edge == h:
        scaled_h = target_size
        scaled_w = int(w * scale)
    return scaled_w, scaled_h

def resize(img, image_size, interpolation=T.InterpolationMode.BICUBIC, antialias = True):
    if min(img.shape[-2:]) == image_size:
        return img
    else:
        return T.functional.resize(img, image_size, interpolation, antialias)

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def cast_num_frames(t, *, frames):
    f = t.shape[-3]

    if f == frames:
        return t

    if f > frames:
        return t[..., :frames, :, :]
    return pad_at_dim(t, (0, frames - f), dim = -3)



def video_to_tensor(
    path: str,              
    num_frames = 100, 
) -> Tensor:                
    
    frames_torch = read_video(path, output_format="TCHW", pts_unit="sec")[0]
    frames_torch = frames_torch.permute(1,0,2,3).div(255.)
    return frames_torch[:, :num_frames, :, :]


def collate_tensors_and_strings(data):
    if is_bearable(data, List[Tensor]):
        return (torch.stack(data),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[Tensor, ...]):
            datum = torch.stack(datum)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)

def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn = collate_tensors_and_strings, **kwargs)

class VideoDataset(Dataset):
    def __init__(
        self,
        meta_path,
        image_size,
        channels = 3,
        num_frames = 17,
        force_num_frames = True,
        dtype=torch.float32

    ):
        super().__init__()
        meta_path = Path(meta_path)
        assert meta_path.is_file(), f'{str(meta_path)} must be a folder containing videos'
        self.dtype = dtype
        self.image_size = image_size
        self.num_frames = num_frames
        self.channels = channels
        self.meta_info = []
        with open(meta_path, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                # path, duration, frame_rate = line.split("\t", maxsplit=2)
                # duration = float(duration)
                path = line.strip()
                # frame_rate = float(frame_rate)
                # total_num_frames = math.floor(duration * frame_rate)
                # total_num_frames = num_frames
                self.meta_info.append((path,))
                # if total_num_frames < num_frames:
                #     continue
                # self.meta_info.append((path, duration, frame_rate))
        print(f'{len(self.meta_info)} training samples found with {meta_path}')

        self.transform = T.Compose([
            lambda x: resize(x, image_size, interpolation=T.InterpolationMode.BICUBIC, antialias = True),
            T.CenterCrop(image_size),
            T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5],inplace=True),
        ])

        # functions to transform video path to tensor

        # self.gif_to_tensor = partial(gif_to_tensor, channels = self.channels, transform = self.transform)

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, index):
        # path, duration, frame_rate = self.meta_info[index]
        (path,) = self.meta_info[index]
        ext = Path(path).suffix
        path_str = str(path)
        try:
            if ext in ['.mp4', '.avi', '.mkv']:
                tensor = video_to_tensor(path_str, num_frames=self.num_frames)

                frames = tensor.unbind(dim = 1)
                
                tensor = torch.stack([*map(self.transform, frames)], dim = 1).to(self.dtype)

            else:
                raise ValueError(f'unknown extension {ext}')
            return self.cast_num_frames_fn(tensor), str(index)
        except:
            print(f"ERR: {path_str}")
            # return torch.Tensor(torch.ones([3, 17, 256, 256])), str(index)

class ImageDataset(Dataset):
    def __init__(
        self,
        meta_path,
        image_size,
        channels = 3,
        dtype=torch.float32
    ):
        super().__init__()
        meta_path = Path(meta_path)
        assert meta_path.is_file(), f'{str(meta_path)} must be a folder containing images'
        self.dtype = dtype
        self.image_size = image_size
        self.channels = channels
        self.meta_info = []
        with open(meta_path, "r", encoding="utf8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                path = line.strip()
                self.meta_info.append(path)
        print(f'{len(self.meta_info)} training samples found with {meta_path}')

        self.transform = T.Compose([
            T.CenterCrop(image_size),
            lambda x:x.to(self.dtype),
            lambda x:x.div(127.5) - 1
        ])

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, index):
        path = self.meta_info[index]
        path_str = str(path)
        # try:
        img = Image.open(path_str)
        if img.mode != "RGB":
            img = img.convert("RGB")

        scaled_w, scaled_h = get_resized_shape(*img.size, target_size=self.image_size)
        img = img.resize((scaled_w, scaled_h), resample=Image.Resampling.BICUBIC)


        tensor = T.functional.pil_to_tensor(img)

        # tensor = T.functional.resize(tensor, size=(256,256), antialias = True)
        # except Exception as e:
        #     err_info = f">> ERR: INDEX: {index} | PATH: {path_str}"
        #     warnings.warn(err_info)
        #     # warnings.warn(str(e))
        #     tensor = torch.Tensor(torch.ones([3, 256, 256]))
        
        tensor = self.transform(tensor)
        return tensor, str(index)
