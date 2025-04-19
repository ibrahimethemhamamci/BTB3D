
from omegaconf import OmegaConf
from torchvision import transforms as T
import torch

def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf

def calculate_resize_shape(source_image_size, target_pixels, min_square_size):
    w, h = source_image_size
    min_edge = min(w, h)
    scale = target_pixels / min_edge
    if min_edge == w:
        scaled_w = target_pixels
        scaled_h = int(h * scale)
    elif min_edge == h:
        scaled_h = target_pixels
        scaled_w = int(w * scale)
    scaled_w = scaled_w  - scaled_w % min_square_size
    scaled_h = scaled_h  - scaled_h % min_square_size

    return scaled_h, scaled_w


def preprocess_vision_input(frames, resize_shape):

    trans_pipe = [
            T.Resize(resize_shape),
            lambda x:x.div(127.5) - 1
        ]
    
    transform = T.Compose(trans_pipe)
    frames = frames.unbind(dim = 1)
    tensor = torch.stack([*map(transform, frames)], dim = 1)
    return tensor