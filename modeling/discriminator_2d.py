import torch
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from modeling.magvit2 import GroupNormSpatial
from modeling.ema import LeCAM_EMA

PADDING_MODE="zeros"

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class BlurPool2D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer("filt", filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt.to(device=inp.device, dtype=inp.dtype), stride=self.stride, groups=inp.shape[1])

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ResBlockDown(nn.Module):
    """3D StyleGAN ResBlock for D."""

    def __init__(
        self,
        in_channels,
        filters,
        activation_fn,
        num_groups=32,
    ):
        super().__init__()

        self.filters = filters
        self.activation_fn = activation_fn

        self.conv1 = nn.Conv2d(
            in_channels, in_channels, (3, 3), padding=1, padding_mode=PADDING_MODE
        )
        # self.norm1 = GroupNormSpatial(num_groups, in_channels)

        self.blur = BlurPool2D(in_channels, filt_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels, self.filters, (1, 1), bias=False, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels, self.filters, (3, 3), padding=1, padding_mode=PADDING_MODE)
        # self.norm2 = GroupNormSpatial(num_groups, self.filters)
        

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.activation_fn(x)
        x = self.blur(x)
        residual = self.blur(residual)
        residual = self.conv2(residual)
        x = self.conv3(x)
        x = self.activation_fn(x)
        out = (residual + x) / math.sqrt(2)
        return out
    
class StyleGANDiscriminator2D(nn.Module):
    """StyleGAN Discriminator."""
    def __init__(self, config, use_lecam_ema=False):
        super().__init__()
        self.spatial_size = config.data.spatial_size
        self.filters = config.model.discriminator.filters
        self.activation_fn = nn.LeakyReLU(negative_slope=0.2)
        self.channel_multipliers = config.model.discriminator.channel_multipliers
        input_channels = [self.filters,] + [multiplier * self.filters for multiplier in self.channel_multipliers[:-1]]
        self.conv_in = nn.Conv2d(1, self.filters, kernel_size=(3, 3), padding=1, padding_mode=PADDING_MODE)

        self.res_blocks = nn.Sequential(
            *[ResBlockDown(in_channels=input_channels[i], filters=self.channel_multipliers[i] * self.filters, activation_fn=self.activation_fn) for i in range(len(self.channel_multipliers))]
        )

        # self.norm_fn = GroupNormSpatial(32, self.filters * self.channel_multipliers[-1], epsilon=1e-5)
        self.conv_out = nn.Conv2d(
            self.filters * self.channel_multipliers[-1], 
            self.filters * self.channel_multipliers[-1], 
            kernel_size=(3, 3),
            padding=1,
            padding_mode=PADDING_MODE)
        self.use_lecam_ema = use_lecam_ema
        if self.use_lecam_ema:
            self.lecam_ema = LeCAM_EMA()

        spatial_downsample_factor = 2 ** len(self.channel_multipliers)
        assert self.spatial_size % spatial_downsample_factor == 0, \
        f"spatial_size: {self.spatial_size}, spatial_downsample_factor: {spatial_downsample_factor}"
        
        self.logit_input_feats = (self.spatial_size // (spatial_downsample_factor)) ** 2 * self.filters * self.channel_multipliers[-1]
        
        self.linear0 = nn.Linear(
            self.logit_input_feats,
            self.filters * self.channel_multipliers[-1]
        )
        
        self.linear1 = nn.Linear(
            self.filters * self.channel_multipliers[-1],                
            1
        )     
        self.apply(init_weights)


    def forward(self, x):
        x = self.conv_in(x)
        x = self.activation_fn(x)
        x = self.res_blocks(x)
        x = self.conv_out(x)
        # x = self.norm_fn(x)
        x = self.activation_fn(x)
        x = x.view(x.shape[0], -1)
        assert self.logit_input_feats == x.shape[1]
        x = self.linear0(x)
        x = self.activation_fn(x)
        x = self.linear1(x)
        return x