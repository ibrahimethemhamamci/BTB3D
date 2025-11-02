from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Sequence
from einops import rearrange
from modeling.magvit2 import GroupNormSpatial, AdaptiveGroupNormSpatial, DepthToSpaceUpsampler2D

PADDING_MODE="zeros"

class ResBlock(nn.Module):

    def __init__(self, chan_in, chan_out, use_conv_shortcut):
        super().__init__()
        self.use_bias = False
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.use_conv_shortcut = use_conv_shortcut
        self.norm_0 = GroupNormSpatial(num_groups=32, num_channels=chan_in, epsilon=1e-5)
        self.activation = nn.SiLU()
        self.conv_0 = nn.Conv2d(chan_in, chan_out, kernel_size=(3,3), stride=(1,1), bias=False, padding=1, padding_mode=PADDING_MODE)
        self.norm_1 = GroupNormSpatial(num_groups=32, num_channels=chan_out, epsilon=1e-5)
        self.conv_1 = nn.Conv2d(chan_out, chan_out, kernel_size=(3,3), stride=(1,1), bias=False, padding=1, padding_mode=PADDING_MODE)

        if self.chan_in != self.chan_out:
            
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(chan_in, chan_out, kernel_size=(3,3), stride=(1,1), bias=self.use_bias, padding=1, padding_mode=PADDING_MODE)
            else:
                self.conv_shortcut = nn.Conv2d(chan_in, chan_out, kernel_size=(1,1), stride=(1,1), bias=self.use_bias, padding=0)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chan_in != inputs.shape[1]:
            raise ValueError(
                f'Input dimension {inputs.shape[1]} does not match {self.chan_in}'
            )
        residual = inputs
        x = self.norm_0(inputs)
        x = self.activation(x)
        x = self.conv_0(x)
        x = self.norm_1(x)
        x = self.activation(x)
        x = self.conv_1(x)
        if self.chan_in != self.chan_out:
            residual = self.conv_shortcut(residual)
        return x + residual

class Decoder(nn.Module):
    """Decoder structure with 3D CNNs."""


    def __init__(self,
                 filters: int = 128,
                 chan_in: int = 18,
                 chan_out: int = 3,
                 num_res_blocks: int = 4,
                 channel_multipliers: Sequence[int] = (1, 2, 2, 4),
                 skip_conv_first = False
                 ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError('num_res_blocks must be >= 1.')
        self.filters = filters
        self.chan_out = chan_out
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.num_blocks = len(self.channel_multipliers)
        self.skip_conv_first = skip_conv_first

        input_filters = chan_in
        output_filters = self.filters * self.channel_multipliers[-1]
        if not skip_conv_first:
            self.conv_first = nn.Conv2d(input_filters, output_filters, kernel_size=(3,3), bias=True, padding=1, padding_mode=PADDING_MODE)

        input_filters = self.filters * self.channel_multipliers[-1]

        self.res_blocks = nn.ModuleDict()
        self.conv_upsample_blocks = nn.ModuleDict()
        self.cond_norm_layers = nn.ModuleDict()
        self.res_blocks["res_block_pre"] = nn.Sequential(OrderedDict(
            [(f"res_block_pre_{j}", ResBlock(output_filters, output_filters, use_conv_shortcut=False)) for j in range(self.num_res_blocks)]
            )
        )

        for i in reversed(range(self.num_blocks)):
            self.cond_norm_layers[f'cond_norm_{self.num_blocks - 1 - i}'] = AdaptiveGroupNormSpatial(
                emb_dim=chan_in,
                dim=output_filters,
                num_groups=32,
                num_channels=output_filters,
                affine=False
            )

            output_filters = self.filters * self.channel_multipliers[i]

            # this part is to initialize the first layer in each of the 4 resblock groups
            # Besides the previous 4 * Resblock 512 which is right after The conv_first,
            # the rest resblocks can be divided as 4 groups:
            # 1. ResBlock 512 -> 512, (ResBlock 512 * 3)
            # 2. ResBlock 512 -> 256, (ResBlock 256 * 3)
            # 3. ResBlock 256 -> 256, (ResBlock 256 * 3)
            # 4. ResBlock 256 -> 128, (ResBlock 128 * 3)
            # ====
            res_blocks_i = OrderedDict()
            res_blocks_i[f'res_block_{self.num_blocks - 1 - i}_0'] = ResBlock(chan_in=input_filters, chan_out=output_filters, use_conv_shortcut=False)
            # ====

            # This part is to initialize the last 3 layers in each of the 4 resblock groups
            # each is with same input and output dim
            # 1. (ResBlock 512 -> 512), ResBlock 512 * 3
            # 2. (ResBlock 512 -> 256), ResBlock 256 * 3
            # 3. (ResBlock 256 -> 256), ResBlock 256 * 3
            # 4. (ResBlock 256 -> 128), ResBlock 128 * 3
            # ====
            res_blocks_i.update(
                [(f'res_block_{self.num_blocks - 1 - i}_{j}',ResBlock(chan_in=output_filters, chan_out=output_filters, use_conv_shortcut=False))
                 for j in range(1, self.num_res_blocks)])
            # ====
            self.res_blocks[f"res_block_{self.num_blocks - 1 - i}"] = nn.Sequential(res_blocks_i)


            # For the first 3 resblock groups, we append a T-Causal conv layer followed by a depth_to_space operator
            # to each of them
            # ====
            if i > 0:
                # depth_to_space: 1x2, 2x2, 2x2
                conv_upsample_chan_out = output_filters * 4
                self.conv_upsample_blocks[f'conv_upsample_{self.num_blocks - 1 - i}'] = DepthToSpaceUpsampler2D(
                    chan_in=output_filters,  # chan_in： 512, 256, 256
                    chan_out=conv_upsample_chan_out, # chan_out: 4096, 2048, 1024
                    kernel_size=(3,3))
            input_filters = output_filters
            # ====

        self.final_norm = GroupNormSpatial(num_groups=32, num_channels=output_filters)
        self.activation = nn.SiLU()
        self.conv_last = nn.Conv2d(output_filters, chan_out, kernel_size=(3,3), bias=True, padding=1, padding_mode=PADDING_MODE)


    def forward(self, inputs: Tensor) -> Tensor:
        cond = inputs
        if not self.skip_conv_first:
            x = self.conv_first(inputs)
        else:
            x = inputs
        # 4 * resblock 512
        x = self.res_blocks["res_block_pre"](x)

        # [3,2,1,0]
        for i in reversed(range(self.num_blocks)):

            # 0 -> 1 -> 2 -> 3
            # adaptive group norm

            x = self.cond_norm_layers[f"cond_norm_{self.num_blocks - 1 - i}"](x, cond)

            # 1 * ResBlock x-> x/y + 3 * ResBlock x
            x = self.res_blocks[f"res_block_{self.num_blocks - 1 - i}"](x)

            # For the first 3 resblock groups, we append a T-Causal conv layer followed by
            # a depth_to_space operator to each of them
            if i > 0:
                output_filters = self.filters * self.channel_multipliers[i]
                space_scaling_factor = 2
                x = self.conv_upsample_blocks[f"conv_upsample_{self.num_blocks - 1 - i}"](x, output_filters, space_scaling_factor)

                cond = F.interpolate(
                    cond,
                    x.shape[2:],
                    mode='nearest',
                    antialias=False,
                )

        x = self.final_norm(x)
        x = self.activation(x)
        x = self.conv_last(x)
        return x



class Encoder(nn.Module):
    """Decoder structure with 3D CNNs."""
    def __init__(self,
                 filters: int = 128,
                 chan_in: int = 18,
                 chan_out: int = 3,
                 num_res_blocks: int = 4,
                 channel_multipliers: Sequence[int] = (1, 2, 2, 4),
                 skip_conv_first = False,
                 skip_conv_last = False
                 ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError('num_res_blocks must be >= 1.')
        self.filters = filters
        self.chan_out = chan_out
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = channel_multipliers
        self.num_blocks = len(self.channel_multipliers)
        self.skip_conv_first = skip_conv_first
        self.skip_conv_last = skip_conv_last

        input_filters = chan_in
        output_filters = self.filters * self.channel_multipliers[0]
        if not skip_conv_first:
            self.conv_first = nn.Conv2d(input_filters, output_filters, kernel_size=(3,3), bias=True, padding=1, padding_mode=PADDING_MODE)

        input_filters = self.filters

        self.res_blocks = nn.ModuleDict()
        self.conv_downsample_blocks = nn.ModuleDict()

        for i in range(self.num_blocks):

            output_filters = self.filters * self.channel_multipliers[i]

            # this part is to initialize the first layer in each of the 4 resblock groups
            # Besides the previous 4 * Resblock 512 which is right after The conv_first,
            # the rest resblocks can be divided as 4 groups:
            # 1. ResBlock 128 -> 128, (ResBlock 128 * 3)
            # 2. ResBlock 128 -> 256, (ResBlock 256 * 3)
            # 3. ResBlock 256 -> 256, (ResBlock 256 * 3)
            # 4. ResBlock 256 -> 512, (ResBlock 512 * 3)
            # ====
            res_blocks_i = OrderedDict()
            res_blocks_i[f'res_block_{i}_0'] = ResBlock(chan_in=input_filters, chan_out=output_filters, use_conv_shortcut=False)
            # ====

            # This part is to initialize the last 3 layers in each of the 4 resblock groups
            # each is with same input and output dim
            # 1. (ResBlock 128) -> 128, ResBlock 128 * 3
            # 2. (ResBlock 128) -> 256, ResBlock 256 * 3
            # 3. (ResBlock 256) -> 256, ResBlock 256 * 3
            # 4. (ResBlock 256) -> 512, ResBlock 512 * 3
            # ====
            res_blocks_i.update(
                [(f'res_block_{0}_{j}',ResBlock(chan_in=output_filters, chan_out=output_filters, use_conv_shortcut=False))
                 for j in range(1, self.num_res_blocks)])
            # ====

            self.res_blocks[f"res_block_{i}"] = nn.Sequential(res_blocks_i)


            # For the last 3 resblock groups, we append a T-Causal conv layer
            # to each of them
            # ====
            if i < self.num_blocks - 1:
                spatial_stride = (2,2)
                self.conv_downsample_blocks[f'conv_downsample_{i}'] = nn.Conv2d(
                    in_channels=output_filters,
                    out_channels=output_filters,
                    kernel_size=(3,3),
                    stride=spatial_stride,
                    bias=True,
                    padding=1,
                    padding_mode=PADDING_MODE
                )
            input_filters = output_filters
            # ====

        self.res_blocks["res_block_final"] = nn.Sequential(OrderedDict(
            [(f"res_block_final_{j}", ResBlock(output_filters, output_filters, use_conv_shortcut=False)) for j in range(self.num_res_blocks)]
            )
        )
        self.final_norm = GroupNormSpatial(num_groups=32, num_channels=output_filters)
        self.activation = nn.SiLU()
        if not self.skip_conv_last:
            self.conv_last = nn.Conv2d(output_filters, chan_out, kernel_size=(1,1), bias=True, padding=0)
        else:
            self.conv_last = nn.Identity()


    def forward(self, inputs: Tensor) -> Tensor:
        if not self.skip_conv_first:
            x = self.conv_first(inputs)
        else:
            x = inputs
        # [0,1,2,3]
        for i in range(self.num_blocks):
            # 0 -> 1 -> 2 -> 3
            # adaptive group norm

            # 1 * ResBlock x-> x/y + 3 * ResBlock x
            x = self.res_blocks[f"res_block_{i}"](x)
            # For the first 3 resblock groups, we append a T-Causal conv layer followed by
            # a depth_to_space operator to each of them
            if i < self.num_blocks - 1:
                x = self.conv_downsample_blocks[f"conv_downsample_{i}"](x)

    
        x = self.res_blocks["res_block_final"](x)


        x = self.final_norm(x)
        x = self.activation(x)
        x = self.conv_last(x)


        return x