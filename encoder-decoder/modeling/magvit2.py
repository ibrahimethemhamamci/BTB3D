from torch import nn
from typing import Tuple
import torch
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Sequence
from einops import rearrange

PADDING_MODE="zeros"

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: int | Tuple[int, int, int],
        stride: Tuple[int, int, int] = (1,1,1),
        pad_mode = 'constant',
        use_bias=True,
        disable_spatial_padding=False,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        self.time_kernel_size, self.height_kernel_size, self.width_kernel_size = kernel_size

        self.pad_mode = pad_mode

        time_pad = self.time_kernel_size - 1
        
        # Calculate spatial padding (same for both videos and single images)
        if disable_spatial_padding:
            height_pad = width_pad = 0
        else:
            height_pad = self.height_kernel_size // 2
            width_pad = self.width_kernel_size // 2

        # Padding tuple: (left, right, top, bottom, front, back)
        # Note: No padding after the input in temporal dimension (back = 0)
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_pad = time_pad
        # Create the 3D convolution layer
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, bias=use_bias)
            
    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        h, w = x.shape[-2:]
        # For single images: x.shape[2] (time dimension) will be 1
        # For videos: x.shape[2] will be > 1
        # Choose padding mode based on input temporal dimension
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'
        
        # Apply padding
        # For single images: This adds zeros before the single frame in time dimension
        # For videos: This adds zeros before the first frame, ensuring causality
        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        # Apply 3D convolution
        # For single images: The convolution will still work, treating it as a single-frame video
        # For videos: The convolution processes the sequence causally
        return self.conv(x)



def depth_to_space(x, t_stride, filters, space_scaling_factor=2):
    if x.ndim == 5:
        b, t, h, w, _ = x.shape
        x = x.view(b, t, h, w, t_stride, space_scaling_factor, space_scaling_factor, filters)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7) # (b, t, t_stride, h, 2, w, 2, filters)
        x = x.reshape(b, t * t_stride, h * space_scaling_factor, w * space_scaling_factor, filters)
    else:
        b, h, w, _ = x.shape
        x = x.view(b, h, w, space_scaling_factor, space_scaling_factor, filters)
        x = x.permute(0, 1, 3, 2, 4, 5) # (b, h, 2, w, 2, filters)
        x = x.reshape(b, h * space_scaling_factor, w * space_scaling_factor, filters)
    return x


class DepthToSpaceUpsampler2D(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size=(3,3),
        stride=(1,1),
        use_bias=True
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, bias=use_bias, padding="same")
        self.depth2space = depth_to_space

    def forward(self, x, output_filters, space_scaling_factor=2):
        """
        input_image: [B C H W]
        """

        out = self.conv1(x)

        out = rearrange(out, "b c h w -> b h w c")
        out = self.depth2space(out, t_stride=1, filters=output_filters, space_scaling_factor=space_scaling_factor)
        out = rearrange(out, "b h w c -> b c h w")

        return out

class DepthToSpaceUpsampler(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size=(3,3,3),
        stride=(1,1,1),
        use_bias=True
    ):
        super().__init__()
        self.conv1 = CausalConv3d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, use_bias=use_bias, pad_mode="constant")
        self.depth2space = depth_to_space

    def forward(self, x, t_stride, output_filters, space_scaling_factor=2):
        """
        input_image: [B C T H W]
        """


        out = self.conv1(x)

        out = rearrange(out, "b c t h w -> b t h w c")
        out = self.depth2space(out, t_stride=t_stride, filters=output_filters, space_scaling_factor=space_scaling_factor)
        out = rearrange(out, "b t h w c -> b c t h w")

        return out
    

class GroupNormSpatial(nn.Module):
    """GroupNorm with spatial dimensions ignored."""
    def __init__(self, num_groups, num_channels, epsilon: float = 1e-5, affine=True):
        super().__init__()
        self.norm_fn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=epsilon, affine=affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 5: # video
            b, c, t, h, w = inputs.shape
            inputs = rearrange(inputs, "b c t h w -> (b t) c h w")
            out = self.norm_fn(inputs)
            return rearrange(out, "(b t) c h w -> b c t h w", b=b, t=t)
        else: # Image, b c h w -> b c h w
            out = self.norm_fn(inputs)
            return out


class AdaptiveGroupNormSpatial(nn.Module):
    """Conditional normalization layer."""
    def __init__(self, emb_dim: int, dim: int, **norm_kwargs):
        super().__init__()
        self.norm = GroupNormSpatial(**norm_kwargs)
        self.dim = dim
        self.emb_dim = emb_dim
        self.gamma = nn.Linear(in_features=emb_dim, out_features=dim, bias=True)
        self.beta = nn.Linear(in_features=emb_dim, out_features=dim, bias=True)

    def forward(self, inputs: Tensor, emb: Tensor) -> Tensor:
        # n c t h w -> n t h w c
        assert emb.shape[1] == self.emb_dim
        x = self.norm(inputs)
        if inputs.ndim == 5: # for video
            emb = rearrange(emb, "n c t h w -> n t h w c")
            gamma = self.gamma(emb)
            beta = self.beta(emb)
            gamma = rearrange(gamma, "n t h w c -> n c t h w")
            beta = rearrange(beta, "n t h w c -> n c t h w")
        else: # for image
            emb = rearrange(emb, "n c h w -> n h w c")
            gamma = self.gamma(emb)
            beta = self.beta(emb)
            gamma = rearrange(gamma, "n h w c -> n c h w")
            beta = rearrange(beta, "n h w c -> n c h w")

        return x * (gamma + 1.0) + beta


class ResBlock(nn.Module):

    def __init__(self, chan_in, chan_out, use_conv_shortcut):
        super().__init__()
        self.use_bias = False
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.use_conv_shortcut = use_conv_shortcut
        self.norm_0 = GroupNormSpatial(num_groups=32, num_channels=chan_in, epsilon=1e-5)
        self.activation = nn.SiLU()
        self.conv_0 = CausalConv3d(chan_in=chan_in, chan_out=chan_out, kernel_size=(3,3,3), stride=(1,1,1), use_bias=False, pad_mode="constant")
        self.norm_1 = GroupNormSpatial(num_groups=32, num_channels=chan_out, epsilon=1e-5)
        self.conv_1 = CausalConv3d(chan_in=chan_out, chan_out=chan_out, kernel_size=(3,3,3), stride=(1,1,1), use_bias=False, pad_mode="constant")

        if self.chan_in != self.chan_out:
            
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(chan_in=chan_in, chan_out=chan_out, kernel_size=(3,3,3), stride=(1,1,1), use_bias=False, pad_mode="constant")
            else:
                self.conv_shortcut = CausalConv3d(chan_in=chan_in, chan_out=chan_out, kernel_size=(1,1,1), stride=(1,1,1), use_bias=False, pad_mode="constant")


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
                 temporal_downsample: Sequence[bool] = (False, True, True),
                 channel_multipliers: Sequence[int] = (1, 2, 2, 4),
                 skip_conv_first = False
                 ):
        super().__init__()
        if num_res_blocks < 1:
            raise ValueError('num_res_blocks must be >= 1.')
        self.filters = filters
        self.chan_out = chan_out
        self.num_res_blocks = num_res_blocks
        self.temporal_downsample = temporal_downsample
        self.channel_multipliers = channel_multipliers
        self.num_blocks = len(self.channel_multipliers)
        self.skip_conv_first = skip_conv_first

        input_filters = chan_in

        output_filters = self.filters * self.channel_multipliers[-1]
        if not skip_conv_first:
            self.conv_first = CausalConv3d(input_filters, output_filters, kernel_size=(3,3,3), use_bias=True, pad_mode="constant")

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


            res_blocks_i = OrderedDict()
            res_blocks_i[f'res_block_{self.num_blocks - 1 - i}_0'] = ResBlock(chan_in=input_filters, chan_out=output_filters, use_conv_shortcut=False)

            res_blocks_i.update(
                [(f'res_block_{self.num_blocks - 1 - i}_{j}',ResBlock(chan_in=output_filters, chan_out=output_filters, use_conv_shortcut=False))
                 for j in range(1, self.num_res_blocks)])

            self.res_blocks[f"res_block_{self.num_blocks - 1 - i}"] = nn.Sequential(res_blocks_i)

            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                conv_upsample_chan_out = output_filters * 4 * t_stride
                self.conv_upsample_blocks[f'conv_upsample_{self.num_blocks - 1 - i}'] = DepthToSpaceUpsampler(
                    chan_in=output_filters,
                    chan_out=conv_upsample_chan_out, 
                    kernel_size=(3,3,3))
            input_filters = output_filters

        self.final_norm = GroupNormSpatial(num_groups=32, num_channels=output_filters)
        self.activation = nn.SiLU()
        self.conv_last = CausalConv3d(output_filters, chan_out, kernel_size=(3,3,3), use_bias=True, pad_mode="constant")


    def forward(self, inputs: Tensor) -> Tensor:
        cond = inputs
        if not self.skip_conv_first:
            x = self.conv_first(inputs)
        else:
            x = inputs
        x = self.res_blocks["res_block_pre"](x)

        for i in reversed(range(self.num_blocks)):
            x = self.cond_norm_layers[f"cond_norm_{self.num_blocks - 1 - i}"](x, cond)
            x = self.res_blocks[f"res_block_{self.num_blocks - 1 - i}"](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                output_filters = self.filters * self.channel_multipliers[i]
                space_scaling_factor = 2
                x = self.conv_upsample_blocks[f"conv_upsample_{self.num_blocks - 1 - i}"](x, t_stride, output_filters, space_scaling_factor)

                cond = F.interpolate(
                    cond,
                    x.shape[2:],
                    mode='nearest',
                    antialias=False,
                )
          
                x = x[:, :, t_stride - 1 :]
                cond = cond[:, :, t_stride - 1 :]
        x = self.final_norm(x)
        x = self.activation(x)
        #print(x.shape, "test")
        x = self.conv_last(x)
        return x


class Encoder(nn.Module):
    """Decoder structure with 3D CNNs."""

    def __init__(self,
                 filters: int = 128,
                 chan_in: int = 18,
                 chan_out: int = 3,
                 num_res_blocks: int = 4,
                 temporal_downsample: Sequence[bool] = (False, True, True),
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
        self.temporal_downsample = temporal_downsample
        self.channel_multipliers = channel_multipliers
        self.num_blocks = len(self.channel_multipliers)
        self.skip_conv_first = skip_conv_first
        self.skip_conv_last = skip_conv_last

        input_filters = chan_in
        output_filters = self.filters * self.channel_multipliers[0]
        if not skip_conv_first:
            self.conv_first = CausalConv3d(input_filters, output_filters, kernel_size=(3,3,3), use_bias=True, pad_mode="constant")

        input_filters = self.filters

        self.res_blocks = nn.ModuleDict()
        self.conv_downsample_blocks = nn.ModuleDict()

        for i in range(self.num_blocks):

            output_filters = self.filters * self.channel_multipliers[i]

            res_blocks_i = OrderedDict()
            res_blocks_i[f'res_block_{i}_0'] = ResBlock(chan_in=input_filters, chan_out=output_filters, use_conv_shortcut=False)

            res_blocks_i.update(
                [(f'res_block_{0}_{j}',ResBlock(chan_in=output_filters, chan_out=output_filters, use_conv_shortcut=False))
                 for j in range(1, self.num_res_blocks)])

            self.res_blocks[f"res_block_{i}"] = nn.Sequential(res_blocks_i)

            if i < self.num_blocks - 1:
                t_stride = 2 if self.temporal_downsample[i] else 1
                spatial_stride = (2,2)
                self.conv_downsample_blocks[f'conv_downsample_{i}'] = CausalConv3d(
                    chan_in=output_filters,
                    chan_out=output_filters,
                    kernel_size=(3,3,3),
                    stride=(t_stride, *spatial_stride),
                    use_bias=True,
                    pad_mode="constant"
                )
            input_filters = output_filters

        self.res_blocks["res_block_final"] = nn.Sequential(OrderedDict(
            [(f"res_block_final_{j}", ResBlock(output_filters, output_filters, use_conv_shortcut=False)) for j in range(self.num_res_blocks)]
            )
        )
        self.final_norm = GroupNormSpatial(num_groups=32, num_channels=output_filters)
        self.activation = nn.SiLU()
        if not self.skip_conv_last:
            self.conv_last = CausalConv3d(output_filters, chan_out, kernel_size=(1,1,1), use_bias=True, pad_mode="constant")
        else:
            self.conv_last = nn.Identity()


    def forward(self, inputs: Tensor) -> Tensor:
        if not self.skip_conv_first:
            x = self.conv_first(inputs)
        else:
            x = inputs
        for i in range(self.num_blocks):

            x = self.res_blocks[f"res_block_{i}"](x)

            if i < self.num_blocks - 1:
                t_stride = 2 if self.temporal_downsample[i] else 1
                output_filters = self.filters * self.channel_multipliers[i]
                x = self.conv_downsample_blocks[f"conv_downsample_{i}"](x)

    
        x = self.res_blocks["res_block_final"](x)


        x = self.final_norm(x)
        x = self.activation(x)
        x = self.conv_last(x)


        return x