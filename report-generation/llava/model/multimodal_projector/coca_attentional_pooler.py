import logging
import pdb
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many


def get_3d_positional_encoding(d_model, X, Y, Z, device=None):
    """
    Generate a 3D sinusoidal positional encoding.

    Args:
        d_model (int): Total number of channels (must be divisible by 3).
        X, Y, Z (int): Spatial dimensions.
        device: Torch device.

    Returns:
        pos_encoding: A tensor of shape (1, X, Y, Z, d_model) to be added to the input.
    """
    if device is None:
        device = torch.device("cpu")
    assert d_model % 3 == 0, "d_model must be divisible by 3"
    d_each = d_model // 3

    # Create coordinate grids for each spatial dimension.
    x_range = torch.arange(X, dtype=torch.bfloat16, device=device)
    y_range = torch.arange(Y, dtype=torch.bfloat16, device=device)
    z_range = torch.arange(Z, dtype=torch.bfloat16, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing="ij")
    # grid_x, grid_y, grid_z have shape (X, Y, Z)

    def get_1d_pe(grid, d):
        """
        Generate sinusoidal encoding for a 1D grid.

        Args:
            grid: Tensor of shape (X, Y, Z)
            d: Number of channels for this axis.
        Returns:
            pos_enc: Tensor of shape (X, Y, Z, d)
        """
        dim_t = torch.arange(d, dtype=torch.bfloat16, device=device)
        # Compute the denominator using the formula from "Attention is All You Need"
        div_term = torch.pow(10000, (2 * (dim_t // 2)) / d)
        grid = grid.unsqueeze(-1)  # Now shape (X, Y, Z, 1)
        pe = grid / div_term  # Broadcast division, shape: (X, Y, Z, d)
        pe_out = torch.zeros_like(pe)
        # Apply sin to even indices and cos to odd indices
        pe_out[..., 0::2] = torch.sin(pe[..., 0::2])
        pe_out[..., 1::2] = torch.cos(pe[..., 1::2])
        return pe_out

    pe_x = get_1d_pe(grid_x, d_each)
    pe_y = get_1d_pe(grid_y, d_each)
    pe_z = get_1d_pe(grid_z, d_each)

    # Concatenate along the channel dimension to get a final shape (X, Y, Z, d_model)
    pos_encoding = torch.cat([pe_x, pe_y, pe_z], dim=-1)
    # Add a batch dimension: shape becomes (1, X, Y, Z, d_model)
    pos_encoding = pos_encoding.unsqueeze(0)
    return pos_encoding


class AttentionalPoolProjector(nn.Module):
    def __init__(
            self,
            embed_dim,
            context_dim,
            projector = None,
            n_head = 8,
            n_queries = 256,
            norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        #print(n_queries,"n_queries")
        #self.attn_pool = AttentionalPooler(d_model=embed_dim,
         #                                  context_dim=context_dim,
         #                                  n_head=n_head,
         #                                  n_queries=n_queries)
        #self.ln = norm_layer(embed_dim)
        #remove attentionalpool for now
        #
        self.proj = projector if projector else nn.Identity()

    def forward(self, x: torch.Tensor):
        print(x.shape)
        #x = x.cuda()
        #print(x.device)
        #print(self.attn_pool.device)

        B, X_dim, Y, Z, D = x.shape
        #pe = get_3d_positional_encoding(D, X_dim, Y, Z, device=x.device).half()
        #x = x + pe
        x = x.flatten(1, 3)
        #print(x)
        #tokens = self.attn_pool(x)
        #tokens = self.ln(tokens)
        ## remove attentionalpool for now
        tokens = x
        tokens = self.proj(tokens)
        print(tokens.shape, "token shape")
        return tokens

class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))

        dim_head = d_model // n_head

        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head

        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')
        #x =
        q = repeat(self.query, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        x = self.ln_k(x)
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(q)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out).squeeze(dim=1)