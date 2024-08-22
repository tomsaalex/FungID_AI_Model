from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce


# # Source: https://github.com/lucidrains/g-mlp-pytorch/blob/main/g_mlp_pytorch/g_mlp_pytorch.py
#
# # functions
#
# def exists(val):
#     return val is not None
#
#
# def pair(val):
#     return (val, val) if not isinstance(val, tuple) else val
#
#
# def dropout_layers(layers, prob_survival):
#     if prob_survival == 1:
#         return layers
#
#     num_layers = len(layers)
#     to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival
#
#     # make sure at least one layer makes it
#     if all(to_drop):
#         rand_index = randrange(num_layers)
#         to_drop[rand_index] = False
#
#     layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
#     return layers
#
#
# def shift(t, amount, mask=None):
#     if amount == 0:
#         return t
#     return F.pad(t, (0, 0, amount, -amount), value=0.)
#
#
# # helper classes
#
# class Residual(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x):
#         return self.fn(x) + x
#
#
# class PreShiftTokens(nn.Module):
#     def __init__(self, shifts, fn):
#         super().__init__()
#         self.fn = fn
#         self.shifts = tuple(shifts)
#
#     def forward(self, x, **kwargs):
#         if self.shifts == (0,):
#             return self.fn(x, **kwargs)
#
#         shifts = self.shifts
#         segments = len(shifts)
#         feats_per_shift = x.shape[-1] // segments
#         splitted = x.split(feats_per_shift, dim=-1)
#         segments_to_shift, rest = splitted[:segments], splitted[segments:]
#         segments_to_shift = list(map(lambda args: shift(*args), zip(segments_to_shift, shifts)))
#         x = torch.cat((*segments_to_shift, *rest), dim=-1)
#         return self.fn(x, **kwargs)
#
#
# class PreNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.fn = fn
#         self.norm = nn.LayerNorm(dim)
#
#     def forward(self, x, **kwargs):
#         x = self.norm(x)
#         return self.fn(x, **kwargs)
#
#
# class Attention(nn.Module):
#     def __init__(self, dim_in, dim_out, dim_inner, causal=False):
#         super().__init__()
#         self.scale = dim_inner ** -0.5
#         self.causal = causal
#
#         self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias=False)
#         self.to_out = nn.Linear(dim_inner, dim_out)
#
#     def forward(self, x):
#         device = x.device
#         q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
#
#         if self.causal:
#             mask = torch.ones(sim.shape[-2:], device=device).triu(1).bool()
#             sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)
#
#         attn = sim.softmax(dim=-1)
#         out = einsum('b i j, b j d -> b i d', attn, v)
#         return self.to_out(out)
#
#
# class SpatialGatingUnit(nn.Module):
#     def __init__(
#             self,
#             dim,
#             dim_seq,
#             causal=False,
#             act=nn.Identity(),
#             heads=1,
#             init_eps=1e-3,
#             circulant_matrix=False
#     ):
#         super().__init__()
#         dim_out = dim // 2
#         self.heads = heads
#         self.causal = causal
#         self.norm = nn.LayerNorm(dim_out)
#
#         self.act = act
#
#         # parameters
#
#         if circulant_matrix:
#             self.circulant_pos_x = nn.Parameter(torch.ones(heads, dim_seq))
#             self.circulant_pos_y = nn.Parameter(torch.ones(heads, dim_seq))
#
#         self.circulant_matrix = circulant_matrix
#         shape = (heads, dim_seq,) if circulant_matrix else (heads, dim_seq, dim_seq)
#         weight = torch.zeros(shape)
#
#         self.weight = nn.Parameter(weight)
#         init_eps /= dim_seq
#         nn.init.uniform_(self.weight, -init_eps, init_eps)
#
#         self.bias = nn.Parameter(torch.ones(heads, dim_seq))
#
#     def forward(self, x, gate_res=None):
#         device, n, h = x.device, x.shape[1], self.heads
#
#         res, gate = x.chunk(2, dim=-1)
#         gate = self.norm(gate)
#
#         weight, bias = self.weight, self.bias
#
#         if self.circulant_matrix:
#             # build the circulant matrix
#
#             dim_seq = weight.shape[-1]
#             weight = F.pad(weight, (0, dim_seq), value=0)
#             weight = repeat(weight, '... n -> ... (r n)', r=dim_seq)
#             weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
#             weight = weight[:, :, (dim_seq - 1):]
#
#             # give circulant matrix absolute position awareness
#
#             pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
#             weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')
#
#         if self.causal:
#             weight, bias = weight[:, :n, :n], bias[:, :n]
#             mask = torch.ones(weight.shape[-2:], device=device).triu_(1).bool()
#             mask = rearrange(mask, 'i j -> () i j')
#             weight = weight.masked_fill(mask, 0.)
#
#         gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)
#
#         gate = einsum('b h n d, h m n -> b h m d', gate, weight)
#         gate = gate + rearrange(bias, 'h n -> () h n ()')
#
#         gate = rearrange(gate, 'b h n d -> b n (h d)')
#
#         if exists(gate_res):
#             gate = gate + gate_res
#
#         return self.act(gate) * res
#
#
# class gMLPBlock(nn.Module):
#     def __init__(
#             self,
#             *,
#             dim,
#             dim_ff,
#             seq_len,
#             heads=1,
#             attn_dim=None,
#             causal=False,
#             act=nn.Identity(),
#             circulant_matrix=False
#     ):
#         super().__init__()
#         self.proj_in = nn.Sequential(
#             nn.Linear(dim, dim_ff),
#             nn.GELU()
#         )
#
#         self.attn = Attention(dim, dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None
#
#         self.sgu = SpatialGatingUnit(dim_ff, seq_len, causal, act, heads, circulant_matrix=circulant_matrix)
#         self.proj_out = nn.Linear(dim_ff // 2, dim)
#
#     def forward(self, x):
#         gate_res = self.attn(x) if exists(self.attn) else None
#
#         x = x.permute(0, 2, 3, 1)
#         x = self.proj_in(x)
#         #x = x.permute(0, 3, 1, 2)
#         x = x.flatten(start_dim=1, end_dim=2)
#         x = self.sgu(x, gate_res=gate_res)
#         x = self.proj_out(x)
#         return x

# Source: https://github.com/jaketae/g-mlp/blob/master/g_mlp/core.py

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)
        self.channel_proj2 = nn.Linear(d_ffn, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = x + residual
        return out

class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.model(x)


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class gMLPForImageClassification(gMLP):
    def __init__(
            self,
            image_size=28,
            patch_size=7,
            in_channels=3,
            d_model=28,
            d_ffn=112,
            seq_len=16,
            num_layers=1,
    ):
        super().__init__(d_model, d_ffn, seq_len, num_layers)
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    # def forward(self, x):
    #     x = x.permute(0, 3, 1, 2)
    #     patches = self.patcher(x)
    #     batch_size, num_channels, _, _ = patches.shape
    #     patches = patches.permute(0, 2, 3, 1)
    #     patches = patches.view(batch_size, -1, num_channels)
    #     embedding = self.model(patches)
    #     return embedding

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        B, C, H, W = x.shape

        x = x.view(B, C, -1)  # (B, C, H*W)
        x = x.permute(0, 2, 1)  # (B, N, D) where N = H*W, D = C

        #batch_size, num_channels, _, _ = patches.shape
        #patches = patches.permute(0, 2, 3, 1)
        #patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(x)

        x = embedding.permute(0, 2, 1)  # (B, D, N)
        x = x.view(B, C, H, W)  # (B, C, H, W)

        return x
