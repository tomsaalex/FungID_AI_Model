import math

import torch
from torch import nn

from model.m_vit_model.g_mlp_pytorch import gMLPForImageClassification
from model.m_vit_model.self_attention import Self_Attn
from timm.models.mlp_mixer import gmlp_b16_224


class GridAttention(nn.Module):
    def __init__(self, channels, width, height, grid_size):
        super(GridAttention, self).__init__()

        self.batch_size = None
        self.input_shape = (channels, width, height)
        self.grid_size = grid_size
        self.layer_norm = nn.LayerNorm((channels, width, height))
        self.layer_norm2 = nn.LayerNorm((width, height, channels))
        self.depth_wise_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                                         groups=channels)  # TODO assumed size in (I think) absence of data
        self.self_attention = Self_Attn(channels)

        self.gmlpBlock = gMLPForImageClassification(
            image_size=width,
            patch_size=grid_size,
            in_channels=channels,
            d_model=channels,
            d_ffn=channels * 4,
            seq_len=(width * height),  # // (window_size * window_size),
            num_layers=1
        )

        self.unfold = torch.nn.Unfold(kernel_size=self.grid_size, stride=self.grid_size)
        self.fold = torch.nn.Fold(output_size=(width, height), kernel_size=self.grid_size, stride=self.grid_size)

    def grid_partition(self, input_tensor):
        C, H, W = self.input_shape
        self.batch_size = input_tensor.shape[0]
        assert H % self.grid_size == 0 and W % self.grid_size == 0, "H and W must be divisible by window_size"

        unfolded_tensor = self.unfold(input_tensor)
        #rearranged_tensor = unfolded_tensor.view(B, C, self.grid_size, self.grid_size, -1).permute(0, 4, 1, 2, 3)
        rearranged_tensor = unfolded_tensor.view(self.batch_size, C, (W * H) // (self.grid_size * self.grid_size), self.grid_size, self.grid_size)#.permute(0, 4, 1, 2, 3)

        flattened_tensor = rearranged_tensor.flatten(3, 4)#.permute(0, 2, 1, 3)

        return flattened_tensor

    def grid_reverse(self, windows):
        C, H, W = self.input_shape

        output_tensor = windows.unflatten(3, (self.grid_size, self.grid_size))
        output_tensor = output_tensor.reshape(self.batch_size, C, W, H)
        #output_tensor = windows.permute(0, 2, 1, 3).unflatten(3, (self.grid_size, self.grid_size))
        #output_tensor = output_tensor.permute(0, 2, 3, 4, 1).reshape(B, C, W, H)
        return output_tensor

    def grid_sa(self, x):
        original_x = x

        x = self.layer_norm(x)
        x = self.depth_wise_conv(x)
        x = self.grid_partition(x)
        out, _ = self.self_attention(x)
        x = self.grid_reverse(out)

        return x + original_x

    def gmlp(self, x):
        return self.gmlpBlock(x)

    def forward(self, x):
        x = self.grid_sa(x)
        x = x.permute(0, 2, 3, 1)
        x = self.gmlp(self.layer_norm2(x))

        return x
