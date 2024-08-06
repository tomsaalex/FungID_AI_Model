import torch
from torch import nn

from m_vit.g_mlp_pytorch import gMLPBlock, gMLPForImageClassification
from m_vit.self_attention import Self_Attn


class BlockAttention(nn.Module):

    def __init__(self, channels, width, height, window_size):
        super(BlockAttention, self).__init__()

        self.batch_size = None
        self.input_shape = (channels, width, height)
        self.window_size = window_size
        self.layer_norm = nn.LayerNorm((channels, width, height))
        self.layer_norm2 = nn.LayerNorm((width, height, channels))
        self.depth_wise_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,
                                         groups=channels)  # TODO assumed size in (I think) absence of data
        self.self_attention = Self_Attn(channels)

        self.gmlpBlock = gMLPForImageClassification(
            image_size=width,
            patch_size=window_size,
            in_channels=channels,
            d_model=channels,
            d_ffn=channels * 4,
            seq_len=(width * height),# // (window_size * window_size),
            num_layers=1
        )

        self.unfold = torch.nn.Unfold(kernel_size=self.window_size, stride=self.window_size)
        self.fold = torch.nn.Fold(output_size=(width, height), kernel_size=self.window_size, stride=self.window_size)

    def split_into_windows(self, input_tensor):
        """
        Split an input tensor into non-overlapping windows.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (B, C, H, W)
            window_size (int): Size of the window to split the input into

        Returns:
            torch.Tensor: Tensor of windows of shape (B * num_windows, C, window_size, window_size)
        """
        C, H, W = self.input_shape
        self.batch_size = input_tensor.shape[0]
        assert H % self.window_size == 0 and W % self.window_size == 0, "H and W must be divisible by window_size"

        unfolded_tensor = self.unfold(input_tensor)
        rearranged_tensor = unfolded_tensor.view(self.batch_size, C, self.window_size, self.window_size, -1).permute(0, 4, 1, 2, 3)
        flattened_tensor = rearranged_tensor.flatten(3, 4).permute(0, 2, 1, 3)

        return flattened_tensor

    def combine_windows(self, windows):
        C, H, W = self.input_shape

        output_tensor = windows.permute(0, 2, 1, 3).unflatten(3, (self.window_size, self.window_size))
        output_tensor = output_tensor.permute(0, 2, 3, 4, 1).reshape(self.batch_size, C, W, H)
        return output_tensor

    def block_sa(self, x):
        original_x = x

        x = self.layer_norm(x)
        x = self.depth_wise_conv(x)
        x = self.split_into_windows(x)
        out, _ = self.self_attention(x)
        x = self.combine_windows(out)

        return x + original_x

    def gmlp(self, x):
        return self.gmlpBlock(x)

    def forward(self, x):
        x = self.block_sa(x)
        x = x.permute(0, 2, 3, 1)
        x = self.gmlp(self.layer_norm2(x))

        return x
