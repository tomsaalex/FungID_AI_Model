import math

import torch
from torch import nn
from torch.nn import Dropout

from model.m_vit_model.aspp_second_impl import _ASPP
from model.m_vit_model.block_attention import BlockAttention
from model.m_vit_model.grid_attention import GridAttention
from model.m_vit_model.mv2_block import MV2_Block
from timm.layers import make_divisible, to_2tuple, ConvNormAct, ClassifierHead, SelectAdaptivePool2d, Linear
from timm.models.byobnet import LayerFn, BottleneckBlock, num_groups
from timm.models.mobilevit import MobileVitBlock
from timm.models.vision_transformer import Block as TransformerBlock


class MVitClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, batch_size):
        super(MVitClassifier, self).__init__()
        #self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False) # Conv 3 x 3
        self.conv1 = ConvNormAct(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False,
                                 act_layer=nn.SiLU)  # Conv 3 x 3

        self.MV2_Block1 = MV2_Block(in_channels=16, mid_channels=64, out_channels=32, stride=1)
        self.MV2_Block2 = MV2_Block(in_channels=32, mid_channels=128, out_channels=64, stride=2)
        self.MV2_Block3 = MV2_Block(in_channels=64, mid_channels=256, out_channels=64, stride=1)
        self.MV2_Block4 = MV2_Block(in_channels=64, mid_channels=256, out_channels=64, stride=1)

        self.MV2_Block5 = MV2_Block(in_channels=64, mid_channels=256, out_channels=96, stride=2)
        # self.MobileViTBlock = MobileVitBlock(
        #     in_chs=96,
        #     transformer_dim=144,
        #     patch_size=7,
        #     transformer_depth=2,
        # )
        self.MViTBlock1 = MVitBlock(
            input_shape=(batch_size, 96, 28, 28),
            transformer_dim=144,
            patch_size=7,
            transformer_depth=2
        )

        self.MV2_Block6 = MV2_Block(in_channels=96, mid_channels=384, out_channels=128, stride=2)
        # self.MobileViTBlock2 = MobileVitBlock(
        #     in_chs=128,
        #     transformer_dim=192,
        #     patch_size=7,
        #     transformer_depth=4,
        # )
        self.MViTBlock2 = MVitBlock(
            input_shape=(batch_size, 128, 14, 14),
            transformer_dim=192,
            patch_size=7,
            transformer_depth=4
        )

        self.MV2_Block7 = MV2_Block(in_channels=128, mid_channels=512, out_channels=160, stride=2)
        # self.MobileViTBlock3 = MobileVitBlock(
        #     in_chs=160,
        #     transformer_dim=240,
        #     patch_size=7,
        #     transformer_depth=3,
        # )
        self.MViTBlock3 = MVitBlock(
            input_shape=(batch_size, 160, 7, 7),
            transformer_dim=240,
            patch_size=7,
            transformer_depth=3,
        )

        self.aspp = _ASPP(160, 160, [6, 12, 18])

        self.final_conv = ConvNormAct(in_channels=160, out_channels=640, kernel_size=1, stride=1, padding=0, bias=False,
                                      act_layer=nn.SiLU)
        self.global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True
        )
        self.dropout = Dropout(p=0.0, inplace=False)
        self.fc = Linear(in_features=640, out_features=num_classes, bias=True)

        #self.finalConv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        #self.globalPool = nn.AvgPool2d(kernel_size=7)

    def forward(self, x):
        #x = x.permute(1, 0, 2, 3)
        x = self.conv1(x)

        x = self.MV2_Block1(x)
        x = self.MV2_Block2(x)
        x = self.MV2_Block3(x)
        x = self.MV2_Block4(x)

        x = self.MV2_Block5(x)
        self.MViTBlock1(x)
        x = self.MV2_Block6(x)
        self.MViTBlock2(x)

        x = self.MV2_Block7(x)
        self.MViTBlock3(x)

        x = self.aspp(x)
        x = self.final_conv(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


# The MVitBlock module from the MVit network. Consider the input to have height H, width W and C channels.
class MVitBlock(nn.Module):

    def __init__(
            self,
            input_shape=(1, 3, 224, 224),
            out_channels: int = None,
            kernel_size=3,
            stride=1,
            bottle_ratio=1.0,
            group_size=None,
            dilation=(1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim=None,
            transformer_depth=2,
            attn_drop=0.0,
            drop=0,
            no_fusion=False,
            num_heads=4,
            transformer_norm_layer=nn.LayerNorm,
            drop_path_rate=0.0,
            layers=None,
            patch_size=7,
            grid_size=7,
            block_size=7
    ):
        # Global attention branch
        super(MVitBlock, self).__init__()

        self.input_shape = input_shape
        _, in_channels, height, width = input_shape

        layers = layers or LayerFn()
        groups = num_groups(group_size, in_channels)
        out_channels = out_channels or in_channels
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_channels)

        self.conv_nxn_global = layers.conv_norm_act(
            in_channels, in_channels, kernel_size, stride, groups, dilation[0]
        )
        self.conv_1x1_global = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(
            *[
                TransformerBlock(
                    transformer_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    attn_drop=attn_drop,
                    proj_drop=drop,
                    drop_path=drop_path_rate,
                    act_layer=layers.act,  #TODO check if this is correct
                    norm_layer=transformer_norm_layer
                )
                for _ in range(transformer_depth)
            ]
        )

        self.norm = transformer_norm_layer(transformer_dim)
        self.conv_proj = layers.conv_norm_act(transformer_dim, out_channels, kernel_size=1, stride=1)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]

        # MDA branch
        self.conv_1x1_mda = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)

        self.final_conv_1x1_MDA = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.block_attention = BlockAttention(in_channels, width, height, block_size)
        self.grid_attention = GridAttention(in_channels, width, height, grid_size)

        # Fusion
        self.conv_fusion = layers.conv_norm_act(3 * in_channels, in_channels, kernel_size=kernel_size, stride=1)

    def unfold(self, x):
        # Unfold (feature map -> patches)
        patch_h, patch_w = self.patch_size
        B, C, H, W = x.shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, n_w, p_h, p_w]
        x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] --> [BP, N, C] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)

        return x, (B, C, H, W)

    def fold(self, x, req_shape):
        patch_h, patch_w = self.patch_size
        B, C, H, W = req_shape
        new_h, new_w = math.ceil(H / patch_h) * patch_h, math.ceil(W / patch_w) * patch_w
        num_patch_h, num_patch_w = new_h // patch_h, new_w // patch_w  # n_h, n_w
        num_patches = num_patch_h * num_patch_w  # N

        # Fold (patch -> feature map)
        # [B, P, N, C] --> [B*C*n_h, n_w, p_h, p_w]
        x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)

        return x

    def forward(self, x):
        original_x = x.clone()

        # MDA Branch
        mda_x = self.conv_1x1_mda(x)
        mda_x = self.block_attention(mda_x)
        mda_x = self.grid_attention(mda_x)
        #TODO: mda_x might need a conv proj instead of final_conv_1x1_MDA, not sure tho
        mda_x = self.final_conv_1x1_MDA(mda_x)

        # Global attention branch
        global_x = self.conv_nxn_global(x)
        global_x = self.conv_1x1_global(global_x)

        global_x, req_shape = self.unfold(global_x)

        global_x = self.transformer(global_x)
        global_x = self.norm(global_x)

        global_x = self.fold(global_x, req_shape)
        global_x = self.conv_proj(global_x)

        # Fusion

        final_x = self.conv_fusion(torch.cat([original_x, mda_x, global_x], dim=1))
        #final_x = self.conv_fusion(torch.cat([original_x, global_x], dim=1))
        #final_x = global_x
        return final_x
