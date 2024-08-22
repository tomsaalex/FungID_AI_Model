import math

import torch
from torch import nn
from torch.nn import Dropout, Identity

from model.m_vit_model.aspp_second_impl import _ASPP
from model.m_vit_model.block_attention import BlockAttention
from model.m_vit_model.grid_attention import GridAttention
from model.m_vit_model.mv2_block import MV2_Block
from timm import is_exportable
from timm.layers import make_divisible, to_2tuple, ConvNormAct, ClassifierHead, SelectAdaptivePool2d, Linear, GroupNorm1
from timm.models.byobnet import LayerFn, BottleneckBlock, num_groups
from timm.models.mobilevit import MobileVitBlock, LinearTransformerBlock
from timm.models.vision_transformer import Block as TransformerBlock


class MVitClassifier2(nn.Module):
    def __init__(self, in_channels, num_classes, batch_size):
        super(MVitClassifier2, self).__init__()

        self.conv1 = ConvNormAct(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False,
                                 act_layer=nn.SiLU)  # Conv 3 x 3

        self.MV2_Block1 = MV2_Block(in_channels=64, mid_channels=128, out_channels=128, stride=1)
        self.MV2_Block2 = MV2_Block(in_channels=128, mid_channels=256, out_channels=256, stride=2)
        self.MV2_Block3 = MV2_Block(in_channels=256, mid_channels=512, out_channels=256, stride=1)
        self.MV2_Block4 = MV2_Block(in_channels=256, mid_channels=512, out_channels=512, stride=2)

        self.MViT2Block1 = MVit2Block(
            input_shape=(batch_size, 512, 28, 28),
            transformer_dim=256,
            patch_size=7,
            transformer_depth=2,
            groups=512
        )

        self.MV2_Block5 = MV2_Block(in_channels=512, mid_channels=1024, out_channels=768, stride=2)
        self.MViT2Block2 = MVit2Block(
            input_shape=(batch_size, 768, 14, 14),
            transformer_dim=384,
            patch_size=7,
            transformer_depth=4,
            groups=768
        )

        self.MV2_Block6 = MV2_Block(in_channels=768, mid_channels=1536, out_channels=1024, stride=2)
        self.MViT2Block3 = MVit2Block(
            input_shape=(batch_size, 1024, 7, 7),
            transformer_dim=512,
            patch_size=7,
            transformer_depth=3,
        )

        self.aspp = _ASPP(1024, 1024, [6, 12, 18])

        self.final_conv = Identity()
        self.global_pool = SelectAdaptivePool2d(
            pool_type='avg',
            flatten=True
        )
        self.dropout = Dropout(p=0.0, inplace=False)
        self.fc = Linear(in_features=1024, out_features=num_classes, bias=True)

    def forward(self, x):
        x = self.conv1(x)

        x = self.MV2_Block1(x)
        x = self.MV2_Block2(x)
        x = self.MV2_Block3(x)
        x = self.MV2_Block4(x)

        self.MViT2Block1(x)
        x = self.MV2_Block5(x)

        self.MViT2Block2(x)
        x = self.MV2_Block6(x)

        self.MViT2Block3(x)

        x = self.aspp(x)
        #x = self.final_conv(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class MVit2Block(nn.Module):

    def __init__(
            self,
            input_shape=(1, 3, 224, 224),
            out_channels: int = None,
            kernel_size=3,
            stride=1,
            bottle_ratio=1.0,
            group_size=1,
            dilation=(1, 1),
            mlp_ratio: float = 2.0,
            transformer_dim=None,
            transformer_depth=2,
            attn_drop=0.0,
            drop=0,
            no_fusion=False,
            num_heads=4,
            transformer_norm_layer=GroupNorm1,
            drop_path_rate=0.0,
            layers=None,
            patch_size=7,
            grid_size=7,
            block_size=7,
            groups=1,
    ):
        # Global attention branch
        super(MVit2Block, self).__init__()

        self.input_shape = input_shape
        _, in_channels, height, width = input_shape

        layers = layers or LayerFn()
        out_channels = out_channels or in_channels
        transformer_dim = transformer_dim or make_divisible(bottle_ratio * in_channels)

        self.conv_nxn_global = layers.conv_norm_act(
            in_channels, in_channels, kernel_size, stride, groups, dilation[0]
        )
        self.conv_1x1_global = nn.Conv2d(in_channels, transformer_dim, kernel_size=1, bias=False)

        self.transformer = nn.Sequential(
            *[
                LinearTransformerBlock(
                    transformer_dim,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path_rate,
                    act_layer=layers.act,
                    norm_layer=transformer_norm_layer
                )
                for _ in range(transformer_depth)
            ]
        )

        self.norm = transformer_norm_layer(transformer_dim)
        self.conv_proj = layers.conv_norm_act(transformer_dim, out_channels, kernel_size=1, stride=1, apply_act=False)

        self.patch_size = to_2tuple(patch_size)
        self.patch_area = self.patch_size[0] * self.patch_size[1]
        self.coreml_exportable = is_exportable()

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
