from torch import nn
from torchvision.ops import SqueezeExcitation

from timm.layers import ConvNormAct


# The MV2 module from the MobileNetV2 network.
# The structure is
# 1 x 1 Conv to increase channels -> DepthwiseConv 3x3 to decrease channels -> Squeeze and Excitation -> 1 x 1Conv to recover dimensionality
class MV2_Block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride):
        super(MV2_Block, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1)
        self.depth_wise_conv = ConvNormAct(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False, act_layer=nn.SiLU)

        self.se = SqueezeExcitation(mid_channels, out_channels) #TODO: Enable when finalizing model (it had 64, 16, not sure about now)
        self.conv2 = ConvNormAct(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, act_layer=nn.SiLU)

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.depth_wise_conv(x)
        x = self.se(x) #TODO: Reenable this when finalizing
        x = self.conv2(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x += identity

        return x
