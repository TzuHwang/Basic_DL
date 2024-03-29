import torch
import torch.nn as nn

from .components import *

"""
ref: https://github.com/milesial/Pytorch-UNet
"""

class UNet(nn.Module):
    def __init__(self, input_channels, output_channel_num, bilinear=False, conv_bn=True):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channel_num = output_channel_num
        self.bilinear = bilinear

        self.inc = (DoubleConv(input_channels, 64, conv_bn=conv_bn))
        self.down1 = (Down(64, 128, conv_bn=conv_bn))
        self.down2 = (Down(128, 256, conv_bn=conv_bn))
        self.down3 = (Down(256, 512, conv_bn=conv_bn))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, conv_bn=conv_bn))
        self.up1 = (Up(1024, 512 // factor, bilinear, conv_bn=conv_bn))
        self.up2 = (Up(512, 256 // factor, bilinear, conv_bn=conv_bn))
        self.up3 = (Up(256, 128 // factor, bilinear, conv_bn=conv_bn))
        self.up4 = (Up(128, 64, bilinear, conv_bn=conv_bn))
        self.outc = (OutConv(64, output_channel_num))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits