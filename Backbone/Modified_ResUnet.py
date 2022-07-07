import sys
sys.path.append(r'D:\dl_project\dl_project_cnn\Backbone')
sys.path.append(r'D:\dl_project\dl_project_cnn\Utils')

import torch
import torch.nn as nn
from Utils.modules import BasicPreActBlock
from Utils.modules import crop_patchs
from Utils.modules import merge_patchs

class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(InputBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):

        out = self.input_layer(x) + self.input_skip(x)
        return out


class feature_extractor(nn.Module):
    def __init__(self, in_channels, filters=[64, 128, 256, 512], norm_layer=None):
        super(feature_extractor, self).__init__()

        self.Input_conv = InputBlock(in_channels=in_channels, out_channels=filters[0], norm_layer=norm_layer)  #512

        self.crop_1 = crop_patchs(cropsize=256, figsize=512)

        self.residual_conv_1 = BasicPreActBlock(filters[0], filters[1], stride=1, norm_layer=norm_layer)

        self.crop_2 = crop_patchs(cropsize=128, figsize=256)

        self.residual_conv_2 = BasicPreActBlock(filters[1], filters[2], stride=1, norm_layer=norm_layer)

        self.crop_bridge = crop_patchs(cropsize=64, figsize=128)

        self.bridge = BasicPreActBlock(filters[2], filters[3], stride=1, norm_layer=norm_layer)

    def forward(self, x):

        x1 = self.Input_conv(x)

        x2 = self.residual_conv_1(x1)

        x2 = self.crop_1(x2)

        x3 = self.residual_conv_2(x2)

        x3 = self.crop_2(x3)

        x4 = self.bridge(x3)

        x4 = self.crop_bridge(x4)

        return x4

