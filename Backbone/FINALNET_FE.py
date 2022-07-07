import sys
sys.path.append(r'D:\dl_project\dl_project_cnn\Backbone')
sys.path.append(r'D:\dl_project\dl_project_cnn\Utils')

import torch
import torch.nn as nn

from Utils.modules import BasicPreActBlock
from Utils.modules import Bottleneck
from Utils.modules import convert_to_separable_conv
from Utils.modules import BasicBlock
from Utils.modules import ChannelAttention, SpatialAttention



class rf_attention(nn.Module):
    def __init__(self, kernel_size=7, rf_num = 3):
        super(rf_attention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2,
                               bias=False)

        self.attention = nn.Sequential(
            nn.Conv2d(rf_num, rf_num*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(rf_num*16),
            nn.ReLU(inplace=True),
            nn.Conv2d(rf_num*16, rf_num, kernel_size=3, padding=1),
            # nn.BatchNorm2d(rf_num),
        )

    def spatial_norm(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # x = torch.sigmoid(x)
        return x

    def forward(self, x_list):

        out = torch.cat([self.spatial_norm(x) for x in x_list], dim=1)  # 三个具有代表性的特征图
        out = self.attention(out)  # 计算权重图里每一个像素的softmax，看对于每个像素需要哪个分支
        out = self.softmax(out)

        return out

class inception_res_1(nn.Module):
    def __init__(self, in_channels=128, m_channels=[32, 48, 64], out_channels=128, norm_layer=nn.BatchNorm2d):
        super(inception_res_1, self).__init__()
        self.rf1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )
        self.rf3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )
        self.rf5 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
            norm_layer(m_channels[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[0], out_channels=m_channels[1], kernel_size=3, padding=1),
            norm_layer(m_channels[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=m_channels[1], out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True)
        )

        # self.rf7 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=m_channels[0], kernel_size=1),
        #     norm_layer(m_channels[0]),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(in_channels=m_channels[0], out_channels=m_channels[1], kernel_size=3, padding='same'),
        #     norm_layer(m_channels[1]),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(in_channels=m_channels[1], out_channels=m_channels[2], kernel_size=3, padding='same'),
        #     norm_layer(m_channels[2]),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv2d(in_channels=m_channels[2], out_channels=out_channels, kernel_size=3, padding='same'),
        #     norm_layer(out_channels),
        #     nn.ReLU(inplace=True),
        # )

        self.out = nn.Sequential(
            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),

            
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            norm_layer(out_channels),
        )

        self.ca = ChannelAttention(in_channels=out_channels)

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.rf_attention = rf_attention(rf_num=3)

        # self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        rf1_x = self.rf1(x)
        rf3_x = self.rf3(x)
        rf5_x = self.rf5(x)
        # rf7_x = self.rf7(x)

        out = self.rf_attention([rf1_x, rf3_x, rf5_x])
        rf1_w, rf3_w, rf5_w = torch.split(out, split_size_or_sections=1, dim=1)
        identity_2 = rf1_x * rf1_w + rf3_x * rf3_w + rf5_x * rf5_w+ identity

        # identity_2 = self.relu(self.norm(identity_2) + identity)
        
        # out = rf1_x * rf1_w + rf3_x * rf3_w + rf5_x * rf5_w

        out = self.out(identity_2)

        out = self.ca(out) * out

        return self.relu(out + identity_2)
        # return out + identity


class parralle_downsp(nn.Module):
    def __init__(self, dim):
        super(parralle_downsp, self).__init__()
        self.maxp = nn.MaxPool2d(2)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, stride=2)

    def forward(self, x):
        maxp = self.maxp(x)
        conv = self.conv(x)
        return torch.cat([maxp, conv],dim=1)



class FINALNET_FE(nn.Module):
    def __init__(self, norm_layer=None):
        super(FINALNET_FE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inconv = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            # norm_layer(64),
            # nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            # norm_layer(64),
            # nn.ReLU(inplace=True)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2),
            norm_layer(64),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1),
            # norm_layer(64),
            # nn.ReLU(inplace=True)
            BasicBlock(in_channels=64, out_channels=64, stride=1, norm_layer=norm_layer),
        )  # 128*128*64

        self.stem1 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            # norm_layer(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            parralle_downsp(dim=64)
            # BasicBlock(in_channels=64, out_channels=128, stride=2, norm_layer=norm_layer),
            # nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1),
            # norm_layer(128),
            # nn.ReLU(inplace=True)

            # _pool_conv_cat(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # out 64*64*128
            # norm_layer(128),
            # nn.ReLU(inplace=True),

            # BasicBlock(in_channels=64, out_channels=64, stride=1, norm_layer=norm_layer),
            # nn.Conv2d(192, 128, kernel_size=1),
            # norm_layer(128),
            # nn.ReLU(inplace=True),
        )  # 256*256*64

        self.rf_attention_pooling_1_1 = nn.Sequential(
            inception_res_1(in_channels=128, m_channels=[64, 96, 128], out_channels=128, norm_layer=norm_layer),
            # norm_layer(128),
            # nn.ReLU(inplace=True),

            BasicBlock(in_channels=128, out_channels=128, stride=1, norm_layer=norm_layer),
            # BasicBlock(in_channels=128, out_channels=128, stride=1, norm_layer=norm_layer),
            # BasicBlock(in_channels=128, out_channels=128, stride=1, norm_layer=norm_layer),
            # inception_res_1(in_channels=128, m_channels=[32, 48, 64], out_channels=128, norm_layer=norm_layer),
            # norm_layer(128),
            # nn.ReLU(inplace=True)
        )   # 64*64*128

        # self.pooling_conv_1 = nn.Sequential(
        #     _pool_conv_cat(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # out 64*64*384
        #     norm_layer(128),
        #     nn.ReLU(inplace=True),
        # )
        self.stem2 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            # norm_layer(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            parralle_downsp(dim=128)
            # BasicBlock(in_channels=128, out_channels=256, stride=2, norm_layer=norm_layer),
            # nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=3, padding=1),
            # norm_layer(256),
            # nn.ReLU(inplace=True)
        )  # 128*128*128

        self.rf_attention_pooling_2_1 = nn.Sequential(
            inception_res_1(in_channels=256, m_channels=[128, 192, 256], out_channels=256, norm_layer=norm_layer),
            # norm_layer(256),
            # nn.ReLU(inplace=True),

            BasicBlock(in_channels=256, out_channels=256, stride=1, norm_layer=norm_layer),
        )  # 64*64*256

        self.rf_attention_pooling_2_2 = nn.Sequential(
            inception_res_1(in_channels=256, m_channels=[128, 192, 256], out_channels=256, norm_layer=norm_layer),
            # norm_layer(256),
            # nn.ReLU(inplace=True),

            # inception_res_1(in_channels=256, m_channels=[128, 192, 256], out_channels=256, norm_layer=norm_layer),
            # norm_layer(256),
            # nn.ReLU(inplace=True),
            BasicBlock(in_channels=256, out_channels=256, stride=1, norm_layer=norm_layer),

        )  # 64*64*256

        self.stem3 = nn.Sequential(
            # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
            # norm_layer(64),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            parralle_downsp(dim=256)
            # BasicBlock(in_channels=256, out_channels=512, stride=2, norm_layer=norm_layer),
            # nn.Conv2d(in_channels=256, out_channels=512, stride=2, kernel_size=3, padding=1),
            # norm_layer(512),
            # nn.ReLU(inplace=True)
        )  # 64*64*128


        self.rf_attention_pooling_3_1 = nn.Sequential(
            inception_res_1(in_channels=512, m_channels=[256, 384, 512], out_channels=512, norm_layer=norm_layer),
            # Bottleneck(in_channels=512, out_channels=128, stride=1, norm_layer=norm_layer),
            # norm_layer(512),
            # nn.ReLU(inplace=True),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
            # BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )  # 64*64*512

        self.rf_attention_pooling_3_2 = nn.Sequential(
            inception_res_1(in_channels=512, m_channels=[256, 384, 512], out_channels=512, norm_layer=norm_layer),

            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )  # 64*64*512

        self.rf_attention_pooling_3_3 = nn.Sequential(
            inception_res_1(in_channels=512, m_channels=[256, 384, 512], out_channels=512, norm_layer=norm_layer),


            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )  # 64*64*512

        self.rf_attention_pooling_3_4 = nn.Sequential(
            inception_res_1(in_channels=512, m_channels=[256, 384, 512], out_channels=512, norm_layer=norm_layer),


            BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        )  # 64*64*512

        # self.stem4 = nn.Sequential(
        #     # nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
        #     # norm_layer(64),
        #     # nn.ReLU(inplace=True),
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     # BasicBlock(in_channels=256, out_channels=512, stride=2, norm_layer=norm_layer),
        #     nn.Conv2d(in_channels=512, out_channels=1024, stride=2, kernel_size=2, padding=0),
        #     # norm_layer(512),
        #     # nn.ReLU(inplace=True)

        #     # _pool_conv_cat(in_channels=64, out_channels=128, kernel_size=3, padding=1),  # out 64*64*128
        #     # norm_layer(128),
        #     # nn.ReLU(inplace=True),

        #     # BasicBlock(in_channels=64, out_channels=64, stride=1, norm_layer=norm_layer),
        #     # nn.Conv2d(192, 128, kernel_size=1),
        #     # norm_layer(128),
        #     # nn.ReLU(inplace=True),
        # )  # 64*64*128

        # self.rf_attention_pooling_4_1 = nn.Sequential(
        #     inception_res_1(in_channels=1024, m_channels=[512, 768, 1024], out_channels=1024, norm_layer=norm_layer),
        #     # Bottleneck(in_channels=512, out_channels=128, stride=1, norm_layer=norm_layer),
        #     norm_layer(1024),
        #     nn.ReLU(inplace=True),

        #     BasicBlock(in_channels=1024, out_channels=1024, stride=1, norm_layer=norm_layer),
        #     # BasicBlock(in_channels=512, out_channels=512, stride=1, norm_layer=norm_layer),
        # )  # 64*64*512


        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out = self.inconv(input)  # 512*512

        out1 = self.stem1(out)  # 64*64

        out = self.rf_attention_pooling_1_1(out1) # 256*256

        out = self.stem2(out)

        out = self.rf_attention_pooling_2_1(out)

        out = self.rf_attention_pooling_2_2(out)# 128*128

        out = self.stem3(out)

        out = self.rf_attention_pooling_3_1(out)

        out = self.rf_attention_pooling_3_2(out)

        out = self.rf_attention_pooling_3_3(out)

        out = self.rf_attention_pooling_3_4(out)# 64*64

        return out

if __name__ == '__main__':
    x = FirstStagenet_FE('GN')