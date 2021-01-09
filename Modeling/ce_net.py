import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from Modeling import base


class DACBlock(nn.Module):
    def __init__(self, in_channels):
        super(DACBlock, self).__init__()
        self.path_1 = nn.Conv2d(in_channels, in_channels, 3, 1, (3-1)//2, 1)
        self.path_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, (3+3+1-1) // 2, 3),
            nn.Conv2d(in_channels, in_channels, 1, 1, (1-1) // 2, 1)
        )
        self.path_3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, (3-1) // 2, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, (3+3+1-1) // 2, 3),
            nn.Conv2d(in_channels, in_channels, 1, 1, (1-1) // 2, 1)
        )
        self.path_4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, (3-1) // 2, 1),
            nn.Conv2d(in_channels, in_channels, 3, 1, (3+3+1-1) // 2, 3),
            nn.Conv2d(in_channels, in_channels, 3, 1, (5+5+1-1) // 2, 5),
            nn.Conv2d(in_channels, in_channels, 1, 1, (1-1) // 2, 1)
        )
    
    def forward(self, x):
        route_1 = self.path_1(x)
        route_2 = self.path_2(x)
        route_3 = self.path_3(x)
        route_4 = self.path_4(x)

        result = route_1 + route_2 + route_3 + route_4
        # result = torch.cat([route_1, route_2, route_3, route_4], dim=1)

        return result


class RMPBlock(nn.Module):
    def __init__(self, in_channels):
        super(RMPBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=6, stride=6)

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]

        sub_1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode="bilinear")
        sub_2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode="bilinear")
        sub_3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode="bilinear")
        sub_4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode="bilinear")

        out = torch.cat([sub_1, sub_2, sub_3, sub_4, x], dim=1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        return x


class CENet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(CENet, self).__init__()
        self.block_i = 0
        self.block_list = nn.ModuleList()

        resnet = models.resnet34(pretrained=True)

        self.register_block(nn.Conv2d(in_channels, 3, kernel_size=1, stride=1))
        self.register_block(resnet.conv1)
        self.register_block(resnet.bn1)
        self.register_block(resnet.relu)
        self.register_block(resnet.maxpool)

        self.register_block(resnet.layer1)
        self.register_block(resnet.layer2)
        self.register_block(resnet.layer3)
        self.register_block(resnet.layer4)

        self.register_block(DACBlock(base_channels*8))
        self.register_block(RMPBlock(base_channels*8))

        self.register_block(DecoderBlock(base_channels*8+4, base_channels*4))
        self.register_block(DecoderBlock(base_channels*4, base_channels*2))
        self.register_block(DecoderBlock(base_channels*2, base_channels))
        self.register_block(DecoderBlock(base_channels, base_channels))

        self.register_block(nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1))
        self.register_block(nn.ReLU(inplace=True))
        self.register_block(nn.Conv2d(base_channels // 2, base_channels // 2, kernel_size=3, padding=1))
        self.register_block(nn.ReLU(inplace=True))
        self.register_block(nn.Conv2d(base_channels // 2, num_classes, kernel_size=3, padding=1))

    def register_block(self, block):
        self.block_list.append(block)
        self.block_i += 1

    def forward(self, x):
        o = x
        block_i = 0
        
        # Preprocess
        for i in range(5):
            o = self.block_list[block_i](o)
            block_i += 1
    
        # Encoder
        e1 = self.block_list[block_i](o); block_i += 1
        e2 = self.block_list[block_i](e1); block_i += 1
        e3 = self.block_list[block_i](e2); block_i += 1
        e4 = self.block_list[block_i](e3); block_i += 1

        # Middle
        for i in range(2):
            e4 = self.block_list[block_i](e4)
            block_i += 1
        
        # Decoder
        d4 = self.block_list[block_i](e4) + e3; block_i += 1
        d3 = self.block_list[block_i](d4) + e2; block_i += 1
        d2 = self.block_list[block_i](d3) + e1; block_i += 1
        d1 = self.block_list[block_i](d2); block_i += 1
        
        # Output 
        for i in range(5):
            o = self.block_list[block_i](o)
            block_i += 1
        o = F.interpolate(o, scale_factor=2, mode="bilinear")
        o_softmax = nn.Softmax(dim=1)(o)

        return o, o_softmax
