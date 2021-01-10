import torch
import torch.nn as nn

from Modeling import base


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, kernel_size=3, stride=1):
        super(ConvUnit, self).__init__()
        self.block_list = nn.ModuleList()
        self.conv = base._create_conv_block

        for i in range(n):
            op = self.conv(i, in_channels, out_channels, kernel_size=kernel_size, stride=stride, bn=1)
            in_channels = out_channels
            self.block_list.append(op)

    def forward(self, x):
        for i in range(len(self.block_list)):
            x = self.block_list[i](x)

        return x 


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(UNetPlusPlus, self).__init__()
        self.block_i = 0
        
        # Downsampling
        self.op0_0 = ConvUnit(in_channels, base_channels)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.op1_0 = ConvUnit(base_channels, base_channels*2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.op2_0 = ConvUnit(base_channels*2, base_channels*4)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.op3_0 = ConvUnit(base_channels*4, base_channels*8)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.op4_0 = ConvUnit(base_channels*8, base_channels*16)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.op0_1 = ConvUnit(base_channels+base_channels*2, base_channels)
        self.upsample0_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op1_1 = ConvUnit(base_channels*2+base_channels*4, base_channels*2)
        self.upsample1_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op2_1 = ConvUnit(base_channels*4+base_channels*8, base_channels*4)
        self.upsample2_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op3_1 = ConvUnit(base_channels*8+base_channels*16, base_channels*8)
        self.upsample3_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.op0_2 = ConvUnit(base_channels*2+base_channels*2, base_channels)
        self.upsample0_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op1_2 = ConvUnit(base_channels*2*2+base_channels*4, base_channels*2)
        self.upsample1_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op2_2 = ConvUnit(base_channels*4*2+base_channels*8, base_channels*4)
        self.upsample2_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.op0_3 = ConvUnit(base_channels*3+base_channels*2, base_channels)
        self.upsample0_3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.op1_3 = ConvUnit(base_channels*2*3+base_channels*4, base_channels*2)
        self.upsample1_3 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.op0_4 = ConvUnit(base_channels*4+base_channels*2, base_channels)
        self.upsample0_4 = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Multi-task
        self.out_1 = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.out_2 = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.out_3 = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1, padding=0)
        self.out_4 = nn.Conv2d(base_channels, num_classes, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                base.init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                base.init_weights(m, init_type='kaiming')

    def forward(self, x):
        # Encoder
        x0_0 = self.op0_0(x)
        down_x0_0 = self.maxpool0(x0_0)
        x1_0 = self.op1_0(down_x0_0)
        down_x1_0 = self.maxpool1(x1_0)
        x2_0 = self.op2_0(down_x1_0)
        down_x2_0 = self.maxpool2(x2_0)
        x3_0 = self.op3_0(down_x2_0)
        down_x3_0 = self.maxpool3(x3_0)
        x4_0 = self.op4_0(down_x3_0)
        down_x4_0 = self.maxpool4(x4_0)

        # Decoder
        up_x1_0 = self.upsample0_1(x1_0)
        x0_1 = self.op0_1(torch.cat([x0_0, up_x1_0], dim=1))
        up_x2_0 = self.upsample1_1(x2_0)
        x1_1 = self.op1_1(torch.cat([x1_0, up_x2_0], dim=1))
        up_x3_0 = self.upsample2_1(x3_0)
        x2_1 = self.op2_1(torch.cat([x2_0, up_x3_0], dim=1))
        up_x4_0 = self.upsample3_1(x4_0)
        x3_1 = self.op3_1(torch.cat([x3_0, up_x4_0], dim=1))
        
        up_x1_1 = self.upsample0_2(x1_1)
        x0_2 = self.op0_2(torch.cat([x0_0, x0_1, up_x1_1], dim=1))
        up_x2_1 = self.upsample1_2(x2_1)
        x1_2 = self.op1_2(torch.cat([x1_0, x1_1, up_x2_1], dim=1))
        up_x3_1 = self.upsample2_2(x3_1)
        x2_2 = self.op2_2(torch.cat([x2_0, x2_1, up_x3_1], dim=1))
        
        up_x1_2 = self.upsample0_3(x1_2)
        x0_3 = self.op0_3(torch.cat([x0_0, x0_1, x0_2, up_x1_2], dim=1))
        up_x2_2 = self.upsample1_3(x2_2)
        x1_3 = self.op1_3(torch.cat([x1_0, x1_1, x1_2, up_x2_2], dim=1))
        
        up_x1_3 = self.upsample0_4(x1_3)
        x0_4 = self.op0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up_x1_3], dim=1))

        # Multi-task
        x0_1 = self.out_1(x0_1)
        x0_2 = self.out_2(x0_2)
        x0_3 = self.out_3(x0_3)
        x0_4 = self.out_4(x0_4)


        o = (x0_1 + x0_2 + x0_3 + x0_4) / 4
        o_softmax = nn.Softmax(dim=1)(o)
        
        return o, o_softmax



