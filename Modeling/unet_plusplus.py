import torch
import torch.nn as nn


class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, kernel_size=3, stride=1, padding=1):
        super(ConvUnit, self).__init__()
        self.block_list = nn.ModuleList()

        for i in range(n):
            conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True))
            in_channels = out_channels
            self.block_list.append(conv)

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
        # self.op3_1 = ConvUnit(base_channels)


    def forward(self, x):
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

        print(x0_0.shape, x1_0.shape)
        

        # return o, o_softmax


if __name__ == "__main__":
    data = torch.randn((1, 1, 224, 224)).cuda()
    model = UNetPlusPlus(1, 3).cuda()

    with torch.no_grad():
        print(model(data))

