import torch
import torch.nn as nn

from Modeling import base


class R2Block(nn.Module):
    def __init__(self, in_channels, out_channels, n=2):
        super(R2Block, self).__init__()
        self.n = n
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.n):
            if i == 0:
                o = self.conv(x)
            o = self.conv(o + x)

        return o


class R2UNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64, n=2):
        super(R2UNet, self).__init__()
        self.block_i = 0
        self.block_list = nn.ModuleList()
        self.register_block_method()

        # down stage 1
        self.register_block(self.conv(self.block_i, in_channels, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels, base_channels, n=n))
        self.register_block(R2Block(base_channels, base_channels, n=n))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 2
        self.register_block(self.conv(self.block_i, base_channels, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*2, base_channels*2, n=n))
        self.register_block(R2Block(base_channels*2, base_channels*2, n=n))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 3
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*4, base_channels*4, n=n))
        self.register_block(R2Block(base_channels*4, base_channels*4, n=n))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 4
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*8, base_channels*8, n=n))
        self.register_block(R2Block(base_channels*8, base_channels*8, n=n))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # middle
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*16, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*16, base_channels*16, n=n))
        self.register_block(R2Block(base_channels*16, base_channels*16, n=n))

        # up stage 4
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*8, base_channels*8, n=n))
        self.register_block(R2Block(base_channels*8, base_channels*8, n=n))

        # up stage 3
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*4, base_channels*4, n=n))
        self.register_block(R2Block(base_channels*4, base_channels*4, n=n))

        # up stage 2
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels*2, base_channels*2, n=n))
        self.register_block(R2Block(base_channels*2, base_channels*2, n=n))
        
        # up stage 1
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(R2Block(base_channels, base_channels, n=n))
        self.register_block(R2Block(base_channels, base_channels, n=n))

        # output layer
        self.register_block(self.conv(self.block_i, base_channels, num_classes, kernel_size=1, stride=1, bn=0, activation=None))

    def register_block(self, block):
        self.block_list.append(block)
        self.block_i += 1

    def register_block_method(self):
        self.conv = base._create_conv_block
        self.maxpool = base._create_maxpool_block
        self.upsample = base._create_upsample_block 

    def forward(self, x):
        o = x
        block_i = 0
        
        # down stage 1
        for i in range(3):
            o = self.block_list[block_i](o)
            block_i += 1
        route_stage_1 = o
        o = self.block_list[block_i](o); block_i += 1
        
        # down stage 2
        for i in range(3):
            o = self.block_list[block_i](o)
            block_i += 1
        route_stage_2 = o
        o = self.block_list[block_i](o); block_i += 1

        # down stage 3
        for i in range(3):
            o = self.block_list[block_i](o)
            block_i += 1
        route_stage_3 = o
        o = self.block_list[block_i](o); block_i += 1

        # down stage 4
        for i in range(3):
            o = self.block_list[block_i](o)
            block_i += 1
        route_stage_4 = o
        o = self.block_list[block_i](o); block_i += 1

        # middle
        for i in range(3):
            o = self.block_list[block_i](o)
            block_i += 1
        
        # up stage 4
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                o = torch.cat([route_stage_4, o], dim=1)

        # up stage 3
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                o = torch.cat([route_stage_3, o], dim=1)
        
        # up stage 2
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                o = torch.cat([route_stage_2, o], dim=1)
        
        # up stage 1
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                o = torch.cat([route_stage_1, o], dim=1)
        
        # output stage
        o = self.block_list[block_i](o); block_i += 1
        o_softmax = nn.Softmax(dim=1)(o)
        
        return o, o_softmax
