import torch
import torch.nn as nn

from Modeling import base


class DownStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownStage, self).__init__()
        self.conv = base._create_conv_block
        self.main_block = nn.Sequential(
            self.conv(0, in_channels,  out_channels, kernel_size=1, stride=1, bn=1),
            self.conv(1, out_channels, out_channels, kernel_size=3, stride=1, bn=1),
            self.conv(2, out_channels, out_channels, kernel_size=3, stride=1, bn=1)
        )

    def forward(self, x):
        o = self.main_block(x)

        return o
    

class MiddleStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MiddleStage, self).__init__()
        self.conv = base._create_conv_block
        self.main_block = nn.Sequential(
            self.conv(0, in_channels,  out_channels, kernel_size=1, stride=1, bn=1),
            self.conv(1, out_channels, out_channels, kernel_size=3, stride=1, bn=1),
            self.conv(2, out_channels, out_channels, kernel_size=3, stride=1, bn=1)
        )

    def forward(self, x):
        o = self.main_block(x)

        return o


class UpStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpStage, self).__init__()
        self.conv = base._create_conv_block
        self.reduce_channel = self.conv(0, in_channels, out_channels, kernel_size=1, stride=1, bn=1)
        self.integration    = self.conv(1, in_channels, out_channels, kernel_size=1, stride=1, bn=1)
        self.main_block = nn.Sequential(
            self.conv(2, out_channels, out_channels, kernel_size=3, stride=1, bn=1),
            self.conv(3, out_channels, out_channels, kernel_size=3, stride=1, bn=1),
        )

    def forward(self, x, route):
        o = self.reduce_channel(x)
        o = torch.cat([o, route], dim=1)
        o = self.integration(o)

        o = self.main_block(o)

        return o


class WNetFirstStage(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(WNetFirstStage, self).__init__()
        self.block_i = 0
        self.block_list = nn.ModuleList()
        self.register_block_method()

        # First U down stage 1
        self.register_block(DownStage(in_channels, base_channels))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # First U down stage 2
        self.register_block(DownStage(base_channels, base_channels*2))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # First U down stage 3
        self.register_block(DownStage(base_channels*2, base_channels*4))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # First U down stage 4
        self.register_block(DownStage(base_channels*4, base_channels*8))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # First U middle
        self.register_block(MiddleStage(base_channels*8, base_channels*16))
        # First U up stage 4
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*16, base_channels*8))
        # First U up stage 3
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*8, base_channels*4))
        # First U up stage 2
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*4, base_channels*2))
        # First U up stage 1
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*2, base_channels))
        
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
        o = self.block_list[block_i](o); block_i += 1
        route_stage_1 = o
        o = self.block_list[block_i](o); block_i += 1
        
        # down stage 2
        o = self.block_list[block_i](o); block_i += 1
        route_stage_2 = o
        o = self.block_list[block_i](o); block_i += 1

        # down stage 3
        o = self.block_list[block_i](o); block_i += 1
        route_stage_3 = o
        o = self.block_list[block_i](o); block_i += 1

        # down stage 4
        o = self.block_list[block_i](o); block_i += 1
        route_stage_4 = o
        o = self.block_list[block_i](o); block_i += 1

        # middle
        o = self.block_list[block_i](o); block_i += 1
        out_bottom = o
        
        # up stage 4
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_4); block_i += 1
        out_4 = o
        

        # up stage 3
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_3); block_i += 1
        out_3 = o
        
        # up stage 2
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_2); block_i += 1
        out_2 = o
        
        # up stage 1
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_1); block_i += 1
        out_1 = o
        
        return out_1, out_2, out_3, out_4, out_bottom


class WNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(WNet, self).__init__()
        self.block_i = 0
        self.block_list = nn.ModuleList()
        self.register_block_method()

        # First U
        self.register_block(WNetFirstStage(in_channels, base_channels))

        # Second U down stage 1
        self.register_block(DownStage(base_channels, base_channels))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # Second U down stage 2
        self.register_block(self.conv(self.block_i, base_channels, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(DownStage(base_channels*2*2, base_channels*2))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # Second U down stage 3
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(DownStage(base_channels*4*2, base_channels*4))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # Second U down stage 4
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(DownStage(base_channels*8*2, base_channels*8))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))
        # Second U middle
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*16, kernel_size=1, stride=1, bn=1))
        self.register_block(MiddleStage(base_channels*16*2, base_channels*16))
        # Second U up stage 4
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*16, base_channels*8))
        # Second U up stage 3
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*8, base_channels*4))
        # Second U up stage 2
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*4, base_channels*2))
        # Second U up stage 1
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(UpStage(base_channels*2, base_channels))
        
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
        
        # First U
        out_1, out_2, out_3, out_4, out_bottom = self.block_list[block_i](x); block_i += 1

        # Second U
        ## down stage 1
        o = self.block_list[block_i](out_1); block_i += 1
        route_stage_1 = o
        o = self.block_list[block_i](o); block_i += 1
        
        ## down stage 2
        o = self.block_list[block_i](o); block_i += 1
        o = torch.cat([o, out_2], dim=1)
        o = self.block_list[block_i](o); block_i += 1
        route_stage_2 = o
        o = self.block_list[block_i](o); block_i += 1

        ## down stage 3
        o = self.block_list[block_i](o); block_i += 1
        o = torch.cat([o, out_3], dim=1)
        o = self.block_list[block_i](o); block_i += 1
        route_stage_3 = o
        o = self.block_list[block_i](o); block_i += 1

        ## down stage 4
        o = self.block_list[block_i](o); block_i += 1
        o = torch.cat([o, out_4], dim=1)
        o = self.block_list[block_i](o); block_i += 1
        route_stage_4 = o
        o = self.block_list[block_i](o); block_i += 1

        ## middle
        o = self.block_list[block_i](o); block_i += 1
        o = torch.cat([o, out_bottom], dim=1)
        o = self.block_list[block_i](o); block_i += 1
        
        ## up stage 4
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_4); block_i += 1

        ## up stage 3
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_3); block_i += 1
        
        ## up stage 2
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_2); block_i += 1
        
        ## up stage 1
        o = self.block_list[block_i](o); block_i += 1
        o = self.block_list[block_i](o, route_stage_1); block_i += 1
        
        ## out layer
        o = self.block_list[block_i](o); block_i += 1
        o_softmax = nn.Softmax(dim=1)(o)

        return o, o_softmax
