import torch
import torch.nn as nn

from Modeling import base


class AttentionGate(nn.Module):
    def __init__(self, gate_channel, signal_channel, middle_channel=None):
        super(AttentionGate, self).__init__()
        self.conv = base._create_conv_block
        self.projection_gate = self.conv(0, gate_channel, middle_channel, 1, 1, True)
        self.projection_signal = self.conv(1, signal_channel, middle_channel, 1, 1, True)
        self.projection_mask = self.conv(2, middle_channel, 1, 1, 1, True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, gate, signal):
        raw_signal = signal
        
        gate = self.projection_gate(gate)
        signal = self.projection_signal(signal)

        mask = gate + signal
        mask = self.relu(mask)
        mask = self.projection_mask(mask)
        mask = self.sigmoid(mask)

        signal = mask * raw_signal

        return signal


class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes, base_channels=64):
        super(AttentionUNet, self).__init__()
        self.block_i = 0
        self.block_list = nn.ModuleList()
        self.register_block_method()

        # down stage 1
        self.register_block(self.conv(self.block_i, in_channels, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels, base_channels, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels, base_channels, kernel_size=3, stride=1, bn=1))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 2
        self.register_block(self.conv(self.block_i, base_channels, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*2, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*2, kernel_size=3, stride=1, bn=1))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 3
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*4, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*4, kernel_size=3, stride=1, bn=1))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # down stage 4
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*8, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*8, kernel_size=3, stride=1, bn=1))
        self.register_block(self.maxpool(self.block_i, kernel_size=2, stride=2))

        # middle
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*16, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*16, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*16, kernel_size=3, stride=1, bn=1))

        # up stage 4
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(AttentionGate(base_channels*8, base_channels*8, base_channels*8 // 2))
        self.register_block(self.conv(self.block_i, base_channels*16, base_channels*8, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*8, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*8, kernel_size=3, stride=1, bn=1))

        # up stage 3
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(AttentionGate(base_channels*4, base_channels*4, base_channels*4 // 2))
        self.register_block(self.conv(self.block_i, base_channels*8, base_channels*4, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*4, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*4, kernel_size=3, stride=1, bn=1))

        # up stage 2
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(AttentionGate(base_channels*2, base_channels*2, base_channels*2 // 2))
        self.register_block(self.conv(self.block_i, base_channels*4, base_channels*2, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*2, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels*2, kernel_size=3, stride=1, bn=1))

        # up stage 1
        self.register_block(self.upsample(self.block_i, stride=2))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(AttentionGate(base_channels, base_channels, base_channels // 2))
        self.register_block(self.conv(self.block_i, base_channels*2, base_channels, kernel_size=1, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels, base_channels, kernel_size=3, stride=1, bn=1))
        self.register_block(self.conv(self.block_i, base_channels, base_channels, kernel_size=3, stride=1, bn=1))

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
                route_stage_4 = self.block_list[block_i](route_stage_4, o)
                block_i += 1
                o = torch.cat([route_stage_4, o], dim=1)

        # up stage 3
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                route_stage_3 = self.block_list[block_i](route_stage_3, o)
                block_i += 1
                o = torch.cat([route_stage_3, o], dim=1)
        
        # up stage 2
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                route_stage_2 = self.block_list[block_i](route_stage_2, o)
                block_i += 1
                o = torch.cat([route_stage_2, o], dim=1)
        
        # up stage 1
        for i in range(5):
            o = self.block_list[block_i](o); 
            block_i += 1
            if i == 1:
                route_stage_1 = self.block_list[block_i](route_stage_1, o)
                block_i += 1
                o = torch.cat([route_stage_1, o], dim=1)
        
        # output stage
        o = self.block_list[block_i](o); block_i += 1
        o_softmax = nn.Softmax(dim=1)(o)
        
        return o, o_softmax

