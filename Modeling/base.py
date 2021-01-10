import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init


class Mish(nn.Module):
    '''
        Mish 激活函数, 是最近提出的一种新型激活函数, 比起 ReLU 来说有着更加柔和的梯度, 并且
    在多个数据集上测试发现比起 ReLU 都有一定的效果提升。
        论文题目: Mish: A Self Regularized Non-Monotonic Neural Activation Function
        论文链接: https://arxiv.org/pdf/1908.08681.pdf
    '''
    def __init__(self):
        super(Mish, self).__init__()
        # print("Mish activation loaded...")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))

        return x


class Upsample(nn.Module):
    '''
        上采样层, 通过 torch.nn.functional 这个库中的插值函数 interpolate 来实现, 对于 init 函数
    有两个参数:
        scale_factor: 缩放因子, 是可以大于 1 也可以小于 1 的浮点数。
        mode: 插值方式, 即选择不同的插值算法来实现, 一般采用的是最近邻插值。
    对于 forward 函数, 则是直接对输入的特征图进行插值处理然后再输出即可。
    '''
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        
        return x
    

class EmptyLayer(nn.Module):
    '''
        空层, 这主要是为了让 route 层和 shortcut 层也有着相似的层定义。
    '''
    def __init__(self):
        super(EmptyLayer, self).__init__()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def _create_conv_block(block_index, in_channels, out_channels, kernel_size, stride, bn, activation="Mish", init_type="kaiming"):
    block = nn.Sequential()

    block.add_module(
        f"conv_{block_index}",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int(kernel_size - 1) // 2,
            bias=not bn,
        )
    )

    if bn:
        block.add_module(f"bn_{block_index}", nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))

    if activation == "Mish":
        block.add_module(f"mish_{block_index}", Mish())
    elif activation == "LeakyReLU":
        block.add_module(f"leakyReLU_{block_index}", nn.LeakyReLU(0.1))
    elif activation == "Softmax":
        block.add_module(f"softmax_{block_index}", nn.Softmax(dim=1))
    elif activation == "ReLU":
        block.add_module(f"ReLU_{block_index}", nn.ReLU())
    elif activation == "Sigmoid":
        block.add_module(f"Sigmoid_{block_index}", nn.Sigmoid())

    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            init_weights(m, init_type=init_type)
        elif isinstance(m, nn.BatchNorm2d):
            init_weights(m, init_type=init_type)

    return block


# def _create_trans_conv_block(block_index, in_channels, out_channels, kernel_size, stride, bn, rate=1, activation="Mish"):
#     block = nn.Sequential()

#     block.add_module(
#         f"conv_{block_index}",
#         nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=kernel_size,
#             stride=stride,
#             padding=int(kernel_size + rate) // 2,
#             bias=not bn,
#             rate=rate
#         )
#     )

#     if bn:
#         block.add_module(f"bn_{block_index}", nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5))

#     if activation == "Mish":
#         block.add_module(f"mish_{block_index}", Mish())
#     elif activation == "LeakyReLU":
#         block.add_module(f"leakyReLU_{block_index}", nn.LeakyReLU(0.1))
#     elif activation == "Softmax":
#         block.add_module(f"softmax_{block_index}", nn.Softmax(dim=1))
#     elif activation == "ReLU":
#         block.add_module(f"ReLU_{block_index}", nn.ReLU())
#     elif activation == "Sigmoid":
#         block.add_module(f"Sigmoid_{block_index}", nn.Sigmoid())

#     for m in block.modules():
#         if isinstance(m, nn.Conv2d):
#             init_weights(m, init_type=init_type)
#         elif isinstance(m, nn.BatchNorm2d):
#             init_weights(m, init_type=init_type)

#     return block

    
def _create_maxpool_block(block_index, kernel_size, stride):
    block = nn.Sequential()

    if kernel_size == 2 and stride == 1:
        block.add_module(f"_pad_fix_{block_index}", nn.ZeroPad2d((0, 1, 0, 1)))

    block.add_module(
        f"maxpool_{block_index}",
        nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=int(kernel_size - 1) // 2
        )
    )

    return block

    
def _create_upsample_block(block_index, stride, mode="nearest"):
    block = nn.Sequential()

    block.add_module(
        f"upsample_{block_index}",
        Upsample(
            scale_factor=stride
        ),
    )

    return block



