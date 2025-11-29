import torch
import torch.nn as nn



"""
此文件构造三个基本结构:
    - 纯卷积层
    - 卷积->平均池化
    - 卷积->最大池化
"""

class BasicConv(nn.Module):
    """
    conv->ReLU
    """
    def __init__(
            self,
            inchannels:int,
            outchannels:int,
            kernelsize:int = 3,
            stride:int = 1,
            padding:int = 0
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=inchannels,
            out_channels=outchannels,
            kernel_size=kernelsize,
            stride=stride,
            padding=padding
        )
        self.relu = nn.ReLU()
        self.model = nn.Sequential(self.conv, self.relu)

    def forward(self, x):
        return self.model(x)
    

class BasicConvMaxPool(nn.Module):
    """
    conv-ReLU-maxpool
    """
    def __init__(
            self,
            inch : int,
            outch : int,
            convkernel : int,
            convstride : int = 1,
            convpadding : int = 0,
            poolkernel : int = 2,
            poolpadding : int = 0,
            poolstride : int = 2
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=convkernel,
            stride=convstride,
            padding=convpadding
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(
            kernel_size=poolkernel,
            stride=poolstride,
            padding=poolpadding
        )

        self.model = nn.Sequential(
            self.conv2d,
            self.relu,
            self.maxpool
        )


    def forward(self, input):
        return self.model(input)
    

class BasicConvAvgPool(nn.Module):
    """
    conv-ReLU-avgpool
    """
    def __init__(
            self,
            inch : int,
            outch : int,
            convkernel : int,
            convstride : int = 1,
            convpadding : int = 0,
            poolkernel : int = 2,
            poolpadding : int = 0,
            poolstride : int = 2
    ):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=inch,
            out_channels=outch,
            kernel_size=convkernel,
            stride=convstride,
            padding=convpadding
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(
            kernel_size=poolkernel,
            stride=poolstride,
            padding=poolpadding
        )

        self.model = nn.Sequential(
            self.conv2d,
            self.relu,
            self.maxpool
        )


    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    """调试用"""

    x = torch.randn(3,10,10)
    model = BasicConvAvgPool(3,3,2)
    y = model(x)
    print(y)
