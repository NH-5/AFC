import torch
import torch.nn as nn
from basic import BasicConvAvgPool as bcap

class CNNet(nn.Module):
    def __init__(self, inch):
        super().__init__()

        modelist = nn.ModuleList()
        for _ in range(4):
            modelist.append(
                bcap(
                    inch=inch,
                    outch=2*inch,
                    convkernel=3
                ),
            )
            modelist.append(nn.ReLU())
            inch = 2*inch
        
        modelist.append(nn.Flatten(start_dim=1))
        self.model = nn.Sequential(*modelist)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    """调试用"""
    x = torch.rand(1,3,100,100,device='cuda')
    net = CNNet(inch=3)
    net = net.to('cuda')
    y = net(x)
    print(y.shape)
