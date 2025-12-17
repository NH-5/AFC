import torch
import torch.nn as nn
from CNN import CNNet as cnn
from MLP import MLP as mlp

class NetWork(nn.Module):
    def __init__(self, inch, insize, size:list):
        super().__init__()

        self.cnn = cnn(inch)
        with torch.no_grad():
            imgsize = (inch, *insize)
            dummy = torch.zeros(1, *imgsize)
            out = self.cnn(dummy)
            self.flat_dim = out.numel()
        size.insert(0,self.flat_dim)
        self.mlp = mlp(size)

    def forward(self, x):
        return self.mlp(self.cnn(x))
    

if __name__ == '__main__':
    x = torch.randn(8, 3, 224, 224, device='cuda')
    net = NetWork(3, (224, 224), [20, 2])
    net = net.to('cuda')
    y = net(x)
    print(y)