import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size
        self.num_layers = len(size)


        modelist = nn.ModuleList()
        for i in range(self.num_layers - 1):
            modelist.append(nn.Linear(self.size[i], self.size[i+1]))
            if i < self.num_layers - 2:
                modelist.append(nn.Sigmoid())

        self.model = nn.Sequential(*modelist)

    
    def forward(self, x):
        return self.model(x)
    

if __name__ == '__main__':
    x = torch.randn(10,device='cuda')
    size = [10,6,2]
    net = MLP(size)
    net = net.to('cuda')
    y = net(x)
    print(y)