import torch
import torch.nn as nn
from pathlib import Path
from Network import NetWork
from loader import PNGLoader

def train():
    pass

def test():
    pass

def valid():
    pass

if __name__ == '__main__':
    modelpath = Path(__file__).resolve()
    trainpath = modelpath.parent / 'data/train'
    testpath = modelpath.parent / 'data/test'
    validpath = modelpath.parent / 'data/val'

    epoches = 30
    lr = 0.01
    batch_size = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader, validloader = PNGLoader(trainpath,testpath,validpath,batch_size,True)
    model = NetWork(3, (224,224), [50,20,2])

    for epoch in range(epoches):
        train()
        test()

    valid()

    # save_model

    # save_logs
