from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def PNGLoader(trainpath, testpath, validpath,batch_size=64, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    train_dataset = ImageFolder(root=trainpath, transform=transform)
    test_dataset = ImageFolder(root=testpath, transform=transform)
    valid_dataset = ImageFolder(root=validpath, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader

if __name__ == '__main__':
    """调试用"""
    from pathlib import Path
    modelpath = Path(__file__).resolve()
    modelpath = modelpath.parent.parent

    trainpath = modelpath / 'data/train'
    testpath = modelpath / 'data/test'
    validpath = modelpath / 'data/val'
    _, _, validloader = PNGLoader(trainpath,testpath,validpath,4)
    for batch in validloader:
        print(len(batch))
        print(batch[0][3].size())
        print(batch[1][3])
        break