import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from Network import NetWork
from loader import PNGLoader

def train(model, train_loader, criterion, optimizer, device):

    model.train()
    total_loss = 0

    for batch in train_loader:
        features, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    return total_loss / len(train_loader)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            features, labels = batch[0].to(device), batch[1].to(device)
            output = model(features)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

def valid(model, valid_loader, device):
    accuracy = evaluate(model, valid_loader, device)
    print(f"Accuracy for Validation is {accuracy}.\n")
    return accuracy

if __name__ == '__main__':

    modelpath = Path(__file__).resolve()
    trainpath = modelpath.parent.parent / 'data/train'
    testpath = modelpath.parent.parent / 'data/test'
    validpath = modelpath.parent.parent / 'data/val'


    epoches = 30
    lr = 0.001
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    trainloader, testloader, validloader = PNGLoader(trainpath,testpath,validpath,batch_size,True)
    print(f"data load done")

    model = NetWork(3, (3,224,224), [256,192,2])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    print(f"model create done")

    for epoch in range(epoches):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_accuracy = evaluate(model, testloader, device)
        print(f"Epoch {epoch+1} : loss-{train_loss} accuracy-{test_accuracy}.")

    valid_accuracy = valid(model, validloader, device)

    # save_model
    if valid_accuracy >= 0.9:
        torch.save(model.state_dict(), 'model.pth')
        print(f"model save done")

    # save_logs
