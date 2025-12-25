import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from Network import NetWork
from loader import PNGLoader
import matplotlib.pyplot as plt

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
    print(f"Accuracy for Validation is {accuracy}.")
    return accuracy

if __name__ == '__main__':

    codepath = Path(__file__).resolve()
    rootpath = codepath.parent.parent
    trainpath = rootpath / 'data/train'
    testpath = rootpath / 'data/test'
    validpath = rootpath / 'data/val'


    epoches = 30
    lr = 0.001
    batch_size = 128
    size = [256,32,2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Loss = []
    Accuracy = []
    Parameters = {
        'epoches':epoches,
        'lr':lr,
        'batch_size':batch_size,
        'size':size
    }


    trainloader, testloader, validloader = PNGLoader(trainpath,testpath,validpath,batch_size,True)
    print(f"data load done")

    model = NetWork(3, (256,256), size)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    print(f"model create done")

    for epoch in range(epoches):
        train_loss = train(model, trainloader, criterion, optimizer, device)
        test_accuracy = evaluate(model, testloader, device)
        print(f"Epoch {epoch+1} : loss is {train_loss} accuracy is {test_accuracy}.")
        Loss.append(train_loss)
        Accuracy.append(test_accuracy)
    print(f"train done")

    valid_accuracy = valid(model, validloader, device)

    # time now
    time = datetime.now().astimezone()
    time = time.strftime("%Y-%m-%d %H:%M:%S UTC+8")

    # save_logs
    logspath = rootpath / f'model/{time}/'
    logspath.mkdir(exist_ok=True)
    filepath = logspath / f"log.txt"
    with open(filepath, 'a') as f:
        for key in Parameters:
            print(f"{key} is {Parameters[key]}.", file=f)

        for epoch in range(epoches):
            info = f"Epoch {epoch + 1}: loss is {Loss[epoch]}, accuracy is {Accuracy[epoch]}."
            print(info, file=f)

        print(f"Accuracy for Validation is {valid_accuracy}.", file=f)

    # save_model
    savepath = logspath / 'model.pt'
    torch.save(model, savepath)

    # visualization
    # Plot 1: Training Loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoches + 1), Loss, label='Train Loss', color='blue', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(logspath / 'training_loss.png')
    plt.close()

    # Plot 2: Test Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoches + 1), Accuracy, label='Test Accuracy', color='green', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(logspath / 'test_accuracy.png')
    plt.close()

