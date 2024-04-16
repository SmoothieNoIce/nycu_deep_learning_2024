# %%
import os
import torch
import torch.optim as optim
from VGG19 import VGG19
from ResNet50 import ResNet50
from dataloader import BufferflyMothLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def evaluate(model, device, test_loader, type="Train", batch_size=1):
    model.eval()
    avg_loss = 0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            avg_loss += criterion(output, target).item()  # sum up batch loss
            _, max = torch.max(output, 1)
            correct += max.eq(target).sum().item()

    avg_loss /= len(test_loader)
    accuracy = 100.0 *  int(correct / batch_size) / len(test_loader)
    print("\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(type, avg_loss, int(correct / batch_size), len(test_loader), accuracy))

    return avg_loss, accuracy

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100.0 * batch_idx / len(train_loader)}%)]\tLoss: {loss.item()}")

if __name__ == "__main__":
    print("Good Luck :)")

    use_cuda = True
    learning_rate = 0.0001
    max_epoch = 30  # 最多訓練的次數
    epoch = 1
    current_epoch = epoch
    is_evaluating = True
    train_loss_all = []
    train_acc_all = []
    valid_loss_all = []
    valid_acc_all = []

    checkpoint = input("Do you want to load checkpoint? (y/n): ")
    if checkpoint == "y":
        checkpoint = True
    else:
        checkpoint = False

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    path = os.getcwd()
    train_loader = DataLoader(BufferflyMothLoader(path + "/dataset", "train"), batch_size=32, shuffle=True)
    test_loader = DataLoader(BufferflyMothLoader(path + "/dataset", "test"), batch_size=1, shuffle=True)
    valid_loader = DataLoader(BufferflyMothLoader(path + "/dataset", "valid"), batch_size=1, shuffle=True)

    num_classes = 100

    model_type = input("Which Model? (vgg19=0/resnet=1): ")
    if model_type == "0":
        model = VGG19(num_classes).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        if checkpoint:
            checkpoint = torch.load("./models/vgg19-20-acc-4.4.pt")
            # model.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]+1
            train_loss_all = checkpoint["train_loss_all"]
            train_acc_all = checkpoint["train_acc_all"]
            valid_loss_all = checkpoint["valid_loss_all"]
            valid_acc_all = checkpoint["valid_acc_all"]
    else:
        model = ResNet50(num_classes).to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
        if checkpoint:
            checkpoint = torch.load("./models/resnet50-20-acc-30.4.pt")
            # model.load_state_dict(checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint["epoch"]+1
            train_loss_all = checkpoint["train_loss_all"]
            train_acc_all = checkpoint["train_acc_all"]
            valid_loss_all = checkpoint["valid_loss_all"]
            valid_acc_all = checkpoint["valid_acc_all"]


    scheduler = StepLR(optimizer, step_size=1)
    summary(model, (3, 224, 224))

    if is_evaluating:
        valid_loss, valid_acc = evaluate(model, device, test_loader, "Testing", 1)
        exit()
        
    for epoch in range(epoch, max_epoch + 1):
        current_epoch = epoch
        train(model, device, train_loader, optimizer, epoch)

        train_loss, train_acc = evaluate(model, device, train_loader, "Train", 32)
        valid_loss, valid_acc = evaluate(model, device, valid_loader, "Valid", 1)

        current_best_valid_acc = max(valid_acc_all) if valid_acc_all else 0
        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        valid_loss_all.append(valid_loss)
        valid_acc_all.append(valid_acc)
        scheduler.step()
        data = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss_all": train_loss_all,
                    "train_acc_all": train_acc_all,
                    "valid_loss_all": valid_loss_all,
                    "valid_acc_all": valid_acc_all,
                }

        if model_type == "0":
            if valid_acc >= current_best_valid_acc:
                print(f"New best model found! Save it as vgg19-{epoch}.pt")
                torch.save(data,f"vgg19-{epoch}-best-acc-{valid_acc}.pt",)
            else:
                torch.save(data,f"vgg19-{epoch}-acc-{valid_acc}.pt",)
        else:
            if valid_acc >= current_best_valid_acc:
                print(f"New best model found! Save it as resnet50-{epoch}.pt")
                torch.save(data,f"resnet50-{epoch}-best-acc-{valid_acc}.pt",)
            else:
                torch.save(data,f"resnet50-{epoch}-acc-{valid_acc}.pt",)

# %%
