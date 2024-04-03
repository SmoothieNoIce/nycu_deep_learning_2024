# %%
import os
import torch
import torch.optim as optim
from VGG19 import VGG19_alt, VGG19
from dataloader import BufferflyMothLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            # print(output)
            t1 = output[0]
            p1 = torch.max(t1)
            p3, p4 = torch.max(output, 1)
            correct += p4.eq(target).sum().item()

    test_loss /= len(test_loader)

    accuracy = 100.0 * correct / len(test_loader)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader), accuracy))

    return test_loss, accuracy


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            # print(output)
            p3, p4 = torch.max(output, 1)
            correct += p4.eq(target).sum().item()

    test_loss /= len(test_loader)

    accuracy = 100.0 * correct / len(test_loader)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(test_loss, correct, len(test_loader), accuracy))

    return test_loss, accuracy


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        # print("反向传播前, 参数的梯度为: ", data.grad)
        loss.backward()
        # print("反向传播后, 参数的梯度为: ", data.grad)
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(data), len(train_loader), 100.0 * batch_idx / len(train_loader), loss.item()))

    """ while True:
        loss = 0
        for i in range(0, train_loader.__len__()):
            img, ground_truth_label, ground_truth_label_id = bufferflyMothLoader_train.__getitem__(i)
            label_tensor = torch.tensor([ground_truth_label_id]).cuda()
            net = VGG19_alt(num_classes).to(device)
            img_torch = torch.from_numpy(img).cuda()
            img_torch = img_torch.unsqueeze(0)
            output = net(img_torch)
            train_loss.append(loss)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, label_tensor)
            print(loss)
        if epoch % 500 == 0:
            print(f"[Epoch:{epoch:6}] [Loss: {loss:.6f}]")
        epoch += 1
        if loss <= loss_thres or epoch >= max_epochs:
            print(f"[Epoch:{epoch:6}] [Loss: {loss:.6f}]")
            break """


if __name__ == "__main__":
    print("Good Luck :)")

    use_cuda = True
    learning_rate = 0.01
    max_epoch = 10  # 最多訓練的次數
    epoch = 1
    current_epoch = epoch
    is_evaluating = False
    loss_all = []
    acc_all = []

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
    train_loader = BufferflyMothLoader(path + "/dataset", "train")
    test_loader = BufferflyMothLoader(path + "/dataset", "test")

    num_classes = train_loader.total_labels
    model = VGG19_alt(num_classes).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    if checkpoint:
        checkpoint = torch.load("vgg19.pt")
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss_all = checkpoint["loss"]
        acc_all = checkpoint["acc"]

    if is_evaluating:
        checkpoint = torch.load("vgg19.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        loss, acc = evaluate(model, device, test_loader)
        exit()

    scheduler = StepLR(optimizer, step_size=1)
    summary(model, (3, 224, 224))

    try:
        for epoch in range(epoch, max_epoch + 1):
            current_epoch = epoch
            train(model, device, train_loader, optimizer, epoch)
            loss, acc = test(model, device, test_loader)
            loss_all.append(loss)
            acc_all.append(acc)
            scheduler.step()
    except KeyboardInterrupt:
        torch.save(
            {
                "epoch": current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "acc": acc_all,
                "loss": loss_all,
            },
            "vgg19.pt",
        )
        print("KeyboardInterrupt: Model saved")
        exit()

    torch.save(
        {
            "epoch": max_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "acc": acc_all,
            "loss": loss_all,
        },
        "vgg19.pt",
    )

# %%
