# %%
import torch
from VGG19 import VGG19
from ResNet50 import ResNet50
import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_classes = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VGG19(num_classes).to(device)
    checkpoint = torch.load("./models/vgg19-20-acc-4.4.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    vgg19_train_loss_all = checkpoint["train_loss_all"]
    vgg19_train_acc_all = checkpoint["train_acc_all"]
    vgg19_valid_loss_all = checkpoint["valid_loss_all"]
    vgg19_valid_acc_all = checkpoint["valid_acc_all"]

    model = ResNet50(num_classes).to(device)
    checkpoint = torch.load("./models/resnet50-20-acc-30.4.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    resnet_train_loss_all = checkpoint["train_loss_all"]
    resnet_train_acc_all = checkpoint["train_acc_all"]
    resnet_valid_loss_all = checkpoint["valid_loss_all"]
    resnet_valid_acc_all = checkpoint["valid_acc_all"]

    plt.plot(vgg19_train_acc_all, label='VGG19 Train Accuracy', color='blue')
    plt.plot(vgg19_valid_acc_all, label='VGG19 Valid Accuracy', color='yellow')
    plt.plot(resnet_train_acc_all, label='ResNet Train Accuracy', color='green')
    plt.plot(resnet_valid_acc_all, label='ResNet Valid Accuracy', color='red')

    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
# %%
