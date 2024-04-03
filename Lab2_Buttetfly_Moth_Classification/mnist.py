import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19
from dataloader import BufferflyMothLoader
import os
path = os.getcwd()
train_loader = BufferflyMothLoader(path + "/dataset", "train")
test_loader = BufferflyMothLoader(path + "/dataset", "test")

# Define VGG19 model
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        self.vgg = vgg19(pretrained=False)
        self.vgg.classifier[6] = nn.Linear(4096, 100)  # Modify last layer for 10 classes

    def forward(self, x):
        return self.vgg(x)

# Instantiate the model
model = vg()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch * len(images), len(train_loader), 100.0 * batch / len(train_loader), loss.item()))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")