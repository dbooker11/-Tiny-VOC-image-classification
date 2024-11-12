import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VOCClassification(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', transform=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, transform=transform, download=download)
        self.categories = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        objects = target['annotation']['object']
        label = self.categories.index(objects[0]['name'])
        return img, label

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 20)  # 20 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = VOCClassification(root='./data', year='2012', image_set='train', transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#trainset = ImageFolder(root=r"C:\Users\ZHENGZHIQIAN\Desktop\TinySeg\images\train", transform=transform)
#trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = VOCClassification(root='./data', year='2012', image_set='val', transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
#testset = ImageFolder(root=r"C:\Users\ZHENGZHIQIAN\Desktop\TinySeg\images\test", transform=transform)
#testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = VGGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, trainloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

def test_model(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data  # 从 testloader 中获取 images 和 labels
            images, labels = images.to(device), labels.to(device)  # 将 images 和 labels 转移到设备上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the vgg model: {100 * correct / total:.2f}%')

train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)
