import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)  # 20 classes

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = VOCClassification(root='./data', year='2012', image_set='train', transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#trainset = ImageFolder(root='path/to/train', transform=transform)
#trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = VOCClassification(root='./data', year='2012', image_set='val', transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
#testset = ImageFolder(root='path/to/test', transform=transform)
#testloader = DataLoader(testset, batch_size=64, shuffle=False)

model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

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
        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}')


def test_model_with_confusion_matrix(model, testloader):
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model: {accuracy:.2f}%')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainloader.dataset.categories,
                yticklabels=trainloader.dataset.categories)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

train_model(model, trainloader, criterion, optimizer, epochs=30)
# Use this function to test and visualize the confusion matrix
test_model_with_confusion_matrix(model, testloader)

