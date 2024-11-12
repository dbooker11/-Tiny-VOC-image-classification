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

class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2)  
        x = x.transpose(1, 2)  
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, mlp_dim, drop_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(drop_rate),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        mlp_output = self.mlp(x)
        x = self.layer_norm2(x + mlp_output)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=20, embed_dim=64, depth=6, heads=8, mlp_dim=128):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, num_channels=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 1, embed_dim))
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, heads, mlp_dim) for _ in range(depth)]
        )
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        patches = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  
        x = torch.cat((cls_tokens, patches), dim=1)  
        x += self.pos_embedding  
        x = self.transformer_blocks(x)  
        cls_output = x[:, 0]  
        x = self.fc(cls_output)  
        return x


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

#trainset = ImageFolder(root='path/to/train', transform=transform)
#trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
trainset = VOCClassification(root='./data', year='2012', image_set='train', transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
#testset = ImageFolder(root='path/to/test', transform=transform)
#testloader = DataLoader(testset, batch_size=64, shuffle=False)
testset = VOCClassification(root='./data', year='2012', image_set='val', transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

def train_model(model, trainloader, criterion, optimizer, epochs=30):
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
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}')

def test_model(model, testloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the vit model: {100 * correct / total:.2f}%')

model = VisionTransformer(image_size=32, patch_size=8, num_classes=20, embed_dim=64, depth=6, heads=8, mlp_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, trainloader, criterion, optimizer, epochs=10)
test_model(model, testloader)
