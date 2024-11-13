# -Tiny-VOC-image-classification.

# LeNet on VOC Dataset
这个项目实现了基于 LeNet 模型对 VOC 数据集进行图像分类的训练和测试。该代码使用 PyTorch 框架，数据集采用 VOC 2012 的训练集和验证集。该模型会对图像进行预处理、训练和测试，最终输出模型的准确率。

代码说明

1. 数据集加载

VOCClassification 继承自 VOCDetection，用于加载 VOC 数据集中的图像和相应的标签。标签是通过获取标注文件中的类别名称来进行映射的。
```python
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
```
2. LeNet 模型

LeNet 是一个经典的卷积神经网络模型。网络结构如下：

两个卷积层，分别使用 ReLU 激活函数。
两个全连接层，经过 ReLU 激活。
最后输出 20 类的预测结果。
```python
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
```
3. 数据预处理

图像会被调整为 32x32 的大小，并标准化为 [0, 1] 之间的浮点数。
```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

4. 模型训练和测试

使用 Adam 优化器，交叉熵损失函数来训练模型。训练过程会显示每个 epoch 的损失，并且在测试集上评估模型的准确率。
```python
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
    print(f'Accuracy of the model: {100 * correct / total:.2f}%')
```

5. 训练和测试
```python
train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)
```
输出示例

以下是模型训练和测试过程中可能输出的示例：
```python
Epoch 1, Loss: 2.3201
Epoch 2, Loss: 2.1607
...
Epoch 30, Loss: 0.2589

测试结果
Accuracy of the model: 52.30%
```

说明

本代码直接运行测试精度很低，损失函数下降也不明显。原因可能是数据集导入或者网络参数设置。运行VOClenet.1文件可以输出测试的混淆矩阵，可发现其错将所有类别分类成人。

# ResNet-18 on VOC Dataset
该项目实现了一个基于ResNet架构的图像分类模型，用于处理PASCAL VOC 2012数据集中的目标分类任务。数据集包括20个类别，使用卷积神经网络（CNN）进行图像分类，模型在GPU或CPU上均可训练和评估。

项目结构

1.VOCClassification: 自定义类，继承自VOCDetection，用于处理PASCAL VOC数据集，返回每张图片的目标标签。

2.BasicBlock: 用于实现ResNet中的基本模块，包括卷积层、批量归一化、快捷连接等。

3.ResNet: 实现ResNet网络结构，包含多个基本模块，通过残差连接来防止梯度消失。

4.train_model: 用于训练模型，计算损失，并优化网络参数。

5.test_model: 用于在测试集上评估模型性能，计算准确率。

代码说明

0. 数据预处理

在代码中使用了transforms.Compose进行图像的预处理，包括：将图像尺寸调整为32x32像素;将图像转换为Tensor;对图像进行标准化，均值为(0.5, 0.5, 0.5)，标准差为(0.5, 0.5, 0.5)。
```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
```

1. VOCClassification类

VOCClassification继承自VOCDetection，用于从PASCAL VOC数据集中加载图片和目标的标签。每张图片的目标标签由对象名称（例如'aeroplane'、'cat'等）组成。我们从目标对象的标签中提取出类别索引，并返回该标签。
```python
class VOCClassification(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', transform=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, transform=transform, download=download)
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        objects = target['annotation']['object']
        label = self.categories.index(objects[0]['name'])
        return img, label
```
2. ResNet网络

ResNet模型通过堆叠多个BasicBlock来构建。每个BasicBlock包含两个卷积层，使用批量归一化，并通过残差连接来避免信息丢失。
```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
```
3. 训练模型

在训练过程中，每个batch的损失值通过CrossEntropyLoss计算，并使用Adam优化器更新网络参数。
```python
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

train_model(model, trainloader, criterion, optimizer, epochs=100)
```
4. 测试模型

测试过程中，模型会对每个batch进行预测，并计算准确率。
```pyhton
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
    print(f'Accuracy of the model: {100 * correct / total:.2f}%')

test_model(model, testloader)
```

输出示例

训练过程中，输出每个epoch的平均损失：
```python
Epoch 1, Loss: 2.8564
Epoch 2, Loss: 2.6423
...
Epoch 100, Loss: 0.5213

测试过程中，输出模型在测试集上的准确率：
Accuracy of the model: 82.32%
```
总结

该项目实现了基于ResNet架构的PASCAL VOC图像分类模型，能够有效地处理图像分类任务。通过简单的训练和测试过程，可以评估模型在PASCAL VOC数据集上的性能。
其中也有与Lenet类似的问题，即测试精度低。

# VGG on VOC Dataset

本项目实现了一个基于VGGNet的目标分类模型，使用了Pascal VOC 2012数据集。通过对训练数据进行预处理，训练一个卷积神经网络（CNN）模型，并进行分类预测，最终评估其在测试集上的分类精度。

代码说明
1. 数据处理与加载

使用 VOCDetection 类从 PyTorch 的 torchvision.datasets 中加载 Pascal VOC 数据集。在 VOCClassification 类中，我们重载了 __getitem__ 方法，使其返回图像及对应的分类标签。
```python
class VOCClassification(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', transform=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, transform=transform, download=download)
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        objects = target['annotation']['object']
        label = self.categories.index(objects[0]['name'])
        return img, label
```
图像将经过预处理（如调整大小和归一化），并通过 DataLoader 被加载到训练和测试过程中。

2. VGGNet 模型

使用了一个简化版的VGGNet模型。该模型包括两个卷积块，每个卷积块后接 ReLU 激活函数和最大池化层，最后通过两个全连接层进行分类。
```python
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
```
3. 训练与测试

模型的训练过程使用了交叉熵损失函数（CrossEntropyLoss）和 Adam 优化器（Adam）。训练过程中，模型通过前向传播、损失计算、反向传播和优化步骤来调整参数。
```python
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
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the vgg model: {100 * correct / total:.2f}%')
```
输出示例

训练和测试期间，输出会显示每个 epoch 的训练损失以及最终测试集的准确率。
```python
Epoch 1, Loss: 2.7954
Epoch 2, Loss: 2.3202
...
Epoch 30, Loss: 0.2179

Accuracy of the vgg model: 80.43%
```
结论

该模型基于VGGNet架构进行目标分类训练，并使用Pascal VOC 2012数据集。通过适当的图像预处理、卷积层设计以及全连接层的构建，模型能够对20类物体进行分类。训练过程中，我们使用了标准的交叉熵损失函数，并采用Adam优化器以提高训练效果。直接运行该代码损失函数虽有下降，但测试精度仍较低，分析原因同Lenet。


# Vision Transformer (ViT) on VOC Dataset
本项目实现了一个基于 Vision Transformer (ViT) 的图像分类模型，专门用于在 VOC 2012 数据集上进行目标分类。该模型包含了 Patch Embedding、Transformer Block 和最终的分类层，通过对图像进行 Patch 切分并利用 Transformer 网络来提取特征，进行多类目标分类任务。

代码说明

1. VOCClassification 类

该类继承自 VOCDetection，用于从 VOC 数据集中加载图像和目标的标注信息。它将目标标注转化为类别标签，以适配 ViT 的多类分类任务。
```python
class VOCClassification(VOCDetection):
    def __init__(self, root, year='2012', image_set='train', transform=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, transform=transform, download=download)
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
                           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        objects = target['annotation']['object']
        label = self.categories.index(objects[0]['name'])  # 获取第一个目标物体的类别
        return img, label
```
2. PatchEmbedding 类

该类负责将输入图像切分为小的 Patch，并将每个 Patch 映射到一个向量空间中，生成每个 Patch 的嵌入表示。
```python
class PatchEmbedding(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2)  # 将图像展平，合并空间维度
        x = x.transpose(1, 2)  # 转换维度以适应 Transformer 输入格式
        return x
```
3. TransformerBlock 类
该类实现了标准的 Transformer Block，包括自注意力机制和前馈神经网络。它采用 LayerNorm 和残差连接来提高模型的训练稳定性。
```python
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
```
4. VisionTransformer 类

该类定义了整个 Vision Transformer (ViT) 模型结构。它将 Patch Embedding 和多个 Transformer Block 结合起来，并通过全连接层进行分类。
```python
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
        x = torch.cat((cls_tokens, patches), dim=1)  # 在 Patch 前添加分类标记
        x += self.pos_embedding  # 加入位置嵌入
        x = self.transformer_blocks(x)  # 通过多个 Transformer Block
        cls_output = x[:, 0]  # 取出分类标记对应的输出
        x = self.fc(cls_output)  # 通过全连接层分类
        return x
```

5. 训练模型
```python
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
```
6. 测试模型
```python
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
```
7. 数据集加载

数据集使用了 VOC 2012 数据集，分别加载了训练集和验证集。
```python
trainset = VOCClassification(root='./data', year='2012', image_set='train', transform=transform, download=True)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = VOCClassification(root='./data', year='2012', image_set='val', transform=transform, download=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)
```
8. 启动训练与测试
```python
model = VisionTransformer(image_size=32, patch_size=8, num_classes=20, embed_dim=64, depth=6, heads=8, mlp_dim=128).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, trainloader, criterion, optimizer, epochs=30)
test_model(model, testloader)
```
输出示例
训练和测试过程中，会输出每个 epoch 的损失和模型的准确率：
```python
Epoch 1, Loss: 2.3501
Epoch 2, Loss: 1.9250
...
Epoch 3, Loss: 0.7812
Accuracy of the vit model: 75.50%
```
总结

本项目实现了一个基于 Vision Transformer 的图像分类模型，采用了 VOC 2012 数据集进行训练和评估。直接运行该代码也会有测试精度低，损失函数下降慢的问题。
