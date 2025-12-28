#import "conf.typ":conf

#show: conf.with(
  exp: "AI生成图像的鉴别系统开发与应用",
  major: "人工智能",
  author: "吴征 朱志飞 钟晨灏",
  author_num: "23060809 23060841 23060840",
  teacher: "韩志敏",
  score: "",
  year: "2025",
  month: "12",
  day: "28"
)

= 任务要求

== 1题目概述

本题要求设计并实现一个基于深度学习的人脸图像分类系统，能够区分真实人脸和AI生成的深度伪造（Deepfake）人脸图像。该系统需要完成从数据获取、模型训练到应用部署的完整机器学习工作流程。

== 题目解析

本题目属于*监督学习*中的*二分类图像识别*任务。具体分析如下：

- *监督学习*：数据集中每张图像都有对应的标签（real/fake），模型通过学习有标签的数据来进行预测。
- *二分类*：任务目标是判断输入图像属于"真实人脸"还是"伪造人脸"，输出为两类概率分布。
- *图像分类*：从原始图像像素中提取特征，识别图像所属类别，而非进行目标检测或语义分割。

== 解题思路

本项目采用*卷积神经网络（CNN）*作为核心分类模型，解题思路分为以下几个阶段：

1. *数据准备阶段*：从Hugging Face开源数据集获取人脸图像，按比例划分为训练集、验证集和测试集。
2. *模型设计阶段*：构建CNN特征提取器结合多层感知机（MLP）的混合网络架构。
3. *模型训练阶段*：使用交叉熵损失函数和AdamW优化器进行有监督训练，通过验证集监控模型性能。
4. *模型部署阶段*：开发网页,Qt图形界面和Flask后端API，实现模型的可视化应用。

= 详细分工

本项目由小组协同完成，各成员分工如下：

#align(center)[
  #table(
    columns: (auto, auto),
    [*姓名*], [*主要任务*],
    [共同讨论开发], [模型设计与训练：CNN架构设计、参数调优、训练脚本编写],
    [朱志飞], [后端开发：Flask服务搭建、模型加载API设计、异常处理],
    [钟晨灏], [前端开发：网页和Qt界面设计、图像上传与显示、分类结果展示],
    [吴征], [数据处理与文档：数据集获取与清洗、报告撰写、版本控制],
  )
]

= 过程设计

== 解题步骤分解

#align(center)[
  #table(
    columns: (auto, auto, auto),
    [*步骤*], [*具体内容*], [*使用工具/算法*],
    [步骤1], [数据集获取与预处理], [Hugging Face datasets、torchvision transforms],
    [步骤2], [数据加载器构建], [PyTorch DataLoader、ImageFolder],
    [步骤3], [CNN模型设计], [卷积层、ReLU激活函数、平均池化层],
    [步骤4], [MLP分类器设计], [全连接层、Dropout、Softmax],
    [步骤5], [模型训练], [CrossEntropyLoss、AdamW优化器],
    [步骤6], [模型评估], [准确率计算、matplotlib可视化],
    [步骤7], [后端API开发], [Flask框架、模型序列化加载],
    [步骤8], [前端界面开发], [PyQt6、信号槽机制],
  )
]

== 关键技术说明

- *数据预处理*：使用`transforms.Compose`将图像统一缩放至256×256像素，并转换为张量格式。
- *CNN特征提取*：采用6层`BasicConvAvgPool`模块，每层将通道数翻倍，逐步提取图像深层特征。
- *分类器*：MLP将卷积层输出展平后，通过全连接层映射到2类输出。
- *优化器*：AdamW带有权重衰减（weight_decay=1e-3），有效防止过拟合。

= 编码实现及结果

== 开发环境

- *操作系统*：Linux (WSL2) / Windows
- *Python版本*：3.14.0
- *深度学习框架*：PyTorch 2.9.1+cu128
- *前端框架*：PyQt6(客户端软件)以及HTML/JS/CSS(网页)
- *后端框架*：Flask
- *可视化工具*：matplotlib

== 数据加载器实现

```python
# model/loader.py
def PNGLoader(trainpath, testpath, validpath, batch_size=64, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一图像尺寸
        transforms.ToTensor()            # 转换为张量 [0,1]
    ])
    train_dataset = ImageFolder(root=trainpath, transform=transform)
    test_dataset = ImageFolder(root=testpath, transform=transform)
    valid_dataset = ImageFolder(root=validpath, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, valid_loader
```

*注释*：`ImageFolder`自动根据目录结构生成标签，目录名即为类别标签。

== 卷积模块实现

```python
# model/basic.py
class BasicConvAvgPool(nn.Module):
    """卷积-ReLU-平均池化模块"""
    def __init__(self, inch, outch, convkernel,
                 convstride=1, convpadding=0,
                 poolkernel=2, poolpadding=0, poolstride=2):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=inch, out_channels=outch,
            kernel_size=convkernel, stride=convstride, padding=convpadding
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.AvgPool2d(
            kernel_size=poolkernel, stride=poolstride, padding=poolpadding
        )
        self.model = nn.Sequential(self.conv2d, self.relu, self.maxpool)

    def forward(self, input):
        return self.model(input)
```

*注释*：平均池化层步长为2，实现下采样功能，减少参数量的同时保留主要特征。

== CNN网络实现

```python
# model/CNN.py
class CNNet(nn.Module):
    def __init__(self, inch):
        super().__init__()
        modelist = nn.ModuleList()
        for _ in range(6):  # 6层卷积，通道数逐层翻倍
            modelist.append(BasicConvAvgPool(inch=inch, outch=2*inch, convkernel=3))
            modelist.append(nn.ReLU())
            inch = 2*inch
        modelist.append(nn.Flatten(start_dim=1))  # 展平用于MLP输入
        self.model = nn.Sequential(*modelist)

    def forward(self, x):
        return self.model(x)
```

*注释*：通过循环堆叠6层模块，输入通道从3开始，每层翻倍，最终得到高维特征向量。

== 4.5 完整网络与训练

```python
# model/Network.py
class NetWork(nn.Module):
    def __init__(self, inch, insize, size:list):
        super().__init__()
        self.cnn = cnn(inch)
        # 动态计算CNN输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, inch, *insize)
            out = self.cnn(dummy)
            self.flat_dim = out.numel()
        size.insert(0, self.flat_dim)
        self.mlp = mlp(size)

    def forward(self, x):
        return self.mlp(self.cnn(x))
```

```python
# model/train.py
# 训练参数设置
epoches = 30
lr = 0.001
batch_size = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

# 训练循环
for epoch in range(epoches):
    train_loss = train(model, trainloader, criterion, optimizer, device)
    test_accuracy = evaluate(model, testloader, device)
    print(f"Epoch {epoch+1}: loss={train_loss:.4f}, accuracy={test_accuracy:.4f}")
```

== 多组实验对比

为了找到最优模型配置，我们进行了6组对照实验，系统性地调整了batch_size和MLP分类器结构。

=== 实验参数对照表

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    [*实验编号*], [*batch_size*], [*MLP结构*], [*CNN输出维度*],
    [Exp 1], [64], [[3456,2048,512,128,2]], [3456],
    [Exp 2], [64], [[3456,256,32,2]], [3456],
    [Exp 3], [64], [[3456,512,128,2]], [3456],
    [Exp 4], [64], [[768,256,32,2]], [768],
    [Exp 5], [32], [[768,256,32,2]], [768],
    [Exp 6], [128], [[768,256,32,2]], [768],
  )
]

=== 实验结果对比

#align(center)[
  #table(
    columns: (auto, auto, auto, auto),
    [*实验编号*], [*测试集最高准确率*], [*最终验证准确率*], [*收敛情况*],
    [Exp 1], [89.13% (Epoch30)], [92.96%], [前12轮不学习，后快速收敛],
    [Exp 2], [89.32% (Epoch18)], [92.84%], [正常收敛],
    [Exp 3], [87.95% (Epoch14)], [88.76%], [过拟合严重，末期骤降],
    [Exp 4], [87.11% (Epoch3)], [91.25%], [快速收敛后波动],
    [Exp 5], [50% (全程)], [50%], [完全不收敛（batch过小）],
    [Exp 6], [87.91% (Epoch7)], [92.12%], [最佳泛化效果],
  )
]

=== 实验结果分析

1. *降维效果*：对比Exp 1/2/3（CNN输出3456维）与Exp 4/5/6（CNN输出768维），虽然高维特征的测试集准确率略高，但768维的验证集泛化效果反而更好（91-92% vs 88-93%），说明降维有效减少了过拟合。

2. *Batch Size影响*：
   - Exp 5（batch=32）：模型完全不收敛，30轮始终保持50%准确率，说明batch过小导致梯度估计方差过大。
   - Exp 6（batch=128）：验证集准确率最高（92.12%），泛化能力最佳。
   - 结论：适当增大batch_size有助于模型稳定收敛和泛化。

3. *MLP结构对比*（Exp 2 vs Exp 3）：
   - [3456,256,32,2] vs [3456,512,128,2]：更深更宽的MLP反而导致过拟合（Exp 3最终测试准确率仅79.1%），而轻量级MLP在Epoch18达到峰值后仍保持稳定。

4. *最佳实验*：Exp 6（batch=128, MLP=[768,256,32,2]）取得最佳验证集准确率92.12%，且训练曲线最稳定。

=== 最终选定模型参数

根据多组实验对比，最终选定以下配置作为实验成果模型：

#align(center)[
  #table(
    columns: (auto, auto),
    [*参数*], [*数值*],
    [batch_size], [128],
    [学习率], [0.001],
    [MLP结构], [[768, 256, 32, 2]],
    [优化器], [AdamW (weight_decay=1e-3)],
    [验证集准确率], [92.12%],
  )
]

=== 选定实验的训练过程

#align(center)[
  #table(
    columns: (auto, auto, auto),
    [*Epoch*], [*Loss*], [*Test Accuracy*],
    [1], [0.6821], [72.26%],
    [2], [0.5568], [84.99%],
    [3], [0.4852], [87.11%],
    [5], [0.4278], [86.43%],
    [7], [0.3821], [87.91%],
    [10], [0.3129], [83.41%],
    [15], [0.2050], [85.31%],
    [20], [0.1404], [84.65%],
    [25], [0.1303], [82.72%],
    [30], [0.0821], [83.16%],
  )
]

=== 可视化图表

以下是最佳实验（Exp 6）的训练过程可视化结果：

#figure(
  image("../../model/2025-12-25 13:47:21 UTC+8/training_loss.png", width: 80%),
  caption: "训练损失曲线"
)

#figure(
  image("../../model/2025-12-25 13:47:21 UTC+8/test_accuracy.png", width: 80%),
  caption: "测试准确率曲线"
)

从图中可以观察到：
- 训练损失从0.68平稳下降至0.04，模型持续学习
- 测试准确率在Epoch 7达到峰值87.91%后出现波动，但整体维持在80%以上
- 验证集准确率92.12%表明模型具有良好的泛化能力

= 程序调试

在项目开发过程中，我们遇到了以下问题及解决方法：

#align(center)[
  #table(
    columns: (auto, auto),
    [*问题描述*], [*解决方法*],
    [模型输入维度不匹配], [使用动态张量探测计算CNN输出维度],
    [训练结果不理想], [添加weight_decay=1e-3正则化],
  )
]

= 总结

== 技术清单

本次实践中使用到的技术、模型和算法：

- *深度学习框架*：PyTorch
- *神经网络架构*：CNN（卷积神经网络）+ MLP（多层感知机）
- *激活函数*：ReLU
- *池化操作*：平均池化（AvgPool）
- *优化器*：AdamW
- *损失函数*：交叉熵损失（CrossEntropyLoss）
- *数据增强*：图像Resize、ToTensor
- *前端框架*：PyQt6(客户端软件)以及HTML/JS/CSS(网页)
- *后端框架*：Flask
- *版本控制*：Git

== 问题与改进方案

*当前模型存在的问题：*

1. *泛化能力有限*：模型仅在特定数据集上训练，面对未知来源的Deepfake图像可能表现不佳。
2. *计算资源需求高*：6层CNN结构较深，在资源受限设备上运行较慢。
3. *缺乏数据增强*：仅使用基础图像缩放，未使用翻转、旋转等增强策略。

*后续改进方案：*

1. *引入预训练模型*：使用ResNet、VGG等预训练模型进行迁移学习，提升泛化能力。
2. *添加数据增强*：在`transforms`中加入随机翻转、随机旋转、颜色抖动等。
3. *模型轻量化*：采用知识蒸馏或模型剪枝技术，减小模型体积。
4. *多模型融合*：结合注意力机制或使用EfficientNet等高效架构。
5. *增加数据集多样性*：收集更多来源的Deepfake数据，覆盖更多生成方法。

== 心得体会

通过本次深度学习实践，我们有以下几点认识：

1. *理论与实践结合*：课堂所学的神经网络原理在实际编码中需要灵活运用，参数设置、模型设计都需要根据具体任务调整。

2. *数据质量决定上限*：再好的模型也需要高质量、丰富的数据支撑，数据预处理和增强是提升性能的关键。

3. *调试能力的重要性*：深度学习项目中很多问题难以通过阅读代码发现，需要借助日志、可视化等工具进行调试。

4. *团队协作的价值*：项目涉及数据处理、模型训练、后端开发、前端界面等多个环节，合理分工与协作能显著提升效率。

5. *持续迭代的思维*：模型优化是一个反复试验的过程，需要不断尝试新方法、分析结果、改进方案。

本次实践让我们亲身体验了从零开始构建深度学习应用的全过程，为后续学习和工作积累了宝贵经验。
