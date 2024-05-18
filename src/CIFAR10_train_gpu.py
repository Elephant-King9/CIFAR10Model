# 使用GPU对网络进行训练
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from CIFAR10model import *

# 创建数据集
train_dataset = torchvision.datasets.CIFAR10('../datasets', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10('../datasets', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

# 使用GPU进行训练，如果不支持就使用CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 创建DataLoader
train_dataLoader = DataLoader(dataset=train_dataset, batch_size=64)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64)

# 创建自定义的神经网络
model = CIFAR10model().to(device)

# 加载损失函数与梯度下降算法
loss_fn = nn.CrossEntropyLoss().to(device)
learn_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

# 定义TensorBoard
writer = SummaryWriter('../logs')

# 定义训练网络的时的变量
# 循环轮数
epoch = 30
# 总训练次数
total_train_step = 0

# 记录开始时间
start_time = time.time()

for i in range(epoch):
    print(f'-----------------------start {i+1} training-----------------------')
    # 更新模型为训练模式
    model.train()

    # 定义每轮的总训练损失
    pre_train_loss = 0.0
    # 定义每轮的训练次数
    pre_train_step = 0
    for data in train_dataLoader:
        inputs, labels = data
        # 使用GPU进行训练
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 优化器清零
        optimizer.zero_grad()

        # 根据模型将输入转化为输出
        outputs = model(inputs)

        # 计算损失，并且找到最大的下降梯度
        loss = loss_fn(outputs, labels)
        loss.backward()

        # 优化器进行梯度下降更新
        optimizer.step()

        # 参数更新，用于TensorBoard与print
        pre_train_loss += loss.item()
        pre_train_step += 1
        total_train_step += 1

        # 每循环100次输出一下
        if pre_train_step % 100 == 0:
            end_time = time.time()
            print(f'Epoch:{i + 1},Step:{pre_train_step},Loss:{pre_train_loss / pre_train_step},Time:{end_time - start_time}')

            # 更新到TensorBoard
            writer.add_scalar('train_loss', pre_train_loss / pre_train_step, total_train_step)
    print(f'-----------------------end {i + 1} training-----------------------')
    # 更新模型为测试模式
    model.eval()

    # 记录训练集中判断正确的个数
    test_accuracy = 0
    # 创建一个没有梯度下降的上下文环境，这么做的目的是可以降低执行的损耗
    print(f'-----------------------start {i + 1} testing-----------------------')
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # 计算训练集中正确的个数
            test_accuracy += outputs.argmax(1).eq(labels).sum().item()

    # 测试集验证结束
    print(f'Epoch:{i+1} Test Accuracy: {test_accuracy / len(test_dataset)}')

    # 添加正确率到TensorBoard
    writer.add_scalar('test_accuracy', test_accuracy/len(test_dataset), i)
    print(f'-----------------------end {i + 1} testing-----------------------')
    torch.save(model, f'../models/cifar10_model{i+1}.pth')
    print(f'-----------------------save {i + 1} model-----------------------')

end_time = time.time()
print(f'Finished_Time:{end_time - start_time}')
writer.close()



