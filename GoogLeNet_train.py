# -*- coding= utf-8 -*-
# @Time : 2023/3/31 17:23
# @Author : 尘小风
# @File : GoogLeNet_train.py
# @software : PyCharm

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from GoogLeNet_model import Model

batch_size = 32
transform = {
    "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    "test":transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}


# 获得数据集的路径
data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))
data_path = data_root + "/dataset/SIPaKMeD/"

# 加载训练集
train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=transform["train"])
train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size)
# 加载测试集
test_dataset= datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=transform["test"])
test_loader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=batch_size)

print("using {} images for training, {} images for test".format(len(train_dataset), len(test_dataset)))


model = Model(num_classes=5, aux_logits=True, init_weights=True)
# 如果要使用官方的预训练权重，注意是将权重载入官方的模型，不是我们自己实现的模型
# 官方的模型中使用了bn层以及改了一些参数，不能混用
# import torchvision
# net = torchvision.models.googlenet(num_classes=5)
# model_dict = net.state_dict()
# # 预训练权重下载地址: https://download.pytorch.org/models/googlenet-1378be20.pth
# pretrain_model = torch.load("googlenet.pth")
# del_list = ["aux1.fc2.weight", "aux1.fc2.bias",
#             "aux2.fc2.weight", "aux2.fc2.bias",
#             "fc.weight", "fc.bias"]
# pretrain_dict = {k: v for k, v in pretrain_model.items() if k not in del_list}
# model_dict.update(pretrain_dict)
# net.load_state_dict(model_dict)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

loss_list = []
accuracy_list = []
best_acc = 0
for epoch in range(10):
    model.train()
    loss_sum = 0
    for i, (inputs, targets) in enumerate(train_loader):
        logits, logits_aux2, logits_aux1 = model(inputs)
        loss0 = loss_function(logits, targets)
        loss1 = loss_function(logits_aux1, targets)
        loss2 = loss_function(logits_aux2, targets)
        loss = loss0 + 0.3 * loss1 + 0.3 * loss2

        loss_sum += loss.item()
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rate = (i+1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "-" * int(50 * (1-rate))
        print("\rtrain loss:{:.3f}% [{}->{}] {:.3f}".format(rate*100, a, b, loss), end="")
    print()


    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            y_pred = model(images)
            _,predicted = torch.max(y_pred.data, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy_list.append(correct / total)

        print("[epoch %d], train_loss:%.3f, test_accuracy:%.3f%% " % (epoch+1, loss_sum / (i+1), 100 * correct / total))

        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), "GoogLeNet.pth")


print("Finishing Training")
plt.subplot(121)
plt.plot(range(len(loss_list)), loss_list)
plt.xlabel("step")
plt.ylabel("loss")
plt.subplot(122)
plt.plot(range(epoch + 1), accuracy_list)
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()


