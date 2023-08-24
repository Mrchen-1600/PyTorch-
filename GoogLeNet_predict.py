# -*- coding= utf-8 -*-
# @Time : 2023/3/31 20:09
# @Author : 尘小风
# @File : GoogLeNet_predict.py
# @software : PyCharm

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from GoogLeNet_model import Model

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img = Image.open("im_Dyskeratotic.bmp")
plt.imshow(img)
img = transform(img)
img = torch.unsqueeze(img, dim=0)

model = Model(num_classes=5, aux_logits=False)

missing_keys, unexpected_keys = model.load_state_dict(torch.load("GoogLeNet.pth"), strict=False)
classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial_Intermediate"]
model.eval()
with torch.no_grad():
    output = model(img)

    predict = torch.softmax(output, dim=1).numpy() # torch.softmax返回的是一个张量形式的二维矩阵，通过.numpy()转换成数组形式的二维矩阵
    print(predict)
    predict_cla = torch.argmax(output).item() # 把张量形式的值取出来
    # print(predict_cla)
    print("img is predicted as %s, accuracy is %.3f" % (classes[predict_cla], predict[0][predict_cla]))

plt.show()




