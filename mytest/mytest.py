import torch
import torch.nn as nn
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from PIL import Image


class net(nn.Module):
    def __init__(self, inplanes, outplaes):
        super(net, self).__init__()
        self.conv0 = nn.Conv2d(3,16,kernel_size=1, stride=1, padding=1)
        self.conv1 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(16,16,kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16,3,kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x1 = self.pool(x1)
        x2 = self.conv2(x1)
        x2 = self.pool(x2)
        x3 = self.conv3(x2)
        x3 = self.pool(x3)
        x5 = self.conv5(x3)
        x5 = self.pool(x5)
        x6 = self.conv6(x3)
        x6 = self.pool(x6)
        x7 = self.conv7(x3)
        x7 = self.pool(x7)

        x1 = self.conv4(x1)
        x2 = self.conv4(x2)
        x3 = self.conv4(x3)
        x5 = self.conv4(x5)
        x6 = self.conv4(x6)
        x7 = self.conv4(x7)
        return x1, x2, x3,x5, x6, x7

if __name__ == "__main__":
    path = "yuan.jpg"
    torch.manual_seed(1133366)
    image = Image.open(path)
    image = np.array(image, dtype=np.float32)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    model = net(3,16)
    x1, x2, x3, x5,x6,x7 = model(image)
    x1 = x1[0,...].permute(1,2,0).detach().numpy()
    x2 = x2[0,...].permute(1,2,0).detach().numpy()
    x3 = x3[0,...].permute(1,2,0).detach().numpy()
    x5 = x5[0,...].permute(1,2,0).detach().numpy()
    x6 = x6[0,...].permute(1,2,0).detach().numpy()
    x7 = x7[0,...].permute(1,2,0).detach().numpy()

    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x5.shape)
    plt.imshow(x7)
    plt.show()