import torch
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, 2, padding=3, bias=False)
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, 1, padding=0, bias=False)
        self.conv5 = nn.Conv2d(256, 256, 3, 2, padding=1, bias=False)
        self.conv6 = nn.Conv2d(256, 512, 1, padding=0, bias=False)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 100)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.max_pool(x)
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x