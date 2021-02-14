import torch
import torch.nn as nn
import torch.nn.functional as F

# Taking the model from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py


class AlexNet(nn.Module):

    def __init__(self, num_classes=2, param_dim=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256 + param_dim, num_classes)

    def forward(self, img, param):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_full = torch.cat((x, param), 1)
        x = F.softmax(self.fc3(x_full), dim=1)
        return x


class MLP(nn.Module):

    def __init__(self, num_classes=2, param_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(20 * 20, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128 + param_dim, num_classes)

    def forward(self, img, param):
        x = img.view(img.size(0), 20 * 20)
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), training=self.training)
        x_full = torch.cat((x, param), 1)
        x = F.softmax(self.fc3(x_full), dim=1)
        return x
