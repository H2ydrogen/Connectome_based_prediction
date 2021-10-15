import math

from torchvision import models
from torch import nn
import torch.nn.functional as F
import torch


def Alexnet(opt):
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(555, opt.NUM_CLASSES)
    return model


def ResNet50(classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(2048, classes)
    return model


def DenseNet169(classes, pretrained=True):
    model = models.densenet169(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Linear(1664, classes)
    return model


class CNN(nn.Module):
    def __init__(self, n_features, n_output):
        super(CNN, self).__init__()
        self.layer1 = nn.Linear(n_features, 1024)
        self.layer2 = nn.Linear(1024, 2048)
        self.layer3 = nn.Linear(2048, 4096)
        self.layer4 = nn.Linear(4096, 4096)
        self.output = nn.Linear(4096, n_output)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer4(x))
        x = self.output(x)
        return x


class HARmodel(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, args):
        super().__init__()
        self.opt = args
        # 计算全连接层输入维度
        input_len = 1516
        self.classifier_input_len = 64 * (input_len - 12)
        # Extract features, 1D conv layers
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, 5) if self.opt.ensemble else nn.Conv1d(len(self.opt.INPUT_FEATURES), 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm1d(64, momentum=0.5),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.classifier_input_len, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, self.opt.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out


class CNN_2D(nn.Module):
    def __init__(self, opt):
        '''构造函数，定义网络的结构'''
        super().__init__()
        self.opt = opt
        input_len = 800 * len(self.opt.HEMISPHERES)
        if 'anatomical' in self.opt.HEMISPHERES:
            input_len -= (800 - 57)
        input_len = math.ceil(input_len**0.5)
        self.classifier_input_len = 64*(input_len-12)**2

        self.features = nn.Sequential(
            nn.Conv2d(len(self.opt.INPUT_FEATURES), 64, 5),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(self.classifier_input_len, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, self.opt.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class Lenet(nn.Module):
    def __init__(self, opt):
        '''构造函数，定义网络的结构'''
        super().__init__()
        self.opt = opt
        input_len = 800 * len(self.opt.HEMISPHERES)
        if 'anatomical' in self.opt.HEMISPHERES:
            input_len -= (800 - 57)
        input_len = math.ceil(input_len**0.5)
        self.classifier_input_len = 64*(input_len-12)**2

        self.features = nn.Sequential(
            nn.Conv2d(len(self.opt.INPUT_FEATURES), 6, 5, padding=3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.BatchNorm2d(16, momentum=0.5),
            nn.Conv2d(16, 120, 5),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(84, self.opt.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


class new_CNN(nn.Module):  # 1.5D-CNN
    def __init__(self, opt):
        '''构造函数，定义网络的结构'''
        super().__init__()
        self.opt = opt

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(64, momentum=0.5),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(51072, 128),
            nn.BatchNorm1d(128, momentum=0.5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, self.opt.NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out