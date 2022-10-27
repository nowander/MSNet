from torch import nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
from torch.autograd import Variable



model = models.vgg16_bn(pretrained=True)
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}



class vgg_left(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_left, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*24*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        A1_1 = self.features[:6](rgb)
        A1_2 = self.features[6:7](A1_1)
        A2_1 = self.features[7:13](A1_2)
        A2_2 = self.features[13:14](A2_1)
        A3_1 = self.features[14:23](A2_2)
        A3_2 = self.features[23:24](A3_1)
        A4_1 = self.features[24:33](A3_2)
        A4_2 = self.features[33:34](A4_1)
        A5_1 = self.features[34:43](A4_2)
        A5_2 = self.features[43:44](A5_1)
        return A1_1, A1_2, A2_1, A2_2, A3_1, A3_2, A4_1, A4_2, A5_1, A5_2

class vgg_right(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_right, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*24*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        A1_1 = self.features[:6](rgb)
        A1_2 = self.features[6:7](A1_1)
        A2_1 = self.features[7:13](A1_2)
        A2_2 = self.features[13:14](A2_1)
        A3_1 = self.features[14:23](A2_2)
        A3_2 = self.features[23:24](A3_1)
        A4_1 = self.features[24:33](A3_2)
        A4_2 = self.features[33:34](A4_1)
        A5_1 = self.features[34:43](A4_2)
        A5_2 = self.features[43:44](A5_1)
        return A1_1, A1_2, A2_1, A2_2, A3_1, A3_2, A4_1, A4_2, A5_1, A5_2

class vgg_left224(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_left224, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*24*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:
                    model_dict[k] = v
                    # print(k, v)

        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, rgb):
        A1 = self.features[:6](rgb)
        A2 = self.features[6:13](A1)
        A3 = self.features[13:23](A2)
        # A3 = self.features[13:23](rgb)
        # A4 = self.features[23:33](A3)
        # A5 = self.features[33:43](A4)
        # return A1, A2, A3, A4, A5
        return  A1, A2, A3


class vgg_right224(nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_right224, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # first model 224*224*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),  # [:6]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),  # second model 112*112*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # [6:13]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),  # third model 56*56*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),  # [13:23]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, 1, 1),  # forth model 28*28*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [13:33]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, 1, 1),  # fifth model 14*14*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),  # [33:43]
        )

        if pretrained:
            pretrained_vgg = model_zoo.load_url(model_urls['vgg16_bn'])
            model_dict = {}
            state_dict = self.state_dict()
            for k, v in pretrained_vgg.items():
                if k in state_dict:     # 1. filter out unnecessary keys
                    model_dict[k] = v   # 2. overwrite entries in the existing state dict
                    # print(k, v)
        state_dict.update(model_dict)    # 3. load the new state dict
        self.load_state_dict(state_dict)

    def forward(self, thermal):
        A1_d = self.features[:6](thermal)
        A2_d = self.features[6:13](A1_d)
        A3_d = self.features[13:23](A2_d)
        # A3_d = self.features[13:23](thermal)
        # A4_d = self.features[23:33](A3_d)
        # A5_d = self.features[33:43](A4_d)
        # return A1_d, A2_d, A3_d, A4_d, A5_d
        return  A1_d, A2_d, A3_d
