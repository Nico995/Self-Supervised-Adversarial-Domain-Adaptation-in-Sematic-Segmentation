import torch
from torch.nn import Sequential, Conv2d, LeakyReLU, AdaptiveAvgPool2d


class Discriminator(torch.nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.ndf = ndf

        self.conv1 = Conv2d(num_classes, ndf, kernel_size=4, stride=3, padding=1)
        self.relu1 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = Conv2d(ndf, ndf * 2, kernel_size=4, stride=3, padding=1)
        self.relu2 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv3 = Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=3, padding=1)
        self.relu3 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv4 = Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=3, padding=1)
        self.relu4 = LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv5 = Conv2d(ndf * 8, 1, kernel_size=4, stride=3, padding=1)
        self.avgpool = AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], 1)
        return x
