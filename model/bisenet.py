import warnings

import torch
from torch import nn
from torch.nn.functional import interpolate
from torchvision.models import resnet101, resnet18

from model.backbone import Resnet

warnings.filterwarnings(action='ignore')


class ConvBlock(torch.nn.Module):
    """
    Standard [Conv + Batch Norm + ReLu] block implementation.

    Args:
        in_channels (int): Number of channels in inputs image.
        out_channels (int): Number of output feature maps.
        kernel_size (int, optional): Size of the convolving kernel. Default: 3.
        stride (int, optional): Stride of the convolution. Default: 2.
        padding (int, optional): Zero-padding added to both sides of the inputs. Default: 1.

    Attributes:
        conv1 (nn.Conv2d): Convolutional layer of the block
        bn (nn.BatchNorm2d): Batch normalization layer of the block
        relu (nn.ReLu): Activation layer of the block
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        return self.relu(self.bn(x))


class Spatial_path(torch.nn.Module):
    # TODO: Parametrize in/out channels of the conv-blocks
    """
    Spatial Path model branch. This path is supposed to encode "rich spatial information".
    It is made up of 3 convolutional blocks, which in total will resize the inputs image to 1/8 of it's original size.

    Args:

    Attributes:
        conv_block1 (ConvBlock): First convolutional block of the Spatial Path
        conv_block2 (ConvBlock): Second convolutional block of the Spatial Path
        conv_block3 (ConvBlock): Third (and last) convolutional block of the Spatial Path
    """

    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, inputs):
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return x


class AttentionRefinementModule(torch.nn.Module):
    """
    Custom module to refine the features from the Context Path.
    It employs Global Average Pooling to capture the Global context, and is applied at certain layers of the context
    path.
    It outputs an attention vector to guide the feature learning.

    Args:
        in_channels (int): Number of input feature maps.
        out_channels (int): Number of output feature maps.

    Attributes:
        in_channels (int): Number of input feature maps.
        avg_pool (nn.AdaptiveAvgPool2d): Global Average Pooling layer.
        conv (nn.Conv2d): 1x1 Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        sigmoid (nn.Sigmoid): Sigmoid Activation layer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # global average pooling
        x = self.avg_pool(inputs)

        # input and x channels should be same
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))

        x = self.conv(x)
        x = self.sigmoid(x)

        x = torch.mul(inputs, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    """
    Custom module to allow feature fusion coming from Context Path and Spatial Path.
    Since the features coming from these 2 modules have different representation (come from 2 different "models"), we
    need a way to combine them. This module is the way described in the paper.

    Args:
        num_classes (int): Number of target classes for the task at hand.
        in_channels (int): Number of total input channels, coming from both Spatial and Context Path.

    Attributes:
        in_channels (int): Number of total input channels, coming from both Spatial and Context Path.
        avg_pool (nn.AdaptiveAvgPool2d): Global Average Pooling layer.
        conv_block (ConvBlock): Convolutional block (convolutional + batch norm + ReLu).
        conv1 (nn.Conv2d): First 1x1 Convolutional Layer.
        relu (nn.ReLU): First Activation layer (ReLu).
        conv2 (nn.Conv2d): Second 1x1 Convolutional Layer.
        sigmoid (nn.ReLU): Second Activation layer (Sigmoid).
    """

    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_block = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_1, inputs_2):
        x = torch.cat((inputs_1, inputs_2), dim=1)

        # input and x channels should be same
        assert self.in_channels == x.size(1), f'in_channels of ConvBlock should be {x.size(1)}'
        feature = self.conv_block(x)

        x = self.avg_pool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    """
    BiSeNet implementation.

    Args:
        num_classes (int): Number of target classes for the task at hand.
        backbone (str): Name of the model to use for the backbone of the Context Path.
    """

    def __init__(self, num_classes, backbone):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path()

        # build context path
        self.context_path = get_context_path(name=backbone)

        # build attention refinement module  for resnet 101
        if backbone == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        # build attention refinement module  for resnet 18
        elif backbone == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):

        # output of spatial path
        sx = self.saptial_path(inputs)
        # output of context path
        cx1, cx2, tail = self.context_path(inputs)

        # ARM 1
        cx1 = self.attention_refinement_module1(cx1)
        # ARM 2
        cx2 = self.attention_refinement_module2(cx2)

        cx2 = torch.mul(cx2, tail)

        # upsampling
        cx1 = torch.nn.functional.interpolate(cx1, size=sx.size()[-2:], mode='bilinear')
        cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx = torch.cat((cx1, cx2), dim=1)

        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        if self.training:
            # ??? Page 7 "loss function"
            cx1_sup = self.supervision1(cx1)
            cx2_sup = self.supervision2(cx2)
            # ???
            cx1_sup = torch.nn.functional.interpolate(cx1_sup, size=inputs.size()[-2:], mode='bilinear')
            cx2_sup = torch.nn.functional.interpolate(cx2_sup, size=inputs.size()[-2:], mode='bilinear')

            return result, cx1_sup, cx2_sup
        else:
            return result


def get_context_path(name):
    """
    Builds the Context path model
    Args:
        name: name of the backbone architecture to return (resnet18 and resnet101 are the only supported arch.)

    Returns:
        Context path model
    """
    model = {
        'resnet18': Resnet(resnet18(pretrained=True)),
        'resnet101': Resnet(resnet101(pretrained=True))
    }
    return model[name]
