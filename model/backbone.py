import torch
from torchvision.models import resnet18, resnet101


class Resnet(torch.nn.Module):
    """
    Builds resnet backbone for the context path
    """
    def __init__(self, features):
        """

        Args:
            features: weights of a resnet model from pytorch (typically pretrained)
        """
        super(Resnet, self).__init__()
        self.features = features
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.max_pool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(self.bn1(x))
        x = self.max_pool1(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def build_context_path(name):
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
