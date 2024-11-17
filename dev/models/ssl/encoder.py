import torch.nn as nn
from torchvision.models import resnet18

class ResNetEncoder(nn.Module):
    """
    This class implements a ResNet-18-based encoder that is used as a core building block in deep learning models, especially in tasks like self-supervised learning (SSL). 
    Here, PyTorch’s ResNet-18 architecture is customized by removing the fully connected layer, turning it into a feature extractor.
    """
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        resnet = resnet18(pretrained=True) # Load the pre-trained ResNet-18 model on ImageNet.
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Without the final fully connected layer
        self.output_dim = resnet.fc.in_features  # Save output dimension

    def forward(self, x):
        x = self.resnet(x) # x is a batch of images with shape [batch_size, channels, height, width].
        return x.view(x.size(0), -1) # Reorganizes the output into a two-dimensional tensor