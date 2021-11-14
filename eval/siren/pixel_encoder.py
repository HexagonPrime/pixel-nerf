"""
Implements image encoders
"""
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from siren import util
from siren.custom_encoder import ConvEncoder
import torch.autograd.profiler as profiler


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class ImageEncoder(nn.Module):
    """
    Global image encoder
    """

    def __init__(self, backbone="resnet34", pretrained=True, z_dim=512):
        """
        :param backbone Backbone network. Assumes it is resnet*
        e.g. resnet34 | resnet50
        :param pretrained Whether to use model pretrained on ImageNet
        """
        super().__init__()
        self.model = getattr(torchvision.models, backbone)(pretrained=pretrained)
        self.model.fc = nn.Sequential()
        # self.register_buffer("latent", torch.empty(1, 1), persistent=False)
        # self.latent (B, L)
        self.z_dim = z_dim
        self.fc = nn.Linear(512, z_dim+2)

    def index(self, uv, cam_z=None, image_size=(), z_bounds=()):
        """
        Params ignored (compatibility)
        :param uv (B, N, 2) only used for shape
        :return latent vector (B, L, N)
        """
        return self.latent.unsqueeze(-1).expand(-1, -1, uv.shape[1])

    def forward(self, x, alpha):
        """
        For extracting ResNet's features.
        :param x image (B, C, H, W)
        :return latent (B, z_dim)
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x[:, :self.z_dim], x[:, self.z_dim:self.z_dim+2]