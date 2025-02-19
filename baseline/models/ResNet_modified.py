import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
from torchvision.models import ResNet50_Weights


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.extractor = nn.Sequential(
            self._make_layer(in_dim=3, out_dim=32, kernel_size=3, stride=1, padding=1, bias=True),
            self._make_layer(in_dim=32, out_dim=32, kernel_size=1, stride=1, padding=0, bias=True),
            self._make_layer(in_dim=32, out_dim=3, kernel_size=3, stride=1, padding=1, bias=True)
        )
        resnet50 = resnet.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        resnet_children = list(resnet50.children())
        self.enhancer = nn.Sequential(*resnet_children[:-1])
        self.head = nn.Linear(in_features=2048, out_features=1, bias=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.head.weight)
        nn.init.constant_(self.head.bias, 0)

    def _make_layer(self, in_dim: int, out_dim: int, kernel_size: int, stride: int, padding: int
                    , bias: bool) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x) + x
        x = self.enhancer(x)
        mos = self.head(torch.flatten(x, 1))
        return mos

