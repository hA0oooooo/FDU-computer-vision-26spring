import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        hidden = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weight = self.pool(x)
        weight = self.fc(weight)
        return x * weight


class SEWrapper(nn.Module):
    def __init__(self, block: nn.Module, channels: int, reduction: int = 16):
        super().__init__()
        self.block = block
        self.se = SEBlock(channels, reduction)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out


def add_se_to_resnet(model: nn.Module, reduction: int = 16):
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        wrapped_blocks = []

        for block in layer:
            channels = block.conv2.out_channels
            wrapped_blocks.append(SEWrapper(block, channels, reduction))

        setattr(model, layer_name, nn.Sequential(*wrapped_blocks))

    return model
