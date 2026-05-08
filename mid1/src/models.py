import torch.nn as nn
from torchvision import models

from .attention import add_se_to_resnet


def build_resnet34(
    num_classes: int = 37,
    pretrained: bool = True,
    attention: str = "none",
    se_reduction: int = 16,
):
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet34(weights=weights)

    if attention == "se":
        model = add_se_to_resnet(model, reduction=se_reduction)
    elif attention == "none":
        pass
    else:
        raise ValueError(f"Unknown attention type: {attention}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_vit_tiny(
    num_classes: int = 37,
    pretrained: bool = False,
):
    import timm

    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )


def build_swin_t(
    num_classes: int = 37,
    pretrained: bool = False,
):
    weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_classes)
    return model


def build_model(cfg):
    model_cfg = cfg["model"]

    name = model_cfg["name"]
    if name == "resnet34":
        return build_resnet34(
            num_classes=model_cfg.get("num_classes", 37),
            pretrained=model_cfg.get("pretrained", True),
            attention=model_cfg.get("attention", "none"),
            se_reduction=model_cfg.get("se_reduction", 16),
        )

    if name == "vit_tiny":
        return build_vit_tiny(
            num_classes=model_cfg.get("num_classes", 37),
            pretrained=model_cfg.get("pretrained", False),
        )

    if name == "swin_t":
        return build_swin_t(
            num_classes=model_cfg.get("num_classes", 37),
            pretrained=model_cfg.get("pretrained", False),
        )

    raise ValueError(f"Unsupported model in mid1: {name}")
