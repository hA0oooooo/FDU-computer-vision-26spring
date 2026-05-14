import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        cardinality = torch.sum(probs + target_one_hot, dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def build_loss(cfg):
    loss_cfg = cfg.get("loss", {})
    model_cfg = cfg.get("model", {})
    name = loss_cfg.get("name", "ce").lower()
    num_classes = model_cfg.get("num_classes", 3)

    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss(num_classes=num_classes, eps=loss_cfg.get("eps", 1e-6))
    if name in {"ce_dice", "cedice", "ce+dice"}:
        return CEDiceLoss(
            num_classes=num_classes,
            ce_weight=loss_cfg.get("ce_weight", 1.0),
            dice_weight=loss_cfg.get("dice_weight", 1.0),
        )
    raise ValueError(f"Unknown loss: {name}")

