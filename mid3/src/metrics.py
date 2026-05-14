import numpy as np
import torch


CLASS_NAMES = ["foreground", "background", "boundary"]


class SegmentationMetric:
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, logits_or_pred, target):
        if logits_or_pred.ndim == 4:
            pred = logits_or_pred.argmax(dim=1)
        else:
            pred = logits_or_pred

        pred_np = pred.detach().cpu().numpy().reshape(-1)
        target_np = target.detach().cpu().numpy().reshape(-1)
        valid = (target_np >= 0) & (target_np < self.num_classes)
        hist = np.bincount(
            self.num_classes * target_np[valid].astype(int) + pred_np[valid].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)
        self.confusion += hist

    def compute(self):
        cm = self.confusion
        total = cm.sum()
        pixel_acc = np.diag(cm).sum() / total if total > 0 else 0.0

        intersection = np.diag(cm).astype(np.float64)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        miou = float(iou.mean())

        return {
            "pixel_acc": float(pixel_acc),
            "foreground_iou": float(iou[0]),
            "background_iou": float(iou[1]),
            "boundary_iou": float(iou[2]),
            "mIoU": miou,
            "per_class_iou": iou.tolist(),
        }


@torch.no_grad()
def compute_batch_metrics(logits, target, num_classes: int = 3):
    metric = SegmentationMetric(num_classes=num_classes)
    metric.update(logits, target)
    return metric.compute()

