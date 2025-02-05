import torch
from torch import Tensor
from torch.nn import BCELoss
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)


class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=0.1):
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class Metrics:
    def __init__(self, device):
        metrics = {
            "acc": BinaryAccuracy().to(device),
            "precision": BinaryPrecision().to(device),
            "recall": BinaryRecall().to(device),
        }
        self.metrics = MetricCollection(metrics)

        self.bce_loss = BCELoss()
        self.iou_loss = IoULoss()

    def calc(
        self, pred: Tensor, target: Tensor, thr: float = 0.7, prefix: str = ""
    ) -> Tensor:
        bce_score = self.bce_loss(pred.float(), target.float())
        iou_score = self.iou_loss(pred.float(), target.float())

        pred = pred > thr
        pred = pred.to(dtype=torch.int8)
        target = target.to(dtype=torch.int8)

        result = self.metrics(pred, target)
        if pred.max() == 0 and target.max() == 0:
            for key in result.keys():
                result[key] = torch.tensor(
                    result[key], dtype=torch.float32, device=result[key].device
                )
        self.metrics.reset()
        result.update({"bce_loss": bce_score})
        result.update({"iou_loss": iou_score})

        if prefix:
            new_result = dict()
            for key in result.keys():
                new_result[prefix + "_" + key] = result[key]
            result = new_result
        return result
