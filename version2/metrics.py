from torchmetrics import Metric
import torch
import numpy as np

class LossMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("batches", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, loss):
        self.loss += loss
        self.batches += 1

    def compute(self):
        return (self.loss / self.batches).item()

class MeanPixelAccuracy(Metric):
    def __init__(self, num_classes=3):
        super().__init__()
        self.add_state("pixel_accuracy", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        if target.dim() == 4:
            target = target.squeeze(1)
        assert preds.shape == target.shape
        preds, target = preds.cpu(), target.cpu()
        self.pixel_accuracy += torch.sum((preds == target))
        self.total_pixels += target.numel()

    def compute(self):
        return (self.pixel_accuracy / self.total_pixels / self.num_classes).item()


class MeanIoU(Metric):
    def __init__(self, num_classes=3):
        super().__init__()
        self.add_state("IoU", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="mean")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        if target.dim() == 4:
            target = target.squeeze(1)
        assert preds.shape == target.shape
        assert preds.dim() == 3 and target.dim() == 3
        preds, target = preds.cpu(), target.cpu()

        pixels = preds.shape[1] * preds.shape[2]
        bs = preds.shape[0]
        cm = np.zeros((bs, self.num_classes, self.num_classes))
        batches = torch.tile(torch.arange(bs).unsqueeze(1), (1, pixels)).numpy().flatten()
        targets = target.numpy().astype(np.int32).flatten()
        predictions = preds.numpy().astype(np.int32).flatten()
        np.add.at(cm, (batches, targets, predictions), 1)

        tp = cm[:, torch.arange(self.num_classes), torch.arange(self.num_classes)]
        fn = cm.sum(axis=1) - tp
        positives = cm.sum(axis=2)

        denominator = (positives + fn)
        denominator[denominator == 0] = 1

        self.IoU += (tp / denominator).sum() / self.num_classes
        self.samples += bs

    def compute(self):
        return (self.IoU / self.samples).item()


class FrequencyIoU(Metric):
    def __init__(self, num_classes=3):
        super().__init__()
        self.add_state("IoU", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="mean")
        self.add_state("samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_classes = num_classes

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        if target.dim() == 4:
            target = target.squeeze(1)
        assert preds.shape == target.shape
        assert preds.dim() == 3 and target.dim() == 3
        preds, target = preds.cpu(), target.cpu()

        pixels = preds.shape[1] * preds.shape[2]
        bs = preds.shape[0]
        cm = np.zeros((bs, self.num_classes, self.num_classes))
        batches = torch.tile(torch.arange(bs).unsqueeze(1), (1, pixels)).numpy().flatten()
        targets = target.numpy().astype(np.int32).flatten()
        predictions = preds.numpy().astype(np.int32).flatten()
        np.add.at(cm, (batches, targets, predictions), 1)

        tp = cm[:, torch.arange(self.num_classes), torch.arange(self.num_classes)]
        fn = cm.sum(axis=1) - tp
        positives = cm.sum(axis=2)
        fq = positives / pixels

        denominator = (positives + fn)
        denominator[denominator == 0] = 1

        self.IoU += (tp * fq / denominator).sum()
        self.samples += bs


    def compute(self):
        return (self.IoU / self.samples).item()
