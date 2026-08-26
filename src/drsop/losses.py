import torch
import torch.nn as nn


def class_weights_from_counts(counts: list, device) -> torch.Tensor:
    counts = torch.tensor(counts, dtype=torch.float32, device=device)
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    return weights


def build_loss(class_weighted: bool, train_labels, num_classes: int, device) -> nn.Module:
    if not class_weighted:
        return nn.CrossEntropyLoss()
    counts = [(train_labels == c).sum() for c in range(num_classes)]
    weights = class_weights_from_counts(counts, device)
    return nn.CrossEntropyLoss(weight=weights)
