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


def build_multilabel_loss(class_weighted: bool, train_labels: torch.Tensor, device) -> nn.Module:
    """BCE over independent per-disease heads. With class_weighted, each disease's positives are
    up-weighted by (negatives / positives) in train -- the standard pos_weight for BCE, so a
    disease present in ~5% of eyes isn't learned as "always say no". Weights are per disease,
    not shared, since prevalence differs a lot between them (DR ~27% vs AMD ~4%)."""
    if not class_weighted:
        return nn.BCEWithLogitsLoss()
    pos = train_labels.sum(dim=0)
    neg = train_labels.shape[0] - pos
    pos_weight = (neg / pos.clamp(min=1)).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
