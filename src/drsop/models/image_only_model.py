import torch
import torch.nn as nn

from drsop.models.head import MLPHead
from drsop.models.retfound_encoder import RetfoundEncoder


class ImageOnlyModel(nn.Module):
    """RETFound + LoRA straight into the MLP head. No metadata, no fusion gate.

    This is the control the fusion architecture is measured against. Every result so far
    compares fusion variants to each OTHER, which cannot show whether reading the metadata at
    all is worth anything -- if this model matches DRFusionModel, then the metadata encoder and
    the gate are decoration and the honest conclusion is that the image carries the signal.

    Same backbone and same LoRA configuration as DRFusionModel, same head shape, same loss and
    schedule. What it drops is the whole metadata path: MetadataEncoder, GateTransformer, and --
    with proj_dim=None -- the 1024 -> 512 projection that only existed so the gate could add the
    image and metadata embeddings elementwise. With nothing to match widths against, that
    projection is a pure bottleneck, so the backbone's native 1024-d embedding goes straight
    into the head.

    Note what that costs the comparison: against a fusion run this model differs in two ways,
    not one, so a score gap cannot be attributed to the metadata path alone. Set proj_dim to 512
    in the config to keep the bottleneck and isolate the metadata path instead. The default here
    answers the blunter and more useful question -- how well does the image alone do?

    Exposes DRFusionModel's interface so train.py, evaluate.py and the analysis scripts work
    against it unchanged. `return_alpha=True` yields an all-ones alpha, which is not a
    placeholder but the literal truth for this model: the gate equation
    fused = alpha * image_emb + (1 - alpha) * meta_emb reduces to fused = image_emb exactly
    when alpha = 1.
    """

    def __init__(self, retfound_cfg: dict, head_cfg: dict, proj_dim, num_classes: int):
        super().__init__()
        lora_cfg = retfound_cfg.get("lora", {})
        self.image_encoder = RetfoundEncoder(
            repo_path=retfound_cfg["retfound_repo"],
            checkpoint_path=retfound_cfg["retfound_checkpoint"],
            arch=retfound_cfg["retfound_arch"],
            proj_dim=proj_dim,  # None -> no projection; the head takes the native width
            freeze=retfound_cfg["freeze_retfound"],
            use_lora=retfound_cfg.get("use_lora", False),
            lora_r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.1),
        )
        self.head = MLPHead(in_dim=self.image_encoder.out_dim, num_classes=num_classes,
                            **head_cfg)
        # Read by scripts that branch on the gate's missingness token; there is no gate here.
        self.use_missingness_token = False

    def forward(self, batch: dict, return_alpha: bool = False):
        # The dataloader still yields the metadata tensors; this model simply never reads them.
        image_emb = self.image_encoder(batch["image"])
        logits = self.head(image_emb)
        if return_alpha:
            return logits, torch.ones_like(image_emb)
        return logits
