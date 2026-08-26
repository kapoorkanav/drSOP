import torch
import torch.nn as nn

from drsop.models.gate import GateTransformer
from drsop.models.head import MLPHead
from drsop.models.metadata_transformer import MetadataEncoder
from drsop.models.retfound_encoder import RetfoundEncoder


class DRFusionModel(nn.Module):
    def __init__(self, retfound_cfg: dict, meta_cfg: dict, gate_cfg: dict, head_cfg: dict,
                 categorical_cardinalities: list, n_numeric: int, n_comorbidities: int,
                 proj_dim: int, num_classes: int):
        super().__init__()
        self.image_encoder = RetfoundEncoder(
            repo_path=retfound_cfg["retfound_repo"],
            checkpoint_path=retfound_cfg["retfound_checkpoint"],
            arch=retfound_cfg["retfound_arch"],
            proj_dim=proj_dim,
            freeze=retfound_cfg["freeze_retfound"],
        )
        self.meta_encoder = MetadataEncoder(
            n_numeric=n_numeric,
            categorical_cardinalities=categorical_cardinalities,
            n_comorbidities=n_comorbidities,
            proj_dim=proj_dim,
            **meta_cfg,
        )
        self.gate = GateTransformer(dim=proj_dim, **gate_cfg)
        self.head = MLPHead(in_dim=proj_dim, num_classes=num_classes, **head_cfg)

    def forward(self, batch: dict, return_alpha: bool = False):
        image_emb = self.image_encoder(batch["image"])
        meta_emb = self.meta_encoder(batch["numeric"], batch["categorical"], batch["comorbidity"])
        alpha = self.gate(image_emb, meta_emb)
        fused = alpha * image_emb + (1 - alpha) * meta_emb
        logits = self.head(fused)
        if return_alpha:
            return logits, alpha
        return logits
