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
        lora_cfg = retfound_cfg.get("lora", {})
        self.image_encoder = RetfoundEncoder(
            repo_path=retfound_cfg["retfound_repo"],
            checkpoint_path=retfound_cfg["retfound_checkpoint"],
            arch=retfound_cfg["retfound_arch"],
            proj_dim=proj_dim,
            freeze=retfound_cfg["freeze_retfound"],
            use_lora=retfound_cfg.get("use_lora", False),
            lora_r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.1),
        )
        self.meta_encoder = MetadataEncoder(
            n_numeric=n_numeric,
            categorical_cardinalities=categorical_cardinalities,
            n_comorbidities=n_comorbidities,
            proj_dim=proj_dim,
            **meta_cfg,
        )
        # Copied, not mutated: the caller's cfg dict also gets saved into checkpoints.
        gate_cfg = dict(gate_cfg)
        self.use_missingness_token = gate_cfg.pop("use_missingness_token", False)
        # one flag per numeric field, one per categorical field, one for comorbidities
        n_missing_flags = (n_numeric + len(categorical_cardinalities) + 1
                            if self.use_missingness_token else 0)
        self.gate = GateTransformer(dim=proj_dim, n_missing_flags=n_missing_flags, **gate_cfg)
        self.head = MLPHead(in_dim=proj_dim, num_classes=num_classes, **head_cfg)

    def forward(self, batch: dict, return_alpha: bool = False):
        image_emb = self.image_encoder(batch["image"])
        meta_emb = self.meta_encoder(
            batch["numeric"], batch["numeric_missing"],
            batch["categorical"], batch["comorbidity"], batch["comorbidity_missing"],
        )
        missing_flags = None
        if self.use_missingness_token:
            missing_flags = torch.cat([
                batch["numeric_missing"],
                batch["categorical_missing"],
                batch["comorbidity_missing"].unsqueeze(-1),
            ], dim=-1)
        alpha = self.gate(image_emb, meta_emb, missing_flags)
        fused = alpha * image_emb + (1 - alpha) * meta_emb
        logits = self.head(fused)
        if return_alpha:
            return logits, alpha
        return logits
