import sys
from pathlib import Path

import torch
import torch.nn as nn


class RetfoundEncoder(nn.Module):
    """Wraps the official RETFound_MAE ViT, loads pretrained CFP weights,
    and projects the pooled image representation to `proj_dim`.

    Requires the RETFound_MAE repo cloned at `repo_path` (see scripts/setup_retfound.sh)
    since its models_vit.py / position-embedding interpolation isn't pip-installable.
    """

    def __init__(self, repo_path: str, checkpoint_path: str, arch: str,
                 proj_dim: int, freeze: bool = True):
        super().__init__()
        repo_path = str(Path(repo_path).resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import models_vit  # noqa: E402  (provided by the vendored RETFound_MAE repo)
        from util.pos_embed import interpolate_pos_embed  # noqa: E402

        self.backbone = models_vit.__dict__[arch](
            num_classes=0,  # no classification head; we want the pooled embedding
            drop_path_rate=0.0,
            global_pool=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        state_dict = self.backbone.state_dict()
        for k in ("head.weight", "head.bias"):
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, torch.empty(0)).shape:
                del checkpoint_model[k]
        interpolate_pos_embed(self.backbone, checkpoint_model)
        missing, unexpected = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(f"[RetfoundEncoder] missing={len(missing)} unexpected={len(unexpected)} keys on load")

        embed_dim = self.backbone.embed_dim if hasattr(self.backbone, "embed_dim") else 1024
        self.proj = nn.Linear(embed_dim, proj_dim)

        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                feats = self.backbone.forward_features(images)
        else:
            feats = self.backbone.forward_features(images)
        return self.proj(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # keep frozen backbone (BatchNorm/dropout) in eval mode always
        return self
