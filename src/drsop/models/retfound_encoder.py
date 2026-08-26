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
        # weights_only=False: PyTorch >=2.6 defaults to weights_only=True, which rejects this
        # checkpoint (it has an argparse.Namespace of the original authors' training args
        # pickled inside). Safe here since the checkpoint is from the official RETFound release.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_model = checkpoint["model"]
        # Match the key naming used by the current RETFound_MAE loading code (main_finetune.py) --
        # older checkpoints/code used different key names for these.
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}

        state_dict = self.backbone.state_dict()
        for k in ("head.weight", "head.bias"):
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, torch.empty(0)).shape:
                del checkpoint_model[k]
        interpolate_pos_embed(self.backbone, checkpoint_model)
        missing, unexpected = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(f"[RetfoundEncoder] missing={len(missing)} unexpected={len(unexpected)} keys on load")
        if missing:
            print(f"[RetfoundEncoder] missing keys (first 10): {missing[:10]}")
        if unexpected:
            print(f"[RetfoundEncoder] unexpected keys (first 10): {unexpected[:10]}")
        # Only head.weight/head.bias (deleted above, num_classes=0) and fc_norm.weight/bias
        # should ever be legitimately missing. fc_norm is the post-global-pool LayerNorm used
        # when global_pool=True; it never existed during MAE pretraining (which only used the
        # CLS-token `norm`), so it's always freshly initialized on every RETFound fine-tune --
        # not specific to us. It stays at PyTorch's default LayerNorm init (weight=1, bias=0,
        # i.e. plain normalization, not noise), and the trainable `proj` layer right after it
        # can absorb any scale/shift it needs anyway. Anything beyond these four keys missing
        # means the checkpoint's keys didn't actually match the model -- most of the pretrained
        # weights silently failed to load (strict=False doesn't raise on this by itself).
        expected_missing = ("head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias")
        unexpected_missing = [k for k in missing if k not in expected_missing]
        if unexpected_missing:
            raise RuntimeError(
                f"RetfoundEncoder: {len(unexpected_missing)} unexpected missing keys after loading "
                f"the checkpoint -- pretrained weights likely did NOT load correctly. "
                f"First few: {unexpected_missing[:10]}"
            )

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
        # This RETFound_MAE version's forward_features uses mean(dim=1, keepdim=True) when
        # global_pool=True, returning [B, 1, embed_dim] instead of [B, embed_dim].
        if feats.dim() == 3:
            feats = feats.squeeze(1)
        return self.proj(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()  # keep frozen backbone (BatchNorm/dropout) in eval mode always
        return self
