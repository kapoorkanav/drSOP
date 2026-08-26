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
                 proj_dim: int, freeze: bool = True, use_lora: bool = False,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
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
        self.use_lora = use_lora
        if use_lora:
            # LoRA: base weights stay frozen (get_peft_model does this automatically), only
            # small adapter matrices injected into attention/MLP linears become trainable --
            # plus fc_norm via modules_to_save, since it was never pretrained anyway (see the
            # comment above on expected_missing) and there's no pretrained value to preserve.
            # freeze_retfound is ignored in this branch: LoRA always keeps the base frozen.
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                # Matches blocks.<i>.attn.{qkv,proj} and blocks.<i>.mlp.{fc1,fc2} only --
                # explicitly NOT patch_embed.proj (a Conv2d; plain LoRA only supports Linear).
                target_modules=r"^blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(fc1|fc2))$",
                modules_to_save=["fc_norm"],
                bias="none",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            self.backbone.print_trainable_parameters()
        elif freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        # else: freeze=False, use_lora=False -> full fine-tune, all backbone params trainable.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # get_peft_model wraps self.backbone in a PeftModel; get_base_model() is peft's public,
        # version-stable way back to the original module for calling custom methods like
        # forward_features (which peft doesn't know about and doesn't proxy reliably).
        vit = self.backbone.get_base_model() if self.use_lora else self.backbone
        if self.freeze and not self.use_lora:
            with torch.no_grad():
                feats = vit.forward_features(images)
        else:
            feats = vit.forward_features(images)
        # This RETFound_MAE version's forward_features uses mean(dim=1, keepdim=True) when
        # global_pool=True, returning [B, 1, embed_dim] instead of [B, embed_dim].
        if feats.dim() == 3:
            feats = feats.squeeze(1)
        return self.proj(feats)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze and not self.use_lora:
            self.backbone.eval()  # keep fully-frozen backbone (dropout etc) in eval mode always
        return self
