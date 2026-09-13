import torch
import torch.nn as nn


class GateTransformer(nn.Module):
    """Takes [image_emb, meta_emb] as a 2-token sequence and outputs a per-dimension
    alpha in (0, 1): fused = alpha * image_emb + (1 - alpha) * meta_emb.

    With n_missing_flags > 0, a third token built from the per-field missingness pattern
    joins the sequence. The gate can already infer missingness from meta_emb in principle
    (it was built with learned missing-token embeddings), but only picks up a weak signal in
    practice -- it's buried in a pooled 512-dim vector, entangled with the field values
    themselves. This hands it the pattern directly as a few clean bits, and self-attention
    lets the modality tokens condition on it (e.g. "insulin unknown AND this kind of image").
    The missingness token stays contextual: only the two modality tokens are pooled for
    alpha, so alpha_head's shape is unchanged."""

    def __init__(self, dim: int, n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1,
                 n_missing_flags: int = 0):
        super().__init__()
        self.n_missing_flags = n_missing_flags
        n_tokens = 3 if n_missing_flags else 2

        self.modality_emb = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.modality_emb, std=0.02)

        if n_missing_flags:
            # Small init (matching modality_emb's scale) on purpose: image_emb and meta_emb
            # are trained projection outputs with much larger magnitude, so PyTorch's default
            # Linear init here would let this token dominate attention from step one. Starting
            # small makes it a nudge that grows only if training finds it useful. Zero bias
            # means "nothing missing" starts as a neutral token rather than an arbitrary one.
            self.missing_proj = nn.Linear(n_missing_flags, dim)
            nn.init.trunc_normal_(self.missing_proj.weight, std=0.02)
            nn.init.zeros_(self.missing_proj.bias)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.alpha_head = nn.Linear(dim * 2, dim)

    def forward(self, image_emb: torch.Tensor, meta_emb: torch.Tensor,
                missing_flags: torch.Tensor = None):
        tokens = [image_emb, meta_emb]
        if self.n_missing_flags:
            if missing_flags is None:
                raise ValueError(
                    "GateTransformer was built with n_missing_flags="
                    f"{self.n_missing_flags} but forward() got missing_flags=None."
                )
            tokens.append(self.missing_proj(missing_flags))

        x = torch.stack(tokens, dim=1) + self.modality_emb
        x = self.encoder(x)
        pooled = torch.cat([x[:, 0], x[:, 1]], dim=-1)
        alpha = torch.sigmoid(self.alpha_head(pooled))
        return alpha
