import torch
import torch.nn as nn


class GateTransformer(nn.Module):
    """Takes [image_emb, meta_emb] as a 2-token sequence and outputs a per-dimension
    alpha in (0, 1): fused = alpha * image_emb + (1 - alpha) * meta_emb."""

    def __init__(self, dim: int, n_layers: int = 1, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.modality_emb = nn.Parameter(torch.zeros(1, 2, dim))
        nn.init.trunc_normal_(self.modality_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.alpha_head = nn.Linear(dim * 2, dim)

    def forward(self, image_emb: torch.Tensor, meta_emb: torch.Tensor):
        x = torch.stack([image_emb, meta_emb], dim=1) + self.modality_emb
        x = self.encoder(x)
        pooled = torch.cat([x[:, 0], x[:, 1]], dim=-1)
        alpha = torch.sigmoid(self.alpha_head(pooled))
        return alpha
