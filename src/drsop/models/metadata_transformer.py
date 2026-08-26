import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """Encodes numeric + categorical + multi-hot comorbidity fields as a token sequence,
    runs a small transformer encoder, and projects the pooled [CLS] output to `proj_dim`."""

    def __init__(self, n_numeric: int, categorical_cardinalities: list, n_comorbidities: int,
                 proj_dim: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_numeric = n_numeric
        self.n_categorical = len(categorical_cardinalities)

        # one linear projection per numeric field (each scalar -> d_model token)
        self.numeric_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_numeric)])
        self.categorical_emb = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in categorical_cardinalities]
        )
        self.comorbidity_proj = nn.Linear(n_comorbidities, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        n_tokens = 1 + n_numeric + self.n_categorical + 1  # cls + numeric + categorical + comorbidity
        self.pos_emb = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, proj_dim)

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor,
                comorbidity: torch.Tensor) -> torch.Tensor:
        batch = numeric.shape[0]
        tokens = [self.cls_token.expand(batch, -1, -1)]
        for i in range(self.n_numeric):
            tokens.append(self.numeric_proj[i](numeric[:, i:i + 1]).unsqueeze(1))
        for i in range(self.n_categorical):
            tokens.append(self.categorical_emb[i](categorical[:, i]).unsqueeze(1))
        tokens.append(self.comorbidity_proj(comorbidity).unsqueeze(1))

        x = torch.cat(tokens, dim=1) + self.pos_emb
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.out_proj(cls_out)
