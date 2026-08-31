import torch
import torch.nn as nn


class MetadataEncoder(nn.Module):
    """Encodes numeric + categorical + multi-hot comorbidity fields as a token sequence,
    runs a small transformer encoder, and projects the pooled [CLS] output to `proj_dim`.

    Missing values get an explicit, learned "missing" embedding per field (numeric and
    comorbidity) rather than an imputed/faked value -- the model can tell "unknown" apart
    from "average". Categorical fields get this for free via their reserved unseen/missing
    embedding-table index (see MetadataProcessor.num_categories)."""

    def __init__(self, n_numeric: int, categorical_cardinalities: list, n_comorbidities: int,
                 proj_dim: int, d_model: int = 128, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_numeric = n_numeric
        self.n_categorical = len(categorical_cardinalities)

        # one linear projection per numeric field (each scalar -> d_model token)
        self.numeric_proj = nn.ModuleList([nn.Linear(1, d_model) for _ in range(n_numeric)])
        self.numeric_missing_emb = nn.Parameter(torch.zeros(max(n_numeric, 1), d_model))
        nn.init.trunc_normal_(self.numeric_missing_emb, std=0.02)

        self.categorical_emb = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in categorical_cardinalities]
        )
        self.comorbidity_proj = nn.Linear(n_comorbidities, d_model)
        self.comorbidity_missing_emb = nn.Parameter(torch.zeros(1, d_model))
        nn.init.trunc_normal_(self.comorbidity_missing_emb, std=0.02)

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

    def forward(self, numeric: torch.Tensor, numeric_missing: torch.Tensor,
                categorical: torch.Tensor, comorbidity: torch.Tensor,
                comorbidity_missing: torch.Tensor) -> torch.Tensor:
        batch = numeric.shape[0]
        tokens = [self.cls_token.expand(batch, -1, -1)]
        for i in range(self.n_numeric):
            val_tok = self.numeric_proj[i](numeric[:, i:i + 1])
            miss_mask = numeric_missing[:, i:i + 1].bool()
            miss_tok = self.numeric_missing_emb[i].unsqueeze(0).expand(batch, -1)
            tok = torch.where(miss_mask, miss_tok, val_tok)
            tokens.append(tok.unsqueeze(1))
        for i in range(self.n_categorical):
            tokens.append(self.categorical_emb[i](categorical[:, i]).unsqueeze(1))

        comorbid_tok = self.comorbidity_proj(comorbidity)
        comorbid_miss_mask = comorbidity_missing.bool().unsqueeze(-1)
        comorbid_miss_tok = self.comorbidity_missing_emb.expand(batch, -1)
        comorbid_tok = torch.where(comorbid_miss_mask, comorbid_miss_tok, comorbid_tok)
        tokens.append(comorbid_tok.unsqueeze(1))

        x = torch.cat(tokens, dim=1) + self.pos_emb
        x = self.encoder(x)
        cls_out = x[:, 0]
        return self.out_proj(cls_out)
