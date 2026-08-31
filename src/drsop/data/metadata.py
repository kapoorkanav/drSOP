import json
import math
from pathlib import Path

import torch

from drsop.data.text import parse_locale_number, tokenize_comorbidities


class MetadataProcessor:
    """Turns a raw BRSET row into fixed-size numeric/categorical/comorbidity tensors,
    using stats/vocab fit on the training split by scripts/prepare_data.py."""

    def __init__(self, processed_dir: str, numeric_fields: list, categorical_fields: list,
                 comorbidity_field: str):
        processed_dir = Path(processed_dir)
        with open(processed_dir / "metadata_stats.json") as f:
            self.stats = json.load(f)
        with open(processed_dir / "comorbidity_vocab.json") as f:
            self.comorbidity_vocab = json.load(f)

        self.numeric_fields = numeric_fields
        self.categorical_fields = categorical_fields
        self.comorbidity_field = comorbidity_field
        self.cat_to_idx = {
            field: {cat: i for i, cat in enumerate(cats)}
            for field, cats in self.stats["categorical"].items()
        }

    def num_categories(self, field: str) -> int:
        # +1 reserved index for unseen/missing categories at inference time
        return len(self.stats["categorical"][field]) + 1

    def num_tokens(self) -> int:
        # one token per numeric field, one per categorical field, one for comorbidity multi-hot
        return len(self.numeric_fields) + len(self.categorical_fields) + 1

    def transform(self, row) -> dict:
        numeric, numeric_missing = [], []
        for field in self.numeric_fields:
            mean, std = self.stats["numeric"][field]["mean"], self.stats["numeric"][field]["std"]
            val = parse_locale_number(row.get(field))
            is_missing = math.isnan(val)
            # Placeholder 0.0 when missing -- never actually used for prediction, since
            # MetadataEncoder overrides this field's whole token with a learned "missing"
            # embedding. Kept as a real (non-NaN) number so it can pass through the numeric
            # projection layer without polluting gradients with NaN before being overridden.
            numeric.append(0.0 if is_missing else (val - mean) / std)
            numeric_missing.append(1.0 if is_missing else 0.0)

        categorical = []
        for field in self.categorical_fields:
            vocab = self.cat_to_idx[field]
            raw = str(row.get(field))
            categorical.append(vocab.get(raw, len(vocab)))  # unseen/missing-category index

        comorbid_text = row.get(self.comorbidity_field)
        comorbidity_missing = not isinstance(comorbid_text, str)  # true NaN, not just "no tokens found"
        tokens = set(tokenize_comorbidities(comorbid_text))
        multi_hot = [1.0 if tok in tokens else 0.0 for tok in self.comorbidity_vocab]

        return {
            "numeric": torch.tensor(numeric, dtype=torch.float32),
            "numeric_missing": torch.tensor(numeric_missing, dtype=torch.float32),
            "categorical": torch.tensor(categorical, dtype=torch.long),
            "comorbidity": torch.tensor(multi_hot, dtype=torch.float32),
            "comorbidity_missing": torch.tensor(float(comorbidity_missing), dtype=torch.float32),
        }
