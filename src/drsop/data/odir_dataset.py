from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from drsop.data.brset_dataset import build_transform


class ODIRDataset(Dataset):
    """One eye-image per row, with an independent 0/1 label per disease.

    Reads the CSVs written by scripts/prepare_data_odir.py. The label is a float vector (one
    entry per disease code, in `disease_codes` order) because the heads are independent binary
    classifiers trained with BCE, not a single softmax -- an eye can have several at once.

    Returns only "image" and "label": ImageOnlyModel reads nothing else from the batch.
    """

    def __init__(self, split_csv: str, images_dir: str, disease_codes: list,
                 image_size: int, train: bool):
        self.df = pd.read_csv(split_csv)
        self.images_dir = Path(images_dir)
        self.disease_codes = disease_codes
        self.transform = build_transform(image_size, train)
        missing = [c for c in disease_codes if c not in self.df.columns]
        if missing:
            raise ValueError(f"{split_csv} has no column(s) {missing} -- rerun prepare_data_odir.py")

    def __len__(self) -> int:
        return len(self.df)

    def labels(self) -> torch.Tensor:
        """All labels as a [n, n_diseases] tensor -- used to size the per-disease loss weights."""
        return torch.tensor(self.df[self.disease_codes].values, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.images_dir / row["image_id"]
        if not path.exists():
            raise FileNotFoundError(f"No image at {path} (image_id={row['image_id']})")
        image = self.transform(Image.open(path).convert("RGB"))
        label = torch.tensor([float(row[c]) for c in self.disease_codes], dtype=torch.float32)
        return {"image": image, "label": label}
