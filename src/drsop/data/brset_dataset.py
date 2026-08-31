from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from drsop.data.metadata import MetadataProcessor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BRSETDataset(Dataset):
    def __init__(self, split_csv: str, images_dir: str, metadata: MetadataProcessor,
                 label_col: str, image_size: int, train: bool):
        self.df = pd.read_csv(split_csv)
        self.images_dir = Path(images_dir)
        self.metadata = metadata
        self.label_col = label_col
        self.transform = build_transform(image_size, train)

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_id) -> Path:
        # BRSET's exact image extension (.jpg vs .png) can vary by release; check both.
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = self.images_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No image found for image_id={image_id} in {self.images_dir}")

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(row["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        meta = self.metadata.transform(row)
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)

        return {
            "image": image,
            "numeric": meta["numeric"],
            "numeric_missing": meta["numeric_missing"],
            "categorical": meta["categorical"],
            "comorbidity": meta["comorbidity"],
            "comorbidity_missing": meta["comorbidity_missing"],
            "label": label,
        }
