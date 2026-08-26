"""
Evaluate a trained checkpoint on the held-out test set.

    python evaluate.py --config configs/default.yaml --checkpoint runs/exp1/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    label_col = dcfg["label_col"]
    num_classes = mcfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = dcfg["processed_dir"]

    metadata = MetadataProcessor(
        processed_dir=processed_dir,
        numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    n_comorbidities = len(metadata.comorbidity_vocab)
    categorical_cardinalities = [metadata.num_categories(f) for f in dcfg["categorical_fields"]]

    ds = BRSETDataset(
        split_csv=str(Path(processed_dir) / f"{args.split}.csv"),
        images_dir=dcfg["images_dir"],
        metadata=metadata,
        label_col=label_col,
        image_size=dcfg["image_size"],
        train=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = DRFusionModel(
        retfound_cfg=mcfg,
        meta_cfg=mcfg["meta_encoder"],
        gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"],
        categorical_cardinalities=categorical_cardinalities,
        n_numeric=len(dcfg["numeric_fields"]),
        n_comorbidities=n_comorbidities,
        proj_dim=mcfg["proj_dim"],
        num_classes=num_classes,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)  # backbone excluded from checkpoint, see train.py
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (best val QWK at save time: {ckpt['best_qwk']:.4f})")

    all_true, all_pred, all_alpha = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("label")
            logits, alpha = model(batch, return_alpha=True)
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())
            all_alpha.append(alpha.mean(dim=-1).cpu())  # per-example mean alpha across dims

    metrics = compute_metrics(all_true, all_pred)
    print(f"\n{args.split} set ({len(all_true)} images):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    cm = confusion_matrix(all_true, all_pred, labels=list(range(num_classes)))
    print(f"\nConfusion matrix (rows=true grade, cols=predicted grade):")
    print(cm)

    mean_alpha = torch.cat(all_alpha).mean().item()
    print(f"\nMean alpha (image-vs-metadata gate weight, 1.0=all image / 0.0=all metadata): {mean_alpha:.4f}")


if __name__ == "__main__":
    main()
