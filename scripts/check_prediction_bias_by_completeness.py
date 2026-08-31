"""
Checks whether the model's errors concentrate among missing-metadata patients specifically --
i.e. whether it learned a "metadata missing -> predict differently" shortcut rather than
genuinely reading the image, by cross-tabulating predictions against metadata-completeness tier.

    python scripts/check_prediction_bias_by_completeness.py --config configs/lora.yaml \
        --checkpoint runs/exp1_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    label_col = dcfg["label_col"]
    num_classes = mcfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = dcfg["processed_dir"]

    split_csv = Path(processed_dir) / f"{args.split}.csv"
    raw_df = pd.read_csv(split_csv)
    if "completeness_tier" not in raw_df.columns:
        raise RuntimeError(
            f"{split_csv} has no completeness_tier column -- rerun scripts/prepare_data.py "
            "(the current version) to regenerate it."
        )

    metadata = MetadataProcessor(
        processed_dir=processed_dir,
        numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    n_comorbidities = len(metadata.comorbidity_vocab)
    categorical_cardinalities = [metadata.num_categories(f) for f in dcfg["categorical_fields"]]

    ds = BRSETDataset(
        split_csv=str(split_csv), images_dir=dcfg["images_dir"], metadata=metadata,
        label_col=label_col, image_size=dcfg["image_size"], train=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = DRFusionModel(
        retfound_cfg=mcfg, meta_cfg=mcfg["meta_encoder"], gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"], categorical_cardinalities=categorical_cardinalities,
        n_numeric=len(dcfg["numeric_fields"]), n_comorbidities=n_comorbidities,
        proj_dim=mcfg["proj_dim"], num_classes=num_classes,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    all_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            logits = model(batch)
            all_pred.extend(logits.argmax(dim=-1).cpu().tolist())

    # DataLoader with shuffle=False preserves row order, so this lines up 1:1 with raw_df.
    raw_df["pred"] = all_pred
    raw_df["correct"] = raw_df["pred"] == raw_df[label_col]

    print(f"{args.split} set: {len(raw_df)} images")
    print(f"Completeness tiers present: {raw_df['completeness_tier'].value_counts().to_dict()}\n")

    print("Recall (%) by true grade, split by metadata completeness tier:")
    print(f"{'grade':>6} {'n_full':>8} {'recall_full':>12} {'n_partial':>10} {'recall_partial':>15}")
    for grade in range(num_classes):
        sub = raw_df[raw_df[label_col] == grade]
        full_sub = sub[sub["completeness_tier"] == "full"]
        partial_sub = sub[sub["completeness_tier"] == "partial"]
        full_recall = full_sub["correct"].mean() * 100 if len(full_sub) else float("nan")
        partial_recall = partial_sub["correct"].mean() * 100 if len(partial_sub) else float("nan")
        print(f"{grade:>6} {len(full_sub):>8} {full_recall:>11.1f}% {len(partial_sub):>10} "
              f"{partial_recall:>14.1f}%")

    print("\nPredicted-grade distribution for TRUE grade 1 patients, by completeness tier:")
    for tier in ["full", "partial"]:
        sub = raw_df[(raw_df[label_col] == 1) & (raw_df["completeness_tier"] == tier)]
        if len(sub):
            print(f"  {tier} (n={len(sub)}): {sub['pred'].value_counts().sort_index().to_dict()}")
        else:
            print(f"  {tier}: no examples")

    print("\nPredicted-grade distribution for TRUE grade 2 patients, by completeness tier:")
    for tier in ["full", "partial"]:
        sub = raw_df[(raw_df[label_col] == 2) & (raw_df["completeness_tier"] == tier)]
        if len(sub):
            print(f"  {tier} (n={len(sub)}): {sub['pred'].value_counts().sort_index().to_dict()}")
        else:
            print(f"  {tier}: no examples")


if __name__ == "__main__":
    main()
