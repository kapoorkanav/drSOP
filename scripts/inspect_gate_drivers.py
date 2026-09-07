"""
The gate varies its routing substantially per patient (across-patient spread 0.21 per dim)
but demonstrably NOT with metadata missingness (p=0.37, check_prediction_bias_by_completeness).
So what IS it reacting to?

Correlates the gate's behaviour against every candidate driver we have -- true grade, predicted
grade, age, diabetes duration, comorbidity count, image quality, missingness -- and separately
asks whether the routing PATTERN itself (not just its average) shifts with severity.

    python scripts/inspect_gate_drivers.py --config configs/balanced_lora.yaml \
        --checkpoint runs/exp1_balanced_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.data.text import tokenize_comorbidities  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402


def correlate(df: pd.DataFrame, col: str, target: str = "alpha_mean") -> None:
    sub = df[[col, target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 10 or sub[col].nunique() < 2:
        print(f"  {col:>20}: too few usable values (n={len(sub)})")
        return
    r, p = spearmanr(sub[col], sub[target])
    flag = "  <-- significant" if p < 0.05 else ""
    print(f"  {col:>20}: rho={r:+.3f}  p={p:.4g}  (n={len(sub)}){flag}")


def group_means(df: pd.DataFrame, col: str, target: str = "alpha_mean") -> None:
    for value, sub in df.groupby(col):
        print(f"    {str(value):>12} (n={len(sub):>4}): alpha = {sub[target].mean():.4f} "
              f"(sd {sub[target].std():.4f})")


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
    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = dcfg["processed_dir"]

    split_csv = Path(processed_dir) / f"{args.split}.csv"
    df = pd.read_csv(split_csv)

    metadata = MetadataProcessor(
        processed_dir=processed_dir, numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"], comorbidity_field=dcfg["comorbidity_field"],
    )
    ds = BRSETDataset(
        split_csv=str(split_csv), images_dir=dcfg["images_dir"], metadata=metadata,
        label_col=label_col, image_size=dcfg["image_size"], train=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = DRFusionModel(
        retfound_cfg=mcfg, meta_cfg=mcfg["meta_encoder"], gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"],
        categorical_cardinalities=[metadata.num_categories(f) for f in dcfg["categorical_fields"]],
        n_numeric=len(dcfg["numeric_fields"]), n_comorbidities=len(metadata.comorbidity_vocab),
        proj_dim=mcfg["proj_dim"], num_classes=mcfg["num_classes"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    alphas, preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"collecting alpha ({args.split})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            logits, alpha = model(batch, return_alpha=True)
            alphas.append(alpha.cpu().numpy())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    alpha = np.concatenate(alphas, axis=0)  # [n, proj_dim]

    df["alpha_mean"] = alpha.mean(axis=1)
    df["pred"] = preds
    df["n_missing"] = df[metadata_fields].isna().sum(axis=1)
    df["n_comorbidities"] = df[dcfg["comorbidity_field"]].apply(
        lambda t: len(tokenize_comorbidities(t)))

    print(f"\n{args.split} set: {len(df)} images | overall mean alpha "
          f"{df['alpha_mean'].mean():.4f} (sd {df['alpha_mean'].std():.4f})")
    print("(alpha: 1.0 = read the image, 0.0 = read the metadata)")

    print("\n" + "=" * 78)
    print("MEAN ALPHA BY GROUP")
    print("=" * 78)
    print(f"\n  By TRUE grade ({label_col}):")
    group_means(df, label_col)
    print("\n  By PREDICTED grade:")
    group_means(df, "pred")
    print("\n  By metadata completeness (known null result -- shown for reference):")
    group_means(df, "completeness_tier")
    if "quality" in df.columns:
        print("\n  By image quality:")
        group_means(df, "quality")

    print("\n" + "=" * 78)
    print("RANK CORRELATION vs MEAN ALPHA  (Spearman)")
    print("=" * 78)
    for col in [label_col, "pred", "n_missing", "n_comorbidities"] + dcfg["numeric_fields"]:
        if col in df.columns:
            correlate(df, col)

    print("\n" + "=" * 78)
    print("DOES THE ROUTING PATTERN ITSELF SHIFT WITH SEVERITY?")
    print("=" * 78)
    print("  Correlation between the mean 512-dim alpha VECTOR of each grade pair.")
    print("  ~1.00 everywhere => same routing pattern regardless of severity;")
    print("  lower values     => the gate re-routes for different disease severities.\n")
    grades = sorted(df[label_col].unique())
    vectors = {g: alpha[(df[label_col] == g).values].mean(axis=0) for g in grades
               if (df[label_col] == g).sum() >= 5}
    if len(vectors) >= 2:
        gs = sorted(vectors)
        header = "      " + "".join(f"{f'g{g}':>8}" for g in gs)
        print(header)
        for g1 in gs:
            row = f"   g{g1} "
            for g2 in gs:
                row += f"{np.corrcoef(vectors[g1], vectors[g2])[0, 1]:>8.3f}"
            print(row)
        print(f"\n  (grades with fewer than 5 examples were skipped)")
    else:
        print("  Not enough grades with >=5 examples to compare.")


if __name__ == "__main__":
    main()
