"""
Scores a checkpoint separately for each level of metadata completeness.

Metadata can only help a patient who has metadata. Every result so far pools the whole test set,
which averages the patients whose charts are complete together with the patients who have almost
nothing recorded -- so a real benefit confined to the complete-metadata group would be diluted,
and a shortcut keyed on missingness would be hidden.

Run this on the fusion model and on the image-only control and compare them WITHIN the
full-metadata group. That is the sharpest available test of whether the metadata path earns its
place: if fusion does not win among the patients who actually have charts, it does not win
anywhere.

Groups the test set two ways -- by completeness tier (full / partial / none) and by the exact
number of missing fields -- and reports accuracy against each group's own majority baseline,
balanced accuracy, per-class recall, and referral sensitivity.

    python scripts/evaluate_by_completeness.py --config configs/balanced_3class_lora.yaml \
        --checkpoint runs/exp1_balanced_3class_lora/best.pt

    python scripts/evaluate_by_completeness.py \
        --config configs/balanced_3class_imageonly_proj512_lora.yaml \
        --checkpoint runs/exp1_balanced_3class_imageonly_proj512_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.labels import apply_label_map  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.factory import build_model  # noqa: E402


CLASS_NAMES_3 = ["none", "mid", "severe"]


def class_names(n_classes: int) -> list:
    return CLASS_NAMES_3 if n_classes == 3 else [f"g{i}" for i in range(n_classes)]


def group_report(emit, name: str, sub: pd.DataFrame, label_col: str, n_classes: int,
                 matrix: bool = False) -> dict:
    """Metrics for one completeness group, each against its own baseline.

    matrix=True also prints the group's confusion matrix with raw counts, so nothing has to be
    reconstructed from percentages -- and so the off-diagonal is visible. Where a missed case
    went is clinically different: true-mid predicted none is a missed referral, true-mid
    predicted severe is an over-referral.
    """
    if len(sub) == 0:
        emit(f"  {name:<22} (empty)")
        return {}
    true = sub[label_col].values
    pred = sub["pred"].values
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1

    support = cm.sum(1)
    recalls = np.divide(cm.diagonal(), support,
                        out=np.full(n_classes, np.nan, dtype=float), where=support > 0)
    acc = cm.diagonal().sum() / cm.sum()
    # Each group has its own class mix, so each needs its own majority baseline. Comparing a
    # group's accuracy against the whole test set's baseline would be meaningless.
    baseline = support.max() / support.sum()
    diseased, missed = cm[1:, :].sum(), cm[1:, 0].sum()
    sens = (1 - missed / diseased) if diseased else float("nan")

    recall_str = " ".join(
        f"{r * 100:5.1f}%" if not np.isnan(r) else "   --- " for r in recalls)
    emit(f"  {name:<22} n={cm.sum():>4}  acc {acc * 100:5.1f}% (base {baseline * 100:4.1f}%)  "
         f"bal {np.nanmean(recalls) * 100:5.1f}%  recall[{recall_str}]  "
         f"sens {sens * 100:5.1f}%  missed {missed}/{diseased}")

    if matrix:
        names = class_names(n_classes)
        emit("")
        emit(f"    confusion, {name} (rows=true, cols=predicted):")
        # Backslash kept out of the f-string expression: that is a syntax error before 3.12.
        corner = "true\\pred"
        emit("      " + corner.ljust(10) + "".join(f"{n:>9}" for n in names)
             + f"{'n':>7}{'recall':>9}")
        for i, n in enumerate(names):
            row = "      " + f"{n:<10}" + "".join(f"{cm[i, j]:>9}" for j in range(n_classes))
            row += f"{support[i]:>7}"
            row += f"{recalls[i] * 100:>8.1f}%" if not np.isnan(recalls[i]) else f"{'--':>9}"
            emit(row)
        emit("      " + f"{'predicted n':<10}" + "".join(f"{cm[:, j].sum():>9}"
                                                          for j in range(n_classes)))
        if n_classes == 3:
            emit(f"      mid called none (missed referral):   {cm[1, 0]}/{support[1]}")
            emit(f"      mid called severe (over-referral):   {cm[1, 2]}/{support[1]}")
            emit(f"      severe called none (worst error):    {cm[2, 0]}/{support[2]}")
        emit("")

    return {"n": int(cm.sum()), "acc": float(acc), "baseline": float(baseline),
            "balanced": float(np.nanmean(recalls)), "sens": float(sens),
            "support": support.tolist(), "cm": cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = resolve(load_config(args.config), ROOT)
    dcfg, mcfg = cfg["data"], cfg["model"]
    label_col, n_classes = dcfg["label_col"], mcfg["num_classes"]
    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_csv = Path(dcfg["processed_dir"]) / f"{args.split}.csv"
    df = pd.read_csv(split_csv)
    df[label_col] = apply_label_map(df[label_col], dcfg.get("label_map"))

    metadata = MetadataProcessor(
        processed_dir=dcfg["processed_dir"], numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    ds = BRSETDataset(
        split_csv=str(split_csv), images_dir=dcfg["images_dir"], metadata=metadata,
        label_col=label_col, image_size=dcfg["image_size"], train=False,
        label_map=dcfg.get("label_map"),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = build_model(
        mcfg,
        categorical_cardinalities=[metadata.num_categories(f) for f in dcfg["categorical_fields"]],
        n_numeric=len(dcfg["numeric_fields"]),
        n_comorbidities=len(metadata.comorbidity_vocab),
        num_classes=n_classes,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    preds, alphas = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluating ({args.split})", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            logits, alpha = model(batch, return_alpha=True)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            alphas.extend(alpha.mean(dim=-1).cpu().tolist())

    # DataLoader with shuffle=False preserves row order, so this lines up 1:1 with df.
    df["pred"] = preds
    df["alpha"] = alphas
    df["n_missing"] = df[metadata_fields].isna().sum(axis=1)

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    is_image_only = bool(mcfg.get("image_only", False))
    emit(f"config:     {args.config}")
    emit(f"checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    emit(f"model:      {'IMAGE-ONLY (no metadata)' if is_image_only else 'gated fusion'}, "
         f"{n_classes} classes")
    emit(f"split:      {args.split}, {len(df)} images, "
         f"{df['patient_id'].nunique()} patients")
    emit(f"metadata fields counted for completeness: {metadata_fields}")
    if is_image_only:
        emit("")
        emit("NOTE: this model never reads metadata. Its rows below are the control -- any")
        emit("      variation across completeness groups is image difficulty, not metadata.")

    emit("")
    emit("=" * 118)
    emit(f"BY COMPLETENESS TIER        recall[{' '.join(f'    c{i}' for i in range(n_classes))}]")
    emit("=" * 118)
    results = {}
    for tier in ("full", "partial", "none"):
        if "completeness_tier" not in df.columns:
            emit("  no completeness_tier column -- rerun the prep script")
            break
        results[tier] = group_report(
            emit, tier, df[df["completeness_tier"] == tier], label_col, n_classes,
            matrix=True)
    results["ALL"] = group_report(emit, "ALL (pooled)", df, label_col, n_classes, matrix=True)

    emit("")
    emit("=" * 118)
    emit("BY NUMBER OF MISSING FIELDS")
    emit("=" * 118)
    for n_miss in sorted(df["n_missing"].unique()):
        group_report(emit, f"{n_miss} missing", df[df["n_missing"] == n_miss],
                     label_col, n_classes)

    if not is_image_only:
        emit("")
        emit("=" * 118)
        emit("GATE BEHAVIOUR BY TIER   (alpha: 1.0 = read the image, 0.0 = read the metadata)")
        emit("=" * 118)
        for tier in ("full", "partial", "none"):
            sub = df[df["completeness_tier"] == tier]
            if len(sub):
                emit(f"  {tier:<10} n={len(sub):>4}  mean alpha {sub['alpha'].mean():.4f}  "
                     f"(sd {sub['alpha'].std():.4f})")
        emit("")
        emit("  If these are equal, the gate is not reacting to whether metadata is present.")

    emit("")
    emit("=" * 118)
    emit("HOW TO READ THIS")
    emit("=" * 118)
    emit("  The 'full' row is the one that matters. Metadata can only help a patient who has")
    emit("  metadata, so run this for the fusion model and for the image-only control and")
    emit("  compare their 'full' rows directly. If fusion does not beat image-only among the")
    emit("  patients who have complete charts, the metadata path does not help anywhere.")
    emit("")
    emit("  Each group is scored against ITS OWN majority baseline, shown in brackets. Groups")
    emit("  differ in class mix, so a higher accuracy in one group can mean an easier group")
    emit("  rather than a better model -- the gap to its own baseline is the comparable figure.")
    if results.get("full", {}).get("n", 0) and results.get("partial", {}).get("n", 0):
        f, p = results["full"], results["partial"]
        emit("")
        emit(f"  full:    acc {f['acc'] * 100:.1f}% vs baseline {f['baseline'] * 100:.1f}% "
             f"-> {(f['acc'] - f['baseline']) * 100:+.1f} points, balanced {f['balanced'] * 100:.1f}%")
        emit(f"  partial: acc {p['acc'] * 100:.1f}% vs baseline {p['baseline'] * 100:.1f}% "
             f"-> {(p['acc'] - p['baseline']) * 100:+.1f} points, balanced {p['balanced'] * 100:.1f}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
