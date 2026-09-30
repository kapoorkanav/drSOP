"""
Re-scores an existing checkpoint at the unit a clinician actually decides on.

Everything so far has been scored PER IMAGE, which is not how screening works. A patient is
referred if any of their images shows disease; an eye is graded by the worst thing visible in
it. mBRSET images two fields per eye (one macula-centred, one optic-disc-centred), so a model
that spots disease in one of the two is currently being marked half wrong.

This script pools predictions upward and re-computes the metrics:

  image   -> as reported so far
  eye     -> max over that eye's images, for prediction AND ground truth
  patient -> max over that patient's images

Max-pooling is the right aggregation for an ordinal severity grade: the eye's grade is the worst
finding in it, which is also how the dataset's own per-image grades should combine.

It also splits performance by field position. mBRSET records two images per eye without naming
which is which, so they are labelled by their order within the eye -- "field A" and "field B",
consistently assigned. If one field scores far better than the other, that is the macula-centred
view carrying the signal, and it means roughly half the test set is images where the disease is
not visible at all.

No retraining and no GPU pressure beyond one inference pass.

    python scripts/evaluate_by_unit.py --config mbrset/configs/mbrset_3class_lora.yaml \
        --checkpoint runs/mbrset_3class_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.labels import apply_label_map  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.factory import build_model  # noqa: E402


def confusion(true, pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    return cm


def report_level(emit, name, true, pred, n_classes):
    m = compute_metrics(true, pred)
    cm = confusion(true, pred, n_classes)
    recalls = np.divide(cm.diagonal(), cm.sum(1),
                        out=np.full(n_classes, np.nan), where=cm.sum(1) > 0)
    # "predict the most common class for everything" -- the bar any accuracy must clear
    baseline = cm.sum(1).max() / cm.sum() if cm.sum() else float("nan")

    emit("")
    emit("=" * 74)
    emit(f"{name.upper()} LEVEL  ({cm.sum()} units)")
    emit("=" * 74)
    emit(f"  accuracy {m['accuracy']:.4f}   (majority-class baseline {baseline:.4f})")
    emit(f"  QWK      {m['qwk']:.4f}")
    emit(f"  macro-F1 {m['macro_f1']:.4f}")
    emit(f"  balanced accuracy (mean recall) {np.nanmean(recalls):.4f}   "
         f"(baseline {1 / n_classes:.4f})")
    emit("")
    emit("  confusion (rows=true, cols=pred):")
    header = "         " + "".join(f"{f'p{i}':>7}" for i in range(n_classes)) + "     recall"
    emit(header)
    for i in range(n_classes):
        row = f"    t{i}   " + "".join(f"{cm[i, j]:>7}" for j in range(n_classes))
        row += f"   {recalls[i] * 100:>6.1f}%" if not np.isnan(recalls[i]) else "       --"
        emit(row)

    # Referral view: class 0 is "no disease"; anything above it warrants a look.
    diseased = cm[1:, :].sum()
    missed = cm[1:, 0].sum()
    healthy = cm[0, :].sum()
    over = cm[0, 1:].sum()
    if diseased and healthy:
        emit("")
        emit(f"  referral: sensitivity {100 * (1 - missed / diseased):>5.1f}%  "
             f"({diseased - missed}/{diseased} diseased caught)")
        emit(f"            specificity {100 * (1 - over / healthy):>5.1f}%  "
             f"({healthy - over}/{healthy} healthy not referred)")
        if n_classes >= 3:
            worst_missed = cm[-1, 0]
            emit(f"            most-severe class called healthy: {worst_missed}/{cm[-1].sum()}")
    return {"accuracy": m["accuracy"], "baseline": baseline,
            "balanced": float(np.nanmean(recalls)),
            "sensitivity": float(1 - missed / diseased) if diseased else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eye-col", default="laterality",
                        help="column naming the eye; skipped if absent from the split CSV")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    label_col, n_classes = dcfg["label_col"], mcfg["num_classes"]
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

    # Via the factory so this works for the image-only control too, which has no gate and no
    # proj_dim.
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

    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"evaluating ({args.split})", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            preds.extend(model(batch).argmax(dim=-1).cpu().tolist())

    # DataLoader with shuffle=False preserves row order, so this lines up 1:1 with df.
    df["pred"] = preds

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    emit(f"Config: {args.config}   split: {args.split}   classes: {n_classes}")
    emit("Pooling upward with max: an eye's grade is the worst finding visible in it.")

    report_level(emit, "image", df[label_col].values, df["pred"].values, n_classes)

    has_eye = args.eye_col in df.columns
    if has_eye:
        eye = df.groupby(["patient_id", args.eye_col]).agg(
            true=(label_col, "max"), pred=("pred", "max")).reset_index()
        report_level(emit, "eye", eye["true"].values, eye["pred"].values, n_classes)
    else:
        emit("")
        emit(f"(no '{args.eye_col}' column in {split_csv.name} -- skipping eye level)")

    patient = df.groupby("patient_id").agg(
        true=(label_col, "max"), pred=("pred", "max")).reset_index()
    report_level(emit, "patient", patient["true"].values, patient["pred"].values, n_classes)

    # --- which of the two fields per eye carries the signal? ---
    if has_eye:
        emit("")
        emit("=" * 74)
        emit("BY FIELD POSITION WITHIN THE EYE")
        emit("=" * 74)
        emit("  mBRSET images two fields per eye (macula-centred and optic-disc-centred) but")
        emit("  does not record which is which. They are labelled here by their order within")
        emit("  the eye. DR lesions cluster around the macula, so if one field scores much")
        emit("  better, that is the macula view -- and the other half of the test set is")
        emit("  images where the disease may not be visible at all.")
        df = df.sort_values(["patient_id", args.eye_col, "image_id"])
        df["field"] = df.groupby(["patient_id", args.eye_col]).cumcount()
        for field, sub in df[df["field"] < 2].groupby("field"):
            report_level(emit, f"image, field {'AB'[int(field)]}",
                         sub[label_col].values, sub["pred"].values, n_classes)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
