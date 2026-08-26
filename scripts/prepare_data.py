"""
Complete-metadata patient-level split + preprocessing for the two-stage BRSET pipeline.

Run once before training:
    python scripts/prepare_data.py --config configs/default.yaml

Stage 1 (binary): DR present/absent (`data.binary_label_col`), trained on the larger cohort.
Stage 2 (severity): ICDR grade 0-4 (`data.label_col`), trained on the subset that also has a grade.

Both stages are carved out of ONE global patient-level train/val/test split (computed once, over
all complete-metadata rows regardless of label), so a given patient always falls in the same split
for both stages -- this also guarantees both eyes of a patient always land in the same split, since
the split groups by patient_id.

Produces, under data.processed_dir:
    {train,val,test}_stage1.csv     (rows with a non-null binary_label_col)
    {train,val,test}_stage2.csv     (rows with a non-null label_col)
    comorbidity_vocab.json          (top-K free-text tokens found in the stage1 train split)
    metadata_stats.json             (numeric field mean/std + categorical vocabs, fit on stage1 train)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.text import tokenize_comorbidities  # noqa: E402


def split_patients(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(gss1.split(df, groups=df["patient_id"]))
    train_df, rest_df = df.iloc[train_idx], df.iloc[rest_idx]

    rel_test_frac = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test_frac, random_state=seed)
    val_idx, test_idx = next(gss2.split(rest_df, groups=rest_df["patient_id"]))
    val_df, test_df = rest_df.iloc[val_idx], rest_df.iloc[test_idx]

    assert set(train_df.patient_id) & set(val_df.patient_id) == set()
    assert set(train_df.patient_id) & set(test_df.patient_id) == set()
    assert set(val_df.patient_id) & set(test_df.patient_id) == set()
    return train_df, val_df, test_df


def report_missingness(df: pd.DataFrame, fields: list) -> None:
    print("Missingness by field (on the raw data, before any filtering):")
    for field in fields:
        n_missing = df[field].isna().sum()
        pct = 100 * n_missing / len(df) if len(df) else 0.0
        print(f"  {field}: {n_missing}/{len(df)} missing ({pct:.1f}%)")


def build_comorbidity_vocab(train_df: pd.DataFrame, field: str, vocab_size: int) -> list:
    counter = Counter()
    for text in train_df[field].fillna(""):
        counter.update(tokenize_comorbidities(text))
    return [tok for tok, _ in counter.most_common(vocab_size)]


def fit_metadata_stats(train_df: pd.DataFrame, numeric_fields: list, categorical_fields: list) -> dict:
    stats = {"numeric": {}, "categorical": {}}
    for field in numeric_fields:
        vals = pd.to_numeric(train_df[field], errors="coerce")
        stats["numeric"][field] = {"mean": float(vals.mean()), "std": float(vals.std() or 1.0)}
    for field in categorical_fields:
        cats = sorted(train_df[field].dropna().astype(str).unique().tolist())
        stats["categorical"][field] = cats
    return stats


def stage_view(split_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    view = split_df.dropna(subset=[label_col]).copy()
    view[label_col] = view[label_col].astype(int)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]

    df = pd.read_csv(dcfg["raw_labels_csv"])
    print(f"Raw: {len(df)} images, {df.patient_id.nunique()} patients.")

    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    report_missingness(df, metadata_fields + [dcfg["binary_label_col"], dcfg["label_col"]])

    # Complete-case filter: every downstream row (both stages) has full metadata.
    before = len(df)
    df = df.dropna(subset=metadata_fields)
    print(f"Complete-metadata filter: dropped {before - len(df)} of {before} rows "
          f"({len(df)} remain, {df.patient_id.nunique()} patients).")

    # One global patient-level split over the complete-metadata cohort. Both stages are
    # label-filtered VIEWS of this same split, so a patient's split assignment never
    # differs between stage 1 and stage 2, and both eyes always share a split.
    train_df, val_df, test_df = split_patients(
        df, dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], dcfg["split"]["seed"]
    )

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_cols = {"stage1": dcfg["binary_label_col"], "stage2": dcfg["label_col"]}
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for stage_name, label_col in stage_cols.items():
        print(f"\n{stage_name} (label={label_col}):")
        for split_name, split_df in splits.items():
            view = stage_view(split_df, label_col)
            view.to_csv(out_dir / f"{split_name}_{stage_name}.csv", index=False)
            print(f"  {split_name}: {len(view)} images, {view.patient_id.nunique()} patients")

    # Fit metadata artifacts on stage1's train view: it's the broadest train pool, and
    # stage2's train patients are a strict subset of it (same global train split).
    stage1_train = stage_view(train_df, dcfg["binary_label_col"])
    vocab = build_comorbidity_vocab(stage1_train, dcfg["comorbidity_field"], dcfg["comorbidity_vocab_size"])
    print(f"\nComorbidity vocab (top {len(vocab)} tokens from stage1 train): {vocab}")
    stats = fit_metadata_stats(stage1_train, dcfg["numeric_fields"], dcfg["categorical_fields"])

    with open(out_dir / "comorbidity_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    with open(out_dir / "metadata_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote processed splits + metadata artifacts to {out_dir}")


if __name__ == "__main__":
    main()
