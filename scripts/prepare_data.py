"""
Patient-level, stratified-by-(grade x metadata-completeness) split + preprocessing.

Uses ALL labeled rows now, not just complete-metadata ones -- missing metadata fields
get handled by the model (missing-token embeddings) rather than filtered out here.

Run once before training:
    python scripts/prepare_data.py --config configs/default.yaml

Produces, under data.processed_dir:
    train.csv, val.csv, test.csv    (all rows with a non-null label_col)
    comorbidity_vocab.json          (top-K free-text tokens found in the training split)
    metadata_stats.json             (numeric field mean/std + categorical vocabs, fit on
                                      whatever's non-missing in the training split)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.text import parse_locale_number, tokenize_comorbidities  # noqa: E402


def completeness_tier(row, metadata_fields: list) -> str:
    n_present = sum(pd.notna(row[f]) for f in metadata_fields)
    if n_present == len(metadata_fields):
        return "full"
    if n_present == 0:
        return "none"
    return "partial"


def build_patient_strat_key(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One row per patient: worst (max) grade across their images, and their metadata
    completeness tier (same fields are patient-level, so consistent across a patient's
    images in practice -- take the first row's tier). Combined into one stratification
    key so both severity AND completeness are balanced across train/val/test."""
    per_patient = df.groupby("patient_id").agg(
        strat_grade=(label_col, "max"),
        strat_tier=("completeness_tier", "first"),
    ).reset_index()
    per_patient["combined_key"] = (
        per_patient["strat_grade"].astype(str) + "_" + per_patient["strat_tier"]
    )
    per_patient["grade_key"] = per_patient["strat_grade"].astype(str)
    return per_patient


def safe_strat_labels(pool: pd.DataFrame, primary_col: str, fallback_col: str,
                       min_count: int = 2) -> pd.Series:
    """A stratified split errors if any class has fewer than min_count members IN THE POOL
    BEING SPLIT -- and that pool shrinks at each split stage, so a class that was safe
    against the full dataset can become too rare against just the leftover "rest" portion.
    Recompute this fresh right before each split call (not once upfront) against whatever
    pool is actually being split. Collapses rare classes to a coarser fallback key, then
    merges anything still too rare directly into the majority class -- not into a shared
    "leftover" bucket of its own, since that bucket can itself end up too small."""
    counts = pool[primary_col].value_counts()
    rare = counts[counts < min_count].index
    key = pool[primary_col].where(~pool[primary_col].isin(rare), pool[fallback_col])

    counts2 = key.value_counts()
    still_rare = counts2[counts2 < min_count].index
    safe_classes = counts2.drop(still_rare)
    if len(still_rare) and len(safe_classes):
        key = key.where(~key.isin(still_rare), safe_classes.idxmax())
    return key


def split_patients_stratified(df: pd.DataFrame, label_col: str, metadata_fields: list,
                               val_frac: float, test_frac: float, seed: int):
    per_patient = build_patient_strat_key(df, label_col)

    strat1 = safe_strat_labels(per_patient, "combined_key", "grade_key")
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(sss1.split(per_patient, strat1))
    train_patients = per_patient.iloc[train_idx]
    rest_patients = per_patient.iloc[rest_idx].reset_index(drop=True)

    strat2 = safe_strat_labels(rest_patients, "combined_key", "grade_key")
    rel_test_frac = test_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test_frac, random_state=seed)
    val_idx, test_idx = next(sss2.split(rest_patients, strat2))
    val_patients, test_patients = rest_patients.iloc[val_idx], rest_patients.iloc[test_idx]

    train_ids = set(train_patients.patient_id)
    val_ids = set(val_patients.patient_id)
    test_ids = set(test_patients.patient_id)
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)

    train_df = df[df.patient_id.isin(train_ids)]
    val_df = df[df.patient_id.isin(val_ids)]
    test_df = df[df.patient_id.isin(test_ids)]
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
    """Fit only on whatever's non-missing in train -- pandas' mean/std and dropna already
    skip NaN rows automatically, so partial metadata doesn't need special handling here."""
    stats = {"numeric": {}, "categorical": {}}
    for field in numeric_fields:
        vals = train_df[field].apply(parse_locale_number)
        stats["numeric"][field] = {"mean": float(vals.mean()), "std": float(vals.std() or 1.0)}
    for field in categorical_fields:
        cats = sorted(train_df[field].dropna().astype(str).unique().tolist())
        stats["categorical"][field] = cats
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]
    label_col = dcfg["label_col"]
    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]

    df = pd.read_csv(dcfg["raw_labels_csv"])
    print(f"Raw: {len(df)} images, {df.patient_id.nunique()} patients.")

    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(parse_locale_number)

    report_missingness(df, metadata_fields + [label_col])

    before = len(df)
    df = df.dropna(subset=[label_col])  # keep every metadata-completeness level; only the label is required
    df[label_col] = df[label_col].astype(int)
    print(f"Labeled filter: dropped {before - len(df)} of {before} rows "
          f"({len(df)} remain, {df.patient_id.nunique()} patients).")

    df["completeness_tier"] = df.apply(lambda r: completeness_tier(r, metadata_fields), axis=1)
    tier_counts = df["completeness_tier"].value_counts()
    print(f"Metadata completeness: full={tier_counts.get('full', 0)} "
          f"partial={tier_counts.get('partial', 0)} none={tier_counts.get('none', 0)}")

    train_df, val_df, test_df = split_patients_stratified(
        df, label_col, metadata_fields,
        dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], dcfg["split"]["seed"],
    )
    print(f"Split sizes (images): train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Split sizes (patients): train={train_df.patient_id.nunique()} "
          f"val={val_df.patient_id.nunique()} test={test_df.patient_id.nunique()}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name} {label_col} dist: "
              f"{split_df[label_col].value_counts(normalize=True).sort_index().round(3).to_dict()}")
        print(f"  {name} completeness dist: "
              f"{split_df['completeness_tier'].value_counts(normalize=True).round(3).to_dict()}")

    vocab = build_comorbidity_vocab(train_df, dcfg["comorbidity_field"], dcfg["comorbidity_vocab_size"])
    print(f"Comorbidity vocab (top {len(vocab)} tokens from train): {vocab}")
    stats = fit_metadata_stats(train_df, dcfg["numeric_fields"], dcfg["categorical_fields"])

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    with open(out_dir / "comorbidity_vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    with open(out_dir / "metadata_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote processed splits + metadata artifacts to {out_dir}")


if __name__ == "__main__":
    main()
