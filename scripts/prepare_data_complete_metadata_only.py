"""
PRESERVED for reproducing the original complete-metadata-only experiments (frozen-baseline
test QWK 0.7043, LoRA test QWK 0.8351) -- superseded by scripts/prepare_data.py, which now
uses all labeled data with a stratified split instead of dropping incomplete-metadata rows.

Pair this with configs/legacy_frozen.yaml or configs/legacy_lora.yaml (processed_dir points
at data/processed_complete_metadata_only, output_dir at runs/exp1_legacy*) so it can never
collide with the current all-data split/experiments.

Run once before training:
    python scripts/prepare_data_complete_metadata_only.py --config configs/legacy_frozen.yaml

Produces, under data.processed_dir:
    train.csv, val.csv, test.csv    (rows with complete metadata and a non-null label_col)
    comorbidity_vocab.json          (top-K free-text tokens found in the training split)
    metadata_stats.json             (numeric field mean/std + categorical vocabs, fit on train only)
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
from drsop.data.text import parse_locale_number, tokenize_comorbidities  # noqa: E402


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

    df = pd.read_csv(dcfg["raw_labels_csv"])
    print(f"Raw: {len(df)} images, {df.patient_id.nunique()} patients.")

    # Coerce numeric fields up front: comma-decimals (Brazilian locale, e.g. "10,00") get
    # parsed correctly, and anything genuinely unparseable (data-entry typos etc.) becomes NaN
    # here so it's caught by the same complete-metadata filter as truly missing values below,
    # instead of surfacing as a crash later during stats fitting or training.
    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(parse_locale_number)

    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    report_missingness(df, metadata_fields + [label_col])

    before = len(df)
    df = df.dropna(subset=metadata_fields + [label_col])
    df[label_col] = df[label_col].astype(int)
    print(f"Complete-metadata + labeled filter: dropped {before - len(df)} of {before} rows "
          f"({len(df)} remain, {df.patient_id.nunique()} patients).")

    train_df, val_df, test_df = split_patients(
        df, dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], dcfg["split"]["seed"]
    )
    print(f"Split sizes (images): train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Split sizes (patients): train={train_df.patient_id.nunique()} "
          f"val={val_df.patient_id.nunique()} test={test_df.patient_id.nunique()}")
    print(f"Train label distribution ({label_col}): "
          f"{train_df[label_col].value_counts().sort_index().to_dict()}")

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
