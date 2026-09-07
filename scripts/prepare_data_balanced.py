"""
Removes the metadata-missingness/DR-rate confound (found by check_missingness_bias.py: 29.2%
DR rate with full metadata vs 4.2% without) BY CONSTRUCTION, instead of hoping the model learns
to ignore it -- "is metadata present" carries zero predictive information once both groups have
an identical DR rate.

This is an ADDITIVE third data strategy alongside the existing two -- it does not modify or
replace scripts/prepare_data.py (all-data) or scripts/prepare_data_complete_metadata_only.py
(complete-metadata-only); all three remain independently runnable, each into their own
processed_dir via their own config.

Downsamples only DR-NEGATIVE patients (never DR-positive ones -- they're the scarce, valuable
class) from whichever group (full-metadata vs missing-metadata) has the lower natural DR rate,
until both groups match at the higher group's natural rate. That's the highest achievable common
rate without discarding a single DR-positive patient from either group.

Reuses scripts/prepare_data.py's stratified split / stats / vocab logic (imported, not copied) --
only the patient pre-selection step differs.

    python scripts/prepare_data_balanced.py --config configs/balanced_lora.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import prepare_data as base  # noqa: E402
from drsop.config import load_config, resolve  # noqa: E402


def balance_negatives(pos_a: pd.DataFrame, neg_a: pd.DataFrame,
                       pos_b: pd.DataFrame, neg_b: pd.DataFrame, seed: int):
    """Keeps every positive patient from both groups; downsamples only the negatives of
    whichever group has the lower natural rate, up to the other group's (higher) rate --
    the highest common rate achievable without discarding any positive."""
    rate_a = len(pos_a) / (len(pos_a) + len(neg_a)) if (len(pos_a) + len(neg_a)) else 0.0
    rate_b = len(pos_b) / (len(pos_b) + len(neg_b)) if (len(pos_b) + len(neg_b)) else 0.0
    target_r = max(rate_a, rate_b)

    def trim(pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
        if target_r <= 0:
            return neg
        n_keep = min(len(neg), round(len(pos) * (1 - target_r) / target_r))
        return neg if n_keep >= len(neg) else neg.sample(n=n_keep, random_state=seed)

    if rate_a >= rate_b:
        neg_b = trim(pos_b, neg_b)
    else:
        neg_a = trim(pos_a, neg_a)
    return pos_a, neg_a, pos_b, neg_b, target_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]
    label_col = dcfg["label_col"]
    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    seed = dcfg["split"]["seed"]

    if "diabetic_retinopathy" not in pd.read_csv(dcfg["raw_labels_csv"], nrows=0).columns:
        raise RuntimeError("Need a 'diabetic_retinopathy' column in the raw CSV to balance on.")

    df = pd.read_csv(dcfg["raw_labels_csv"])
    print(f"Raw: {len(df)} images, {df.patient_id.nunique()} patients.")

    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(base.parse_locale_number)

    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    df["completeness_tier"] = df.apply(lambda r: base.completeness_tier(r, metadata_fields), axis=1)

    per_patient = df.groupby("patient_id").agg(
        tier=("completeness_tier", "first"),
        has_dr=("diabetic_retinopathy", "max"),
    ).reset_index()

    group_a = per_patient[per_patient["tier"] == "full"]   # "has metadata"
    group_b = per_patient[per_patient["tier"] != "full"]   # "missing metadata" (partial or none)
    pos_a, neg_a = group_a[group_a.has_dr == 1], group_a[group_a.has_dr == 0]
    pos_b, neg_b = group_b[group_b.has_dr == 1], group_b[group_b.has_dr == 0]

    print(f"Before balancing: full-metadata DR rate = {len(pos_a)}/{len(group_a)} "
          f"({100 * len(pos_a) / max(len(group_a), 1):.1f}%)   "
          f"missing-metadata DR rate = {len(pos_b)}/{len(group_b)} "
          f"({100 * len(pos_b) / max(len(group_b), 1):.1f}%)")

    pos_a, neg_a, pos_b, neg_b, target_r = balance_negatives(pos_a, neg_a, pos_b, neg_b, seed)
    kept_patients = pd.concat([pos_a, neg_a, pos_b, neg_b])["patient_id"]

    print(f"Balanced to a common DR rate of {target_r * 100:.1f}% in both groups "
          f"(all {len(pos_a) + len(pos_b)} DR-positive patients kept, negatives trimmed).")
    print(f"Kept {len(kept_patients)} of {len(per_patient)} patients "
          f"(full-metadata: {len(pos_a) + len(neg_a)}, missing-metadata: {len(pos_b) + len(neg_b)}).")

    df = df[df.patient_id.isin(kept_patients)]
    print(f"Resulting images: {len(df)}")
    base.report_missingness(df, metadata_fields + [label_col])

    train_df, val_df, test_df = base.split_patients_stratified(
        df, label_col, metadata_fields, dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], seed,
    )
    print(f"Split sizes (images): train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Split sizes (patients): train={train_df.patient_id.nunique()} "
          f"val={val_df.patient_id.nunique()} test={test_df.patient_id.nunique()}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name} {label_col} dist: "
              f"{split_df[label_col].value_counts(normalize=True).sort_index().round(3).to_dict()}")
        print(f"  {name} completeness dist: "
              f"{split_df['completeness_tier'].value_counts(normalize=True).round(3).to_dict()}")

    vocab = base.build_comorbidity_vocab(train_df, dcfg["comorbidity_field"], dcfg["comorbidity_vocab_size"])
    print(f"Comorbidity vocab (top {len(vocab)} tokens from train): {vocab}")
    stats = base.fit_metadata_stats(train_df, dcfg["numeric_fields"], dcfg["categorical_fields"])

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
