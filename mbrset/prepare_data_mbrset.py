"""
Builds train/val/test splits for mBRSET, reusing the BRSET pipeline unchanged.

mBRSET stores the same information under different names and in a different shape. Rather than
edit the BRSET scripts (which would touch every previous experiment), this script normalises
mBRSET into the column names the existing pipeline already expects, then hands off to
scripts/prepare_data.py's own split/stats/vocab functions. Nothing in scripts/, src/ or configs/
is modified, so every BRSET run stays reproducible exactly as before.

The four normalisations:

  patient          -> patient_id            (the pipeline groups on patient_id)
  file "1.1.jpg"   -> image_id "1.1"        (the loader appends the extension itself)
  final_icdr > 0   -> diabetic_retinopathy  (mBRSET has no binary DR column; BRSET did)
  10 binary flags  -> comorbidities         (free text, e.g. "systemic_hypertension,smoking")

That last one matters: BRSET fed a free-text comorbidity field into a 15-token vocabulary, and
mBRSET records the same conditions as separate 0/1 columns. Joining the set flags back into text
means the metadata encoder is byte-identical across both datasets -- no new input head, so
results stay comparable. The alternative (a new binary-flag branch) would change the
architecture and break that comparison.

    python mbrset/prepare_data_mbrset.py --config mbrset/configs/mbrset_lora.yaml

Writes train.csv / val.csv / test.csv / comorbidity_vocab.json / metadata_stats.json into the
config's processed_dir, exactly like the BRSET prep scripts do.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import prepare_data as base  # noqa: E402
from drsop.config import load_config, resolve  # noqa: E402

# mBRSET's 0/1 comorbidity + lifestyle columns, joined into the free-text field the existing
# metadata encoder consumes. Order is fixed so the vocabulary is stable between runs.
COMORBIDITY_FLAGS = [
    "systemic_hypertension",
    "oraltreatment_dm",
    "vascular_disease",
    "diabetic_foot",
    "acute_myocardial_infarction",
    "neuropathy",
    "nephropathy",
    "obesity",
    "alcohol_consumption",
    "smoking",
]

RENAMES = {"patient": "patient_id"}


def synthesize_comorbidity_text(df: pd.DataFrame, flags: list) -> pd.Series:
    """One comma-separated string per row naming the conditions that are set.

    A row where every flag is 0 becomes "" (recorded, none present) -- distinct from a row where
    the flags themselves are absent, which becomes NaN (not recorded). MetadataProcessor treats
    only the NaN case as missing, so that distinction has to survive.
    """
    available = [f for f in flags if f in df.columns]
    if not available:
        raise RuntimeError(f"None of the comorbidity flags {flags} are in the CSV.")

    present = df[available].notna().any(axis=1)
    text = df[available].apply(
        lambda row: ",".join(name for name, val in row.items() if val == 1), axis=1
    )
    return text.where(present, other=pd.NA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = resolve(load_config(args.config), ROOT)
    dcfg = cfg["data"]
    label_col = dcfg["label_col"]
    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]

    df = pd.read_csv(dcfg["raw_labels_csv"])
    print(f"Raw mBRSET: {len(df)} images, {df['patient'].nunique()} patients, "
          f"{len(df.columns)} columns.")

    # --- normalise into the BRSET column contract ---
    df = df.rename(columns=RENAMES)

    # "1.1.jpg" -> "1.1". The loader tries .jpg/.jpeg/.png itself, and these stems contain dots
    # of their own, so strip only a trailing image extension.
    df["image_id"] = df["file"].astype(str).str.replace(
        r"\.(jpe?g|png|tif{1,2}|bmp|webp)$", "", regex=True, case=False)

    df[dcfg["comorbidity_field"]] = synthesize_comorbidity_text(df, COMORBIDITY_FLAGS)
    n_text = df[dcfg["comorbidity_field"]].notna().sum()
    print(f"Synthesized '{dcfg['comorbidity_field']}' from {len(COMORBIDITY_FLAGS)} binary "
          f"flags: {n_text}/{len(df)} rows have it recorded.")

    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(base.parse_locale_number)

    base.report_missingness(df, metadata_fields + [label_col])

    before = len(df)
    df = df.dropna(subset=[label_col])
    df[label_col] = df[label_col].astype(int)
    print(f"Labeled filter: dropped {before - len(df)} of {before} rows with no {label_col} "
          f"({len(df)} remain, {df.patient_id.nunique()} patients).")

    # mBRSET has no binary DR column; derive it so downstream tooling that expects one works.
    df["diabetic_retinopathy"] = (df[label_col] > 0).astype(int)

    df["completeness_tier"] = df.apply(
        lambda r: base.completeness_tier(r, metadata_fields), axis=1)
    tiers = df["completeness_tier"].value_counts()
    print(f"Metadata completeness: full={tiers.get('full', 0)} "
          f"partial={tiers.get('partial', 0)} none={tiers.get('none', 0)}")
    print("  (mBRSET metadata is near-complete, so 'full' should dominate -- this is why the")
    print("   balanced-cohort and missingness-token experiments do not transfer to it.)")

    train_df, val_df, test_df = base.split_patients_stratified(
        df, label_col, metadata_fields,
        dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], dcfg["split"]["seed"],
    )
    print(f"Split sizes (images): train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Split sizes (patients): train={train_df.patient_id.nunique()} "
          f"val={val_df.patient_id.nunique()} test={test_df.patient_id.nunique()}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df[label_col].value_counts().sort_index().to_dict()
        print(f"  {name} {label_col} counts: {counts}")

    print("\nPer-grade test counts decide whether 5-class recall is reportable at all.")
    print("On BRSET, grades 1 and 3 fell to n=22 and n=13 test images.")

    vocab = base.build_comorbidity_vocab(
        train_df, dcfg["comorbidity_field"], dcfg["comorbidity_vocab_size"])
    print(f"\nComorbidity vocab (top {len(vocab)} from train): {vocab}")
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

    print(f"\nWrote processed splits + metadata artifacts to {out_dir}")


if __name__ == "__main__":
    main()
