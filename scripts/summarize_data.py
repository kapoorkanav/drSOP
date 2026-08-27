"""
Prints a broad summary of the raw BRSET data: label coverage, metadata completeness
tiers (full/partial/none), diabetic-status breakdown, comorbidity coverage, and image
quality -- meant as a quick "what do we actually have" read before deciding what to
train on next.

    python scripts/summarize_data.py --config configs/default.yaml
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.text import parse_locale_number, tokenize_comorbidities  # noqa: E402


def pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)" if total else f"{n} (--%)"


def section(title: str) -> None:
    print(f"\n{'=' * 3} {title} {'=' * (60 - len(title))}")


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
    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(parse_locale_number)

    n_images, n_patients = len(df), df.patient_id.nunique()

    section("Overview")
    print(f"Images: {n_images}   Patients: {n_patients}")
    per_patient = df.groupby("patient_id").size()
    print(f"Images per patient: mean={per_patient.mean():.2f}, "
          f"min={per_patient.min()}, max={per_patient.max()}")

    section("Label coverage")
    for col in (label_col, "diabetic_retinopathy"):
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"{col}: {pct(n_images - missing, n_images)} present")
    if label_col in df.columns:
        dist = df[label_col].dropna().astype(int).value_counts(normalize=True).sort_index()
        print(f"{label_col} distribution (of labeled rows): "
              + ", ".join(f"grade {g}={p*100:.1f}%" for g, p in dist.items()))

    section("Diabetic status")
    if "diabetes" in df.columns:
        is_diabetic = df["diabetes"].astype(str).str.lower() == "yes"
        print(f"diabetes==yes: {pct(is_diabetic.sum(), n_images)}")
        if "diabetic_retinopathy" in df.columns:
            for label, mask in [("diabetic", is_diabetic), ("non-diabetic", ~is_diabetic)]:
                rate = df.loc[mask, "diabetic_retinopathy"].mean()
                print(f"  DR rate among {label}: {rate:.3f}" if pd.notna(rate) else f"  DR rate among {label}: n/a")
    else:
        print("(no 'diabetes' column found)")

    section("Metadata completeness tiers")
    n_present = df[metadata_fields].notna().sum(axis=1)
    n_fields = len(metadata_fields)
    full = n_present == n_fields
    none_ = n_present == 0
    partial = ~full & ~none_
    for name, mask in [("Full (all fields present)", full),
                        ("Partial (some fields present)", partial),
                        ("None (all fields missing)", none_)]:
        sub = df[mask]
        print(f"{name}: {pct(len(sub), n_images)}  "
              f"[{sub.patient_id.nunique()} patients]")
        if label_col in sub.columns and len(sub):
            dist = sub[label_col].dropna().astype(int).value_counts(normalize=True).sort_index()
            if len(dist):
                print("    " + label_col + " dist: "
                      + ", ".join(f"g{g}={p*100:.0f}%" for g, p in dist.items()))

    section("Per-field missingness (raw)")
    for field in metadata_fields + [dcfg.get("comorbidity_field", "comorbidities")]:
        if field in df.columns:
            missing = df[field].isna().sum()
            print(f"{field}: {pct(n_images - missing, n_images)} present")

    comorb_field = dcfg.get("comorbidity_field", "comorbidities")
    if comorb_field in df.columns:
        section("Comorbidities")
        has_any = df[comorb_field].fillna("").apply(lambda t: len(tokenize_comorbidities(t)) > 0)
        print(f"Rows with >=1 comorbidity token: {pct(has_any.sum(), n_images)}")
        counter = Counter()
        for text in df[comorb_field].fillna(""):
            counter.update(tokenize_comorbidities(text))
        print("Top 10 tokens overall: " + ", ".join(f"{t}({c})" for t, c in counter.most_common(10)))

    if "quality" in df.columns:
        section("Image quality field")
        dist = df["quality"].value_counts()
        for val, count in dist.items():
            print(f"{val}: {pct(count, n_images)}")

    if "camera" in df.columns:
        section("Camera / device")
        dist = df["camera"].value_counts()
        for val, count in dist.items():
            print(f"{val}: {pct(count, n_images)}")


if __name__ == "__main__":
    main()
