"""
Schema + statistics report for mBRSET, to run BEFORE any mBRSET config exists.

Standalone on purpose: imports nothing from src/drsop and reads no config, so it works on a
fresh checkout against the raw PhysioNet download. Everything the port needs is decided from
its output -- which column is the patient id, which is the image id, which column carries the
ICDR grade, whether a binary DR flag exists, which metadata fields survive, and whether the
metadata-missingness/DR-rate confound that drove the BRSET balanced cohort is present here too.

    python mbrset/inspect_dataset.py

Writes the same report to mbrset/schema_report.txt so it can be pasted back verbatim.
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_CSV = ("/media/DATA/users/shared/file/"
               "mbrset-a-mobile-brazilian-retinal-dataset-1.0/labels_mbrset.csv")
DEFAULT_IMAGES = ("/media/DATA/users/shared/file/"
                  "mbrset-a-mobile-brazilian-retinal-dataset-1.0/images")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")

# Low-cardinality columns get their full value counts; numeric ones get a five-number summary.
CATEGORICAL_MAX_UNIQUE = 25


class Tee:
    """Prints and captures, so the report lands on stdout and in a pasteable file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def rule(self, title: str) -> None:
        self("")
        self("=" * 78)
        self(title)
        self("=" * 78)

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n")


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def coerce_numeric(series: pd.Series) -> pd.Series:
    """mBRSET may use comma decimals like BRSET did ('10,00'), which pandas reads as object.
    Try a locale-tolerant parse so a numeric column mis-typed as text is still recognised."""
    if is_numeric(series):
        return series
    cleaned = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def describe_columns(out: Tee, df: pd.DataFrame) -> dict:
    """Per-column dtype, missingness, cardinality and values. Returns numeric-coercion info."""
    out.rule("1. COLUMNS -- dtype, missingness, cardinality")
    out(f"{'column':<28} {'dtype':<10} {'missing':>14} {'unique':>8}")
    out("-" * 78)
    numeric_like = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        pct = 100 * n_missing / len(df) if len(df) else 0.0
        out(f"{col:<28} {str(df[col].dtype):<10} {n_missing:>6} ({pct:>5.1f}%) "
            f"{df[col].nunique(dropna=True):>8}")
        if not is_numeric(df[col]):
            coerced = coerce_numeric(df[col])
            present = df[col].notna().sum()
            # "Numeric but stored as text" only if nearly every present value parses.
            if present and coerced.notna().sum() >= 0.95 * present:
                numeric_like[col] = coerced
    if numeric_like:
        out("")
        out("Text columns that parse as numeric (comma decimals or stray whitespace):")
        for col in numeric_like:
            out(f"  {col}")
    return numeric_like


def describe_values(out: Tee, df: pd.DataFrame, numeric_like: dict) -> None:
    out.rule("2. VALUES -- full counts for categoricals, summary for numerics")
    for col in df.columns:
        series = numeric_like.get(col, df[col])
        n_unique = series.nunique(dropna=True)
        out("")
        out(f"--- {col}  (unique={n_unique})")
        if is_numeric(series) and n_unique > CATEGORICAL_MAX_UNIQUE:
            desc = series.describe()
            out(f"    min={desc['min']:.4g}  p25={desc['25%']:.4g}  median={desc['50%']:.4g}  "
                f"p75={desc['75%']:.4g}  max={desc['max']:.4g}  mean={desc['mean']:.4g}")
        elif n_unique <= CATEGORICAL_MAX_UNIQUE:
            counts = series.value_counts(dropna=False).sort_index(
                key=lambda idx: idx.astype(str))
            for value, count in counts.items():
                label = "<missing>" if pd.isna(value) else repr(value)
                out(f"    {label:<28} {count:>7}  ({100 * count / len(df):>5.1f}%)")
        else:
            sample = series.dropna().astype(str).unique()[:6]
            out(f"    high-cardinality free text; first values: {list(sample)}")


def guess_roles(out: Tee, df: pd.DataFrame) -> dict:
    """Heuristics only -- the printed evidence is what decides, not these guesses."""
    out.rule("3. ROLE CANDIDATES -- which column is which")
    roles = {}

    id_like = [c for c in df.columns
               if re.search(r"(^|_)(id|file|image|img|name|path)(_|$)", c, re.I)]
    patient_like = [c for c in id_like if re.search(r"patient|subject|pac", c, re.I)]
    out("")
    out(f"Columns that look like identifiers: {id_like or 'none'}")
    out(f"  ...of those, patient-shaped:     {patient_like or 'none'}")
    for col in id_like:
        n_unique = df[col].nunique(dropna=True)
        per_group = len(df) / n_unique if n_unique else 0
        kind = "one row per value (image-level)" if n_unique == len(df) else \
               f"{per_group:.2f} rows per value (groups them)"
        out(f"    {col:<24} unique={n_unique:<7} {kind}")
    roles["id_candidates"] = id_like

    dr_like = [c for c in df.columns if re.search(r"icdr|retinopath|\bdr\b|_dr|dr_", c, re.I)]
    out("")
    out(f"Columns that look DR-related: {dr_like or 'none'}")
    for col in dr_like:
        vals = sorted(df[col].dropna().unique().tolist(), key=str)
        shape = "binary" if len(vals) == 2 else f"{len(vals)}-valued"
        out(f"    {col:<24} {shape:<12} values={vals[:10]}")
    roles["dr_candidates"] = dr_like

    # Anything DR-related is a LABEL, never a model input -- exclude it even though names like
    # "diabetic_retinopathy" match the metadata patterns below.
    meta_like = [c for c in df.columns
                 if c not in dr_like
                 and re.search(r"age|sex|gender|diabet|insulin|insulin[ae]|comorbid|"
                               r"smok|alcohol|hypert|bmi|hba1c|glucose|duration|time",
                               c, re.I)]
    out("")
    out(f"Columns that look like patient metadata: {meta_like or 'none'}")
    out("  (DR-related columns are excluded here -- they are labels, not inputs)")
    roles["metadata_candidates"] = meta_like

    quality_like = [c for c in df.columns if re.search(r"qualit|gradab|artifact|focus|"
                                                       r"illum|blur", c, re.I)]
    out(f"Columns that look like image quality: {quality_like or 'none'}")
    roles["quality_candidates"] = quality_like
    return roles


def check_images(out: Tee, df: pd.DataFrame, images_dir: Path, roles: dict) -> None:
    out.rule("4. IMAGES ON DISK -- do the CSV ids resolve to files?")
    if not images_dir.exists():
        out(f"  images_dir does not exist: {images_dir}")
        return

    by_ext = Counter()
    stems = set()
    names = set()
    for path in images_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            by_ext[path.suffix.lower()] += 1
            stems.add(path.stem)
            names.add(path.name)
    total = sum(by_ext.values())
    out("")
    out(f"  {total} image files under {images_dir}")
    for ext, count in by_ext.most_common():
        out(f"    {ext:<8} {count}")
    if not total:
        return

    nested = {p.parent for p in images_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
    out(f"  spread over {len(nested)} directory/directories "
        f"({'flat' if len(nested) <= 1 else 'NESTED -- the loader globs one dir only'})")
    out(f"  example filenames: {sorted(names)[:5]}")

    out("")
    out("  Match rate of each identifier column against those files:")
    out(f"  {'column':<24} {'as stem':>12} {'as filename':>14}")
    for col in roles["id_candidates"]:
        vals = df[col].dropna().astype(str)
        if vals.empty:
            continue
        stem_hits = vals.isin(stems).mean() * 100
        name_hits = vals.isin(names).mean() * 100
        out(f"  {col:<24} {stem_hits:>11.1f}% {name_hits:>13.1f}%")
    out("")
    out("  (the image-id column is whichever reaches ~100% -- 'as stem' means the CSV")
    out("   stores the name without an extension, which is what the loader expects)")


def patient_structure(out: Tee, df: pd.DataFrame, patient_col: str, dr_col: str = None) -> None:
    out.rule(f"5. PATIENT STRUCTURE -- grouping by '{patient_col}'")
    n_patients = df[patient_col].nunique()
    per_patient = df.groupby(patient_col).size()
    out("")
    out(f"  {len(df)} rows over {n_patients} patients")
    out(f"  images per patient: min={per_patient.min()} median={per_patient.median():.0f} "
        f"max={per_patient.max()} mean={per_patient.mean():.2f}")
    out(f"  distribution: {per_patient.value_counts().sort_index().to_dict()}")
    out("")
    out("  (patient-level splitting is required -- both eyes of a patient must stay on the")
    out("   same side of the split, so these counts set how coarse the split can be)")


def label_distribution(out: Tee, df: pd.DataFrame, dr_candidates: list,
                        patient_col: str = None) -> None:
    out.rule("6. LABEL DISTRIBUTION -- per DR-related column")
    for col in dr_candidates:
        out("")
        out(f"--- {col}")
        counts = df[col].value_counts(dropna=False).sort_index(key=lambda i: i.astype(str))
        for value, count in counts.items():
            label = "<missing>" if pd.isna(value) else repr(value)
            out(f"    {label:<16} {count:>7} images ({100 * count / len(df):>5.1f}%)")
        if patient_col and patient_col in df.columns:
            worst = df.groupby(patient_col)[col].max()
            pc = worst.value_counts(dropna=False).sort_index(key=lambda i: i.astype(str))
            out("    by patient (worst grade per patient):")
            for value, count in pc.items():
                label = "<missing>" if pd.isna(value) else repr(value)
                out(f"      {label:<14} {count:>7} patients "
                    f"({100 * count / len(worst):>5.1f}%)")
        out("")
        out("    NOTE: a 5-class test split is only reportable if every grade survives at")
        out("    ~15% of patients. On BRSET grades 1 and 3 fell to n=22 and n=13 test images.")


def missingness_confound(out: Tee, df: pd.DataFrame, metadata_fields: list,
                          dr_col: str, patient_col: str) -> None:
    """The check that produced the balanced cohort on BRSET: if DR prevalence differs between
    patients with complete metadata and patients without, 'metadata is present' leaks the
    label and any model can score well without reading the image."""
    out.rule("7. METADATA-MISSINGNESS CONFOUND -- the check that drove the balanced cohort")
    out("")
    out(f"  metadata fields used: {metadata_fields}")
    out(f"  DR column: {dr_col}   patient column: {patient_col}")
    if not metadata_fields or dr_col not in df.columns:
        out("  SKIPPED -- pass --metadata-fields and --dr-col once the columns are known.")
        return

    present = df[metadata_fields].notna().sum(axis=1)
    tier = pd.Series("partial", index=df.index)
    tier[present == len(metadata_fields)] = "full"
    tier[present == 0] = "none"

    per_patient = pd.DataFrame({
        "patient": df[patient_col],
        "tier": tier,
        "dr": pd.to_numeric(df[dr_col], errors="coerce"),
    }).groupby("patient").agg(tier=("tier", "first"), dr=("dr", "max")).reset_index()

    out("")
    out(f"  {'tier':<10} {'patients':>10} {'DR-positive':>13} {'DR rate':>10}")
    for name in ("full", "partial", "none"):
        sub = per_patient[per_patient.tier == name]
        if not len(sub):
            continue
        pos = int((sub.dr > 0).sum())
        out(f"  {name:<10} {len(sub):>10} {pos:>13} {100 * pos / len(sub):>9.1f}%")

    full = per_patient[per_patient.tier == "full"]
    rest = per_patient[per_patient.tier != "full"]
    if len(full) and len(rest):
        r_full = (full.dr > 0).mean()
        r_rest = (rest.dr > 0).mean()
        out("")
        out(f"  full-metadata DR rate {100 * r_full:.1f}%  vs  "
            f"missing-metadata DR rate {100 * r_rest:.1f}%")
        out(f"  difference: {100 * (r_full - r_rest):+.1f} percentage points")
        if abs(r_full - r_rest) < 0.03:
            out("  => Groups already match. The balanced cohort would be a near no-op here,")
            out("     and experiments 3/4 collapse onto experiment 2. Worth knowing early.")
        else:
            out("  => Confound present, as on BRSET. The balanced cohort applies.")
            target = max(r_full, r_rest)
            keep = 0
            for grp in (full, rest):
                pos = int((grp.dr > 0).sum())
                neg = len(grp) - pos
                n_neg_keep = min(neg, round(pos * (1 - target) / target)) if target else neg
                keep += pos + n_neg_keep
            out("")
            out(f"  Projected balanced cohort: ~{keep} of {len(per_patient)} patients "
                f"({100 * keep / len(per_patient):.0f}%), common DR rate {100 * target:.1f}%")
            out(f"  At a 70/15/15 split that is ~{round(keep * 0.15)} test patients.")
            out("  (BRSET kept 1,918 of 8,524 patients and yielded 288 test patients / 541 images)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    parser.add_argument("--patient-col", default=None,
                        help="override the auto-detected patient id column")
    parser.add_argument("--dr-col", default=None,
                        help="the DR column to use for the confound check (section 7)")
    parser.add_argument("--metadata-fields", nargs="*", default=None,
                        help="fields whose presence defines metadata completeness (section 7)")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "schema_report.txt"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"No CSV at {csv_path}")

    out = Tee()
    df = pd.read_csv(csv_path)
    out(f"mBRSET schema report")
    out(f"  csv:    {csv_path}")
    out(f"  images: {args.images_dir}")
    out(f"  shape:  {df.shape[0]} rows x {df.shape[1]} columns")

    numeric_like = describe_columns(out, df)
    describe_values(out, df, numeric_like)
    roles = guess_roles(out, df)
    check_images(out, df, Path(args.images_dir), roles)

    patient_col = args.patient_col
    if not patient_col:
        patient_like = [c for c in roles["id_candidates"]
                        if re.search(r"patient|subject|pac", c, re.I)]
        # Fall back to the identifier that groups rows rather than the one that is unique.
        grouping = [c for c in roles["id_candidates"] if df[c].nunique() < len(df)]
        patient_col = (patient_like or grouping or [None])[0]
    if patient_col:
        patient_structure(out, df, patient_col)
    else:
        out.rule("5. PATIENT STRUCTURE")
        out("  Could not identify a patient column -- rerun with --patient-col.")

    label_distribution(out, df, roles["dr_candidates"], patient_col)

    dr_col = args.dr_col
    if not dr_col:
        binary = [c for c in roles["dr_candidates"] if df[c].nunique(dropna=True) == 2]
        dr_col = (binary or roles["dr_candidates"] or [None])[0]
    meta_fields = args.metadata_fields
    if meta_fields is None:
        meta_fields = [c for c in roles["metadata_candidates"] if c in df.columns]
    if dr_col and patient_col:
        missingness_confound(out, df, meta_fields, dr_col, patient_col)
    else:
        out.rule("7. METADATA-MISSINGNESS CONFOUND")
        out("  SKIPPED -- need both --dr-col and --patient-col.")

    out.rule("DONE")
    out("")
    out("Paste this report back. It decides: patient/image id columns, the ICDR grade column,")
    out("the binary DR column, which metadata fields the encoder gets, and whether the")
    out("balanced cohort is viable at this dataset's size.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
