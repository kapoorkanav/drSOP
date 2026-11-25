"""
Builds per-EYE binary labels (D/G/C/A) for ODIR-5K from each eye's own diagnostic-keyword
text -- NOT from the official N/D/G/C/A/H/M/O columns, which are per-PATIENT (identical for
both eye-rows even when the eyes' own keyword text disagrees, e.g. patient 0 is
left=cataract, right=normal fundus but both rows show the same patient-level C=1).

The keyword->disease mapping below is an EXPLICIT set built from the real 94-phrase
vocabulary found in full_df.csv (not a regex/substring guess, to avoid e.g. "hypertensive
retinopathy" or "myopia retinopathy" accidentally matching "retinopathy" and being counted
as diabetic retinopathy). "Suspected"/"suspicious" variants count as positive (decided
explicitly, not a default).

Patient-level, stratified-by-disease-combination split, same robustness pattern as
scripts/prepare_data.py's safe_strat_labels (recomputed fresh at each split stage, folds
rare combinations into the current majority rather than a separate fallback bucket that can
itself be too small).

    python scripts/prepare_data_odir.py --config configs/odir_frozen.yaml
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402

DISEASE_PHRASES = {
    "D": {
        "diabetic retinopathy", "mild nonproliferative retinopathy",
        "moderate non proliferative retinopathy", "severe nonproliferative retinopathy",
        "proliferative diabetic retinopathy", "severe proliferative diabetic retinopathy",
        "suspected diabetic retinopathy", "suspected moderate non proliferative retinopathy",
        "suspicious diabetic retinopathy",
    },
    "G": {"glaucoma", "suspected glaucoma"},
    "C": {"cataract", "suspected cataract"},
    "A": {
        "age-related macular degeneration", "dry age-related macular degeneration",
        "wet age-related macular degeneration",
    },
}
DISEASE_CODES = ["D", "G", "C", "A"]

# Keyword phrases meaning "this image has no usable diagnostic content" -- distinct from
# merely low-quality (lens dust, low image quality), which still shows a real fundus and is
# kept. These are dropped entirely since there is nothing to learn a label from.
UNUSABLE_PHRASES = {"no fundus image", "image offset", "optic disk photographically invisible"}


def tokenize_keywords(text) -> list:
    if not isinstance(text, str):
        return []
    return [p.strip().lower() for p in re.split(r"[,，]", text) if p.strip()]


def match_diseases(text) -> set:
    tokens = set(tokenize_keywords(text))
    return {code for code, phrases in DISEASE_PHRASES.items() if tokens & phrases}


def is_unusable(text) -> bool:
    return bool(set(tokenize_keywords(text)) & UNUSABLE_PHRASES)


def build_eye_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per eye-image (full_df.csv already has this shape -- two rows per patient,
    one per eye -- but carries both eyes' keyword columns on every row). Picks the keyword
    text matching THAT row's own eye via its filename suffix, and derives this eye's own
    D/G/C/A labels from it."""
    rows = []
    for _, r in df.iterrows():
        fname = str(r["filename"])
        if fname.endswith("_left.jpg"):
            keywords = r["Left-Diagnostic Keywords"]
        elif fname.endswith("_right.jpg"):
            keywords = r["Right-Diagnostic Keywords"]
        else:
            continue  # shouldn't happen given the real file, but don't silently mislabel if it does
        if is_unusable(keywords):
            continue
        hits = match_diseases(keywords)
        row = {
            "image_id": fname,
            "patient_id": r["ID"],
            "eye": "left" if fname.endswith("_left.jpg") else "right",
            "keywords": keywords,
        }
        for code in DISEASE_CODES:
            row[code] = int(code in hits)
        rows.append(row)
    return pd.DataFrame(rows)


def build_patient_strat_key(df: pd.DataFrame, min_count: int = 2) -> pd.Series:
    """Union of diseases across a patient's eyes (if either eye shows it, the patient counts
    as positive for stratification), collapsed to the count-of-diseases if a specific
    combination is too rare, then merged into the current majority if still too rare --
    same two-level fallback as scripts/prepare_data.py's safe_strat_labels, recomputed fresh
    against whatever pool is actually being split (the "Reusing ... safe_strat_labels" promise
    for this script: same algorithm, inlined rather than imported, since the key itself --
    a disease combination vs. a single ordinal grade -- doesn't transfer as a shared function)."""
    per_patient = df.groupby("patient_id")[DISEASE_CODES].max()
    combo_key = per_patient.apply(lambda row: "".join(str(v) for v in row), axis=1)
    count_key = per_patient.sum(axis=1).astype(str)

    counts = combo_key.value_counts()
    rare = counts[counts < min_count].index
    key = combo_key.where(~combo_key.isin(rare), count_key)

    counts2 = key.value_counts()
    still_rare = counts2[counts2 < min_count].index
    safe = counts2.drop(still_rare)
    if len(still_rare) and len(safe):
        key = key.where(~key.isin(still_rare), safe.idxmax())
    return key


def split_patients_stratified(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int):
    strat_key = build_patient_strat_key(df)  # Series indexed by patient_id
    all_ids = strat_key.index.to_numpy()

    def safe_relabel(ids, min_count=2):
        sub_key = strat_key.loc[ids]
        counts = sub_key.value_counts()
        rare = counts[counts < min_count].index
        if not len(rare):
            return sub_key.to_numpy()
        fallback = sub_key.where(~sub_key.isin(rare), "rare")
        counts2 = fallback.value_counts()
        still_rare = counts2[counts2 < min_count].index
        safe = counts2.drop(still_rare)
        if len(still_rare) and len(safe):
            fallback = fallback.where(~fallback.isin(still_rare), safe.idxmax())
        return fallback.to_numpy()

    strat1 = safe_relabel(all_ids)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=seed)
    train_idx, rest_idx = next(sss1.split(all_ids.reshape(-1, 1), strat1))
    train_ids, rest_ids = all_ids[train_idx], all_ids[rest_idx]

    strat2 = safe_relabel(rest_ids)
    rel_test_frac = test_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=rel_test_frac, random_state=seed)
    val_idx, test_idx = next(sss2.split(rest_ids.reshape(-1, 1), strat2))
    val_ids, test_ids = rest_ids[val_idx], rest_ids[test_idx]

    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    train_df = df[df.patient_id.isin(train_ids)]
    val_df = df[df.patient_id.isin(val_ids)]
    test_df = df[df.patient_id.isin(test_ids)]
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]

    raw = pd.read_csv(dcfg["raw_csv"])
    print(f"Raw full_df.csv: {len(raw)} rows, {raw['ID'].nunique()} patients.")

    eyes = build_eye_rows(raw)
    dropped = len(raw) - len(eyes)
    print(f"Eye-images after dropping unusable ({dropped} dropped: no fundus image / "
          f"image offset / optic disk photographically invisible): {len(eyes)}")
    print(f"Patients remaining: {eyes.patient_id.nunique()}")

    print("\nPer-disease prevalence (eye-image level):")
    for code in DISEASE_CODES:
        n = eyes[code].sum()
        print(f"  {code}: {n} / {len(eyes)} ({100 * n / len(eyes):.1f}%)")

    train_df, val_df, test_df = split_patients_stratified(
        eyes, dcfg["split"]["val_frac"], dcfg["split"]["test_frac"], dcfg["split"]["seed"],
    )
    print(f"\nSplit sizes (images): train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"Split sizes (patients): train={train_df.patient_id.nunique()} "
          f"val={val_df.patient_id.nunique()} test={test_df.patient_id.nunique()}")
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rates = {code: round(100 * split_df[code].mean(), 1) for code in DISEASE_CODES}
        print(f"  {name} prevalence %: {rates}")

    out_dir = Path(dcfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    with open(out_dir / "disease_codes.json", "w") as f:
        json.dump(DISEASE_CODES, f, indent=2)

    print(f"\nWrote processed splits to {out_dir}")


if __name__ == "__main__":
    main()
