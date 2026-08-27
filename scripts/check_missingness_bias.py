"""
Checks whether metadata missingness is informative -- i.e. whether patients missing
diabetes_time_y/insuline/patient_age have a different DR_ICDR distribution than
complete-metadata patients. If so, a model trained on missingness patterns could
learn "fields missing -> predict grade 0" as a shortcut instead of reading the image.

    python scripts/check_missingness_bias.py --config configs/default.yaml
    python scripts/check_missingness_bias.py --config configs/default.yaml --diabetic-only

--diabetic-only restricts to diabetes=="yes" first. The full-cohort run found a severe
confound (missingness ~= "not diabetic" ~= "can't have DR"). This flag tests whether that
confound is specifically about diabetic-vs-not, or whether it persists even within diabetics
(in which case it's a narrower charting-completeness signal, safer to use missing-tokens for).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.text import parse_locale_number  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--diabetic-only", action="store_true",
                         help='Restrict to diabetes=="yes" before comparing.')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]
    label_col = dcfg["label_col"]

    df = pd.read_csv(dcfg["raw_labels_csv"])
    for field in dcfg["numeric_fields"]:
        df[field] = df[field].apply(parse_locale_number)

    metadata_fields = dcfg["numeric_fields"] + dcfg["categorical_fields"]
    df = df.dropna(subset=[label_col])  # only rows we could ever train on either way
    df[label_col] = df[label_col].astype(int)

    if args.diabetic_only:
        before = len(df)
        df = df[df["diabetes"].astype(str).str.lower() == "yes"]
        print(f"--diabetic-only: kept {len(df)} of {before} rows where diabetes==\"yes\"\n")

    is_complete = df[metadata_fields].notna().all(axis=1)
    complete_df, incomplete_df = df[is_complete], df[~is_complete]

    print(f"Complete-metadata: {len(complete_df)} images, {complete_df.patient_id.nunique()} patients")
    print(f"Incomplete-metadata: {len(incomplete_df)} images, {incomplete_df.patient_id.nunique()} patients")

    print(f"\n{label_col} distribution, complete-metadata subset (%):")
    print((complete_df[label_col].value_counts(normalize=True).sort_index() * 100).round(1))

    print(f"\n{label_col} distribution, incomplete-metadata subset (%):")
    print((incomplete_df[label_col].value_counts(normalize=True).sort_index() * 100).round(1))

    contingency = pd.crosstab(is_complete, df[label_col])
    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"\nChi-square test (is-complete-metadata vs {label_col}): "
          f"chi2={chi2:.2f}, p={p:.6f}")
    if p < 0.05:
        print("=> Statistically significant association: missingness IS informative about "
              "the label. A missing-token model risks learning a missingness shortcut -- "
              "worth checking which grades skew which way (see the two distributions above) "
              "before trusting it blindly.")
    else:
        print("=> No significant association detected: missingness looks roughly independent "
              "of the label. Lower risk of a missingness shortcut.")

    # Same check on the binary field, if present, since it's populated for the full cohort
    if "diabetic_retinopathy" in df.columns:
        print(f"\ndiabetic_retinopathy rate, complete-metadata: "
              f"{complete_df['diabetic_retinopathy'].mean():.3f}")
        print(f"diabetic_retinopathy rate, incomplete-metadata: "
              f"{incomplete_df['diabetic_retinopathy'].mean():.3f}")


if __name__ == "__main__":
    main()
