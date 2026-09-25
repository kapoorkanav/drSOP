# mBRSET

All mBRSET work lives in this folder. Nothing here touches the BRSET pipeline — the existing
scripts, configs and `data/processed*` directories are unchanged and still runnable exactly as
before.

mBRSET is the handheld-camera companion to BRSET. RETFound was pretrained on tabletop colour
fundus photography, so mBRSET results are a **separate** result, not a cross-check of the BRSET
numbers — the domain differs.

## Step 1 — inspect the raw dataset (run this first)

```bash
python mbrset/inspect_dataset.py
```

Defaults point at:

- `/media/DATA/users/shared/file/mbrset-a-mobile-brazilian-retinal-dataset-1.0/labels_mbrset.csv`
- `/media/DATA/users/shared/file/mbrset-a-mobile-brazilian-retinal-dataset-1.0/images`

Override with `--csv` and `--images-dir`. The report prints to stdout and is saved to
`mbrset/schema_report.txt`.

It answers the seven things the port depends on:

| § | Question |
|---|---|
| 1–2 | Every column: dtype, missingness, cardinality, value counts |
| 3 | Which column is the patient id, the image id, the ICDR grade, the binary DR flag, the metadata fields |
| 4 | Do the CSV ids resolve to files on disk, and is the image directory flat or nested |
| 5 | Images per patient — sets how coarse a patient-level split can be |
| 6 | Grade distribution per image and per patient, for every DR-related column |
| 7 | Whether the metadata-missingness/DR-rate confound exists here, and how many patients a balanced cohort would leave |

Sections 3 and 7 use name heuristics. The printed evidence decides, not the guesses. Once the
real column names are known, re-run section 7 properly:

```bash
python mbrset/inspect_dataset.py \
    --patient-col <patient column> \
    --dr-col <binary DR column> \
    --metadata-fields <field> <field> ...
```

## Step 2 onwards — not built yet

The port needs three column names lifted out of the BRSET scripts into config
(`patient_id`, `image_id`, `diabetic_retinopathy` are currently hardcoded); everything else is
already config-driven. That refactor is backward-compatible and comes after the report, so the
configs are written against the real schema rather than a guess.
