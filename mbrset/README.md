# mBRSET

All mBRSET work lives in this folder. **No file outside it was modified** — every BRSET script,
config and `data/processed*` directory is untouched, so all five previous experiments stay
reproducible exactly as before. mBRSET is normalised into BRSET's column names by its own prep
script, which is why nothing downstream needed changing.

---

## Runbook (in this order — step 2 is the long pole, start it early)

### 1. Prepare the data (~1 minute)

```bash
python mbrset/prepare_data_mbrset.py --config mbrset/configs/mbrset_lora.yaml
```

Both mBRSET configs share `data/processed_mbrset`, so this runs **once** for both.

Check the printout for the per-grade test counts before training — if grade 3 lands near zero,
the 5-class run won't produce reportable per-class recall and only the 3-class run is worth the
GPU time.

### 2. Train (hours — launch first, in the background)

```bash
nohup python train.py --config mbrset/configs/mbrset_lora.yaml        > train_5class.log 2>&1 &
# then, after it finishes:
nohup python train.py --config mbrset/configs/mbrset_3class_lora.yaml > train_3class.log 2>&1 &
```

### 3. Ablation sweep on BRSET — **do this while step 2 runs**

No training and no GPU contention: image embeddings are computed once and only the metadata
branch is re-run, so the whole sweep is minutes.

```bash
# Controlled random removal (MCAR)
python scripts/ablate_metadata.py \
    --config configs/balanced_lora.yaml \
    --checkpoint runs/exp1_balanced_lora/best.pt \
    --out mbrset/results/ablation_brset_mcar.txt

# Removal biased toward healthy patients, reproducing BRSET's real pattern (MNAR)
python scripts/ablate_metadata.py \
    --config configs/balanced_lora.yaml \
    --checkpoint runs/exp1_balanced_lora/best.pt \
    --mode mnar \
    --out mbrset/results/ablation_brset_mnar.txt
```

### 4. Ablation sweep on mBRSET (after step 2)

```bash
python scripts/ablate_metadata.py \
    --config mbrset/configs/mbrset_lora.yaml \
    --checkpoint runs/mbrset_lora/best.pt \
    --out mbrset/results/ablation_mbrset_mcar.txt
```

### 5. Evaluate, as with BRSET

```bash
python evaluate.py --config mbrset/configs/mbrset_lora.yaml        --checkpoint runs/mbrset_lora/best.pt
python evaluate.py --config mbrset/configs/mbrset_3class_lora.yaml --checkpoint runs/mbrset_3class_lora/best.pt
```

---

## Why there is no balanced-cohort or missingness-token run for mBRSET

BRSET experiments 3 and 4 both existed to deal with **missing metadata**. In BRSET, patients with
complete metadata had a 29% DR rate and patients without had 4%, so a model could score well by
reading "chart is empty" instead of reading the retina. Experiment 3 rebuilt the cohort to kill
that shortcut; experiment 4 fed the missingness pattern to the gate.

mBRSET's metadata is ~99% complete — every field except `insulin_time` is missing for 0.9–2.1% of
rows. So there is no shortcut to remove and nothing for a missingness token to read, and
"complete-metadata only" and "all data" are the same cohort. Four of the five BRSET experiments
collapse into one here.

`insulin_time` is 80.3% missing, which looks like an opportunity but is not: it is absent
precisely because those patients are not on insulin, and `insulin` is already a model input. Its
missingness is a near-deterministic function of a field the model already sees, so an experiment
built on it would measure a tautology.

**The ablation sweep (step 3/4) replaces those experiments**, and is a stronger design: because
we hide the metadata ourselves at a known rate, independent of the label, any degradation is
attributable to information loss rather than to a lost shortcut. It also yields real error bars
from repeated random masks without retraining — the only confidence intervals anywhere in this
project — and tests whether the gate re-routes when metadata vanishes under conditions with no
confound left to blame.

---

## What the port had to normalise

mBRSET records the same information under different names and in a different shape:

| mBRSET | Pipeline expects | Handling |
|---|---|---|
| `patient` | `patient_id` | renamed |
| `file` = `1.1.jpg` | `image_id` = `1.1` | trailing image extension stripped; the loader appends it |
| `final_icdr` | `DR_ICDR` | set via `label_col` in the config |
| *(no binary DR column)* | `diabetic_retinopathy` | derived as `final_icdr > 0` |
| 10 binary comorbidity columns | `comorbidities` free text | set flags joined into a string |

That last row is the one that matters. BRSET fed free text into a 15-token vocabulary; mBRSET
stores the same conditions as separate 0/1 columns. Joining them back into text keeps the
metadata encoder **byte-identical** across both datasets, so results stay comparable. Adding a
new binary-flag input head would have changed the architecture and broken that comparison.

The prep script preserves the distinction between "no conditions present" (empty string, which
counts as recorded) and "flags not recorded" (NaN, which counts as missing) — the metadata
encoder treats only the second as missing.

The configs mirror BRSET's field counts exactly (2 numeric, 2 categorical, 1 comorbidity text)
for the same reason. mBRSET does offer more fields (`insurance`, `educational_level`,
`insulin_time`); using them would be a separate, non-comparable experiment.

---

## Files

| File | Purpose |
|---|---|
| `inspect_dataset.py` | Schema + statistics report on the raw CSV. Run before anything else. |
| `prepare_data_mbrset.py` | Normalises mBRSET to BRSET's columns, then reuses `scripts/prepare_data.py` for splits, stats and vocabulary. |
| `configs/mbrset_lora.yaml` | 5-class, LoRA, no missingness token. |
| `configs/mbrset_3class_lora.yaml` | Same, grades merged to none / mid / severe. |

The ablation sweep lives at `scripts/ablate_metadata.py` because it is dataset-agnostic — it runs
against any config and checkpoint, BRSET included.
