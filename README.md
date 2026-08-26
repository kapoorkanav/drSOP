# drSOP — Gated Multimodal Diabetic Retinopathy Grading

Two-stage POC: stage 1 predicts DR present/absent, stage 2 predicts ICDR severity grade (0-4),
each from a single-eye fundus image + patient metadata, using a RETFound image encoder, a small
transformer metadata encoder, and a transformer-based gate that fuses the two embeddings
per-dimension:

    fused = alpha * image_emb + (1 - alpha) * meta_emb      (alpha in [0,1]^d, from GateTransformer)

The two stages are independent models (own weights, own output dir) sharing the same architecture
and the same underlying patient split — not a shared-backbone cascade. Stage 2 trains only on the
subset of patients that have an ICDR grade (the diabetic-graded cohort), which is much smaller than
stage 1's cohort (DR present/absent is recorded more broadly).

Both stages only use patients with **complete metadata** (no imputation) — `scripts/prepare_data.py`
unconditionally drops any row missing `patient_age`/`diabetes_time`/`patient_sex`/`insulin_use`
before splitting. Both eyes of a patient always land in the same split (train/val/test), and a
patient's split assignment is identical across both stages, since both are carved out of one
global patient-level split.

## Setup (on the training VM)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash scripts/setup_retfound.sh   # clones third_party/RETFound_MAE
# then, manually (both are gated/credentialed downloads):
#   - RETFound_cfp_weights.pth -> weights/RETFound_cfp_weights.pth
#   - BRSET labels.csv + fundus_photos/ -> data/
```

`timm==0.4.12` is pinned in requirements.txt because RETFound_MAE's `models_vit.py` targets that
older timm API — don't upgrade it without checking compatibility.

## Pipeline

```bash
python scripts/prepare_data.py --config configs/default.yaml   # one-time: split + metadata stats
python train.py --config configs/default.yaml --stage 1        # DR present/absent -> runs/exp1/stage1/
python train.py --config configs/default.yaml --stage 2        # ICDR severity      -> runs/exp1/stage2/
```

`prepare_data.py` writes `{train,val,test}_stage1.csv` and `{train,val,test}_stage2.csv`, plus
`comorbidity_vocab.json`/`metadata_stats.json` fit on stage 1's train split (the broader pool;
stage 2's train patients are a strict subset of it), all under `data/processed/`. It prints
per-field missingness and the resulting image/patient counts for both stages — check that output
to see the actual usable dataset size once you run it against the real BRSET CSV.

## Repo layout

- `configs/default.yaml` — all hyperparameters, including per-stage `model.num_classes`.
- `scripts/prepare_data.py` — complete-metadata filter, one global patient-level split, two
  label-filtered stage views, metadata preprocessing.
- `src/drsop/data/` — `MetadataProcessor`, `BRSETDataset`.
- `src/drsop/models/` — `RetfoundEncoder` (frozen ViT-L), `MetadataEncoder`, `GateTransformer`,
  `MLPHead`, wired together in `fusion_model.DRFusionModel`.
- `train.py` — training loop for one stage (`--stage 1` or `--stage 2`), AMP, early stopping on
  validation quadratic weighted kappa (QWK).

## Notes / assumptions to verify once real data is available

- BRSET image file extension is assumed `.jpg`/`.jpeg`/`.png` (checked in that order) — confirm
  against the actual `fundus_photos/` contents.
- Comorbidity handling is data-driven: `prepare_data.py` tokenizes the free-text `comorbidities`
  field and keeps the top-K most frequent tokens from the training split as multi-hot flags
  (`configs/default.yaml: data.comorbidity_vocab_size`). Inspect the printed vocab after running
  `prepare_data.py` — if it's noisy (e.g. free-text punctuation artifacts), tighten the tokenizer
  in `src/drsop/data/text.py`.
- RETFound is frozen by default (`model.freeze_retfound: true`). If validation QWK plateaus,
  the next step is unfreezing the last few ViT blocks or adding LoRA adapters — not full
  fine-tuning first, given ~16k images vs. a ViT-Large.
