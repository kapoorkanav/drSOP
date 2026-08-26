# drSOP — Gated Multimodal Diabetic Retinopathy Grading

Predicts ICDR diabetic retinopathy severity grade (0-4) from a single-eye fundus image + patient
metadata, using a RETFound image encoder, a small transformer metadata encoder, and a
transformer-based gate that fuses the two embeddings per-dimension:

    fused = alpha * image_emb + (1 - alpha) * meta_emb      (alpha in [0,1]^d, from GateTransformer)

Only uses patients with **complete metadata** (no imputation) — `scripts/prepare_data.py`
unconditionally drops any row missing `patient_age`/`diabetes_time_y`/`patient_sex`/`insuline`
before splitting. On the real BRSET data this is a large cut (most patients don't have
`diabetes_time_y`/`insuline` recorded), leaving a few thousand images. The split is patient-level
(via `GroupShuffleSplit` on `patient_id`), so both eyes of a patient always land in the same split.

(Earlier iteration of this repo split into two stages — DR present/absent, then severity — on the
assumption that severity grading was only available for a smaller subset. On the real data both
labels turned out to be fully populated, so that split bought nothing and was collapsed back into
this single 5-class model.)

## Setup (on the training VM)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # match your CUDA version
pip install -r requirements.txt

bash scripts/setup_retfound.sh   # clones third_party/RETFound_MAE
# then, manually (gated/credentialed download):
#   - RETFound_cfp_weights.pth -> weights/RETFound_cfp_weights.pth
```

`configs/default.yaml`'s `data.raw_labels_csv` / `data.images_dir` point directly at the shared
BRSET mount rather than a local copy — update those paths if your data lives somewhere else.

`timm==0.4.12` is pinned in requirements.txt because RETFound_MAE's `models_vit.py` targets that
older timm API — don't upgrade it without checking compatibility.

## Pipeline

```bash
python scripts/prepare_data.py --config configs/default.yaml   # one-time: split + metadata stats
python train.py --config configs/default.yaml                  # trains, checkpoints to runs/exp1/
```

`prepare_data.py` fits comorbidity vocab + numeric normalization stats on the train split only
(written to `data/processed/`), and asserts no patient appears in more than one split. It prints
per-field missingness, the final image/patient counts, and the train label distribution — check
that output for class imbalance before training.

## Repo layout

- `configs/default.yaml` — all hyperparameters.
- `scripts/prepare_data.py` — complete-metadata filter, patient-level split, metadata preprocessing.
- `src/drsop/data/` — `MetadataProcessor`, `BRSETDataset`.
- `src/drsop/models/` — `RetfoundEncoder` (frozen ViT-L), `MetadataEncoder`, `GateTransformer`,
  `MLPHead`, wired together in `fusion_model.DRFusionModel`.
- `train.py` — training loop, AMP, early stopping on validation quadratic weighted kappa (QWK).

## Notes / open items

- BRSET image file extension is assumed `.jpg`/`.jpeg`/`.png` (checked in that order).
- Comorbidity handling is data-driven: `prepare_data.py` tokenizes the free-text `comorbidities`
  field and keeps the top-K most frequent tokens from the training split as multi-hot flags
  (`configs/default.yaml: data.comorbidity_vocab_size`). Inspect the printed vocab after running
  `prepare_data.py` — if it's noisy, tighten the tokenizer in `src/drsop/data/text.py`.
- RETFound is frozen by default (`model.freeze_retfound: true`). If validation QWK plateaus,
  the next step is unfreezing the last few ViT blocks or adding LoRA adapters — not full
  fine-tuning first, given the dataset is only ~1-2k images after the complete-metadata filter.
- With only ~1k training images, watch for overfitting on the metadata/gate/head (the only
  trainable parts) — consider raising `head.dropout` or lowering `meta_encoder`/`gate` capacity
  if train/val metrics diverge early.
