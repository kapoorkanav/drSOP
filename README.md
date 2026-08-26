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
python train.py --config configs/default.yaml                  # frozen RETFound baseline -> runs/exp1/
python evaluate.py --config configs/default.yaml --checkpoint runs/exp1/best.pt   # honest test-set numbers
```

`prepare_data.py` fits comorbidity vocab + numeric normalization stats on the train split only
(written to `data/processed/`), and asserts no patient appears in more than one split. It prints
per-field missingness, the final image/patient counts, and the train label distribution — check
that output for class imbalance before training.

**LoRA fine-tuning variant**: `configs/lora.yaml` reuses the same processed data/stats but sets
`model.use_lora: true` — RETFound's base weights stay frozen, but LoRA adapters get injected into
every attention/MLP linear (`blocks.*.attn.{qkv,proj}`, `blocks.*.mlp.{fc1,fc2}`) plus `fc_norm`
becomes fully trainable (it was never pretrained regardless of fine-tuning mode, see the note
below). Same architecture code path (`RetfoundEncoder(use_lora=...)`), separate config and
`output_dir: runs/exp1_lora/` so it never overwrites the frozen-baseline checkpoints:

```bash
python train.py --config configs/lora.yaml
python evaluate.py --config configs/lora.yaml --checkpoint runs/exp1_lora/best.pt
```

## Repo layout

- `configs/default.yaml` — frozen-RETFound baseline hyperparameters. `configs/lora.yaml` — same,
  with LoRA enabled and its own `output_dir`.
- `scripts/prepare_data.py` — complete-metadata filter, patient-level split, metadata preprocessing.
- `src/drsop/data/` — `MetadataProcessor`, `BRSETDataset`.
- `src/drsop/models/` — `RetfoundEncoder` (frozen ViT-L, or LoRA-adapted via `use_lora=True`),
  `MetadataEncoder`, `GateTransformer`, `MLPHead`, wired together in `fusion_model.DRFusionModel`.
- `train.py` — training loop, AMP, early stopping on validation quadratic weighted kappa (QWK).
- `evaluate.py` — loads a checkpoint, reports QWK/accuracy/macro-F1 + confusion matrix + mean
  learned `alpha` on any split (defaults to test).

## Results so far

Both runs use the same split: 1,551 images (809 patients) after the complete-metadata filter --
1,078 train / 237 val / 236 test images (566/121/122 patients), patient-level split.

| | Frozen RETFound (`configs/default.yaml`) | LoRA fine-tuned (`configs/lora.yaml`) |
|---|---|---|
| Best val QWK | 0.8354 (epoch 33) | 0.9128 (epoch 13) |
| Test QWK | 0.7043 | **0.8351** |
| Test accuracy | 0.58 | 0.74 |
| Test macro-F1 | 0.44 | 0.56 |
| Mean learned alpha | 0.55 | 0.53 |

LoRA (r=8, adapters on every attention/MLP linear + fully-trainable `fc_norm`, base weights frozen,
~1% of backbone params trainable) meaningfully beat the frozen baseline on the held-out test set,
not just validation -- per-grade accuracy improved from 61%->77% (grade 0), 44%->68% (grade 2), and
70%->93% (grade 4). Grade 3 stayed weak in both (only 19 training examples) -- adapting image
features doesn't fix a class that's fundamentally data-starved, so that's the next thing to address
(oversampling/augmentation) rather than more fine-tuning. Mean `alpha` stayed ~0.5 in both runs --
the gate consistently blends both modalities rather than collapsing to one, which was the core
hypothesis this architecture was built to test, and it held even as the image branch got better.

## Notes / open items

- BRSET image file extension is assumed `.jpg`/`.jpeg`/`.png` (checked in that order).
- Comorbidity handling is data-driven: `prepare_data.py` tokenizes the free-text `comorbidities`
  field and keeps the top-K most frequent tokens from the training split as multi-hot flags
  (`configs/default.yaml: data.comorbidity_vocab_size`). Inspect the printed vocab after running
  `prepare_data.py` — if it's noisy, tighten the tokenizer in `src/drsop/data/text.py`.
- With only ~1k training images, watch for overfitting on the metadata/gate/head (the only
  trainable parts in the frozen-baseline config) — consider raising `head.dropout` or lowering
  `meta_encoder`/`gate` capacity if train/val metrics diverge early.
- The middle severity grades (1-3) are the weakest part of the frozen baseline, purely from having
  few training examples (54/108/19 respectively) — LoRA fine-tuning may or may not help this
  specifically (it adapts image features, not class balance), so also worth considering targeted
  augmentation or oversampling for those grades if LoRA alone doesn't move the needle.
