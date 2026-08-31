# drSOP — Gated Multimodal Diabetic Retinopathy Grading

Predicts ICDR diabetic retinopathy severity grade (0-4) from a single-eye fundus image + patient
metadata, using a RETFound image encoder, a small transformer metadata encoder, and a
transformer-based gate that fuses the two embeddings per-dimension:

    fused = alpha * image_emb + (1 - alpha) * meta_emb      (alpha in [0,1]^d, from GateTransformer)

Uses **all labeled BRSET rows**, including patients with partial or no metadata — the goal is
specifically to learn how to handle missing modalities gracefully, not to filter them out. The
split is patient-level and stratified on (worst-eye severity grade × metadata-completeness tier),
so train/val/test each get a fair, proportional mix of both (`scripts/prepare_data.py`). Earlier
iterations of this repo instead dropped any row with incomplete metadata; that approach is
preserved separately (see "Reproducing the earlier complete-metadata-only experiments" below) for
comparison, since it's what the current best (LoRA) results were measured on.

Missing values get an explicit, learned "missing" embedding rather than an imputed/faked value:
numeric fields (age, diabetes duration) and comorbidities each have a dedicated learned "unknown"
vector that replaces the token entirely when absent; categorical fields (sex, insulin use) get
this for free via their reserved unseen/missing embedding-table index. The model can tell "we
don't know this patient's age" apart from "this patient is average age."

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

`prepare_data.py` uses **every labeled row**, not just complete-metadata ones (metadata gaps are
meant to be handled by the model, not filtered out) — split via a **stratified group split**:
patient-level grouping (no leakage) combined with stratification on a (worst-eye severity grade ×
metadata-completeness tier) key, so train/val/test each get a fair, proportional mix of both.
Comorbidity vocab + numeric normalization stats are fit on the train split only. It prints
per-field missingness, metadata-completeness tiers, and per-split label/completeness distributions
— check that output before training.

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

**Reproducing the earlier complete-metadata-only experiments** (frozen test QWK 0.7043, LoRA
test QWK 0.8351 — see "Results so far" below): those used a different data prep step (drops any
row missing metadata, plain grouped-not-stratified split), preserved as
`scripts/prepare_data_complete_metadata_only.py`, paired with `configs/legacy_frozen.yaml` /
`configs/legacy_lora.yaml` (separate `processed_dir` and `output_dir`, so these can never collide
with the current all-data experiments):

```bash
python scripts/prepare_data_complete_metadata_only.py --config configs/legacy_frozen.yaml
python train.py --config configs/legacy_frozen.yaml
python train.py --config configs/legacy_lora.yaml
```

## Repo layout

- `configs/default.yaml` — frozen-RETFound baseline hyperparameters. `configs/lora.yaml` — same,
  with LoRA enabled and its own `output_dir`. `configs/legacy_frozen.yaml` / `configs/legacy_lora.yaml`
  — preserved complete-metadata-only variants, own `processed_dir`/`output_dir`.
- `scripts/prepare_data.py` — all-data, stratified (grade × completeness) patient-level split.
  `scripts/prepare_data_complete_metadata_only.py` — preserved earlier version (drops incomplete rows).
- `scripts/check_missingness_bias.py` — tests whether metadata missingness is informative about
  the label (it is, but the confound is mostly explained by diabetic-vs-not status — see
  `--diabetic-only`), to catch a model learning a missingness shortcut before it happens.
- `scripts/summarize_data.py` — broad raw-data readout: label coverage, metadata completeness
  tiers, comorbidity coverage, image quality/camera distributions.
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
