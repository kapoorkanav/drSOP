"""
Controlled missing-metadata robustness sweep: hide metadata on purpose and measure the damage.

Every missingness result so far has been OBSERVATIONAL -- on BRSET, metadata was missing for
reasons correlated with disease, so a drop in accuracy could never be cleanly attributed to lost
information rather than to a lost shortcut. The balanced cohort removed that correlation, but
missingness was still something the data did to us, not something we controlled.

Here we control it. Metadata is hidden at a known rate, chosen at random, independent of the
label. Any degradation is therefore attributable to information loss and nothing else. Three
things come out of it that the observational runs could not give:

  1. A degradation CURVE across removal rates, instead of a single number.
  2. Real error bars, from several random masks per rate -- no retraining needed, so this is the
     cheapest confidence interval available anywhere in the project.
  3. A clean test of the gate's central claim. If alpha does not shift toward the image even at
     100% metadata removed, the gate is not adapting to missingness -- measured under conditions
     where there is no confound left to blame.

The 100%-removal row is the image-only ceiling. If accuracy barely moves from 0% to 100%, the
metadata branch was contributing little and the fusion story is weak -- worth knowing plainly.

    # BRSET, best architecture without the missingness token (experiment 3)
    python scripts/ablate_metadata.py --config configs/balanced_lora.yaml \
        --checkpoint runs/exp1_balanced_lora/best.pt

    # mBRSET, same architecture
    python scripts/ablate_metadata.py --config mbrset/configs/mbrset_lora.yaml \
        --checkpoint runs/mbrset_lora/best.pt

Speed: image embeddings do not depend on metadata, so they are computed ONCE and reused for
every rate and every mask. The sweep costs one pass over the images plus some very cheap
transformer calls -- not one full pass per cell.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402

META_KEYS = ["numeric", "numeric_missing", "categorical", "categorical_missing",
             "comorbidity", "comorbidity_missing"]


def cache_forward_inputs(model, loader, device):
    """One pass over the images. Returns the image embeddings plus the raw metadata tensors,
    so the sweep can re-run only the metadata/gate/head path."""
    image_emb, labels = [], []
    meta = {k: [] for k in META_KEYS}
    with torch.no_grad():
        for batch in tqdm(loader, desc="encoding images (once)", leave=False):
            labels.append(batch.pop("label"))
            image_emb.append(model.image_encoder(batch["image"].to(device)).cpu())
            for key in META_KEYS:
                meta[key].append(batch[key])
    return (torch.cat(image_emb), {k: torch.cat(v) for k, v in meta.items()},
            torch.cat(labels).numpy())


def build_mask(n: int, n_fields: int, rate: float, granularity: str, mode: str,
               labels: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """[n, n_fields] boolean: True = hide this field for this example.

    granularity 'patient' hides the whole metadata vector at once (the missing-MODALITY
    question); 'field' hides each field independently (partial charts).

    mode 'mcar' removes at random, independent of the label. mode 'mnar' removes only from
    DR-negative examples, reproducing BRSET's real pattern where missing metadata implied
    health -- so a model CAN cheat, and we can see whether it does.
    """
    if mode == "mnar":
        negative = labels == 0
        frac_neg = negative.mean()
        # Concentrate the same overall removal budget onto the negatives only.
        p = min(1.0, rate / frac_neg) if frac_neg > 0 else 0.0
        eligible = negative
    else:
        p = rate
        eligible = np.ones(len(labels), dtype=bool)

    if granularity == "patient":
        drop = (rng.random(n) < p) & eligible
        return torch.from_numpy(np.repeat(drop[:, None], n_fields, axis=1))
    drop = (rng.random((n, n_fields)) < p) & eligible[:, None]
    return torch.from_numpy(drop)


def apply_mask(meta: dict, mask: torch.Tensor, n_numeric: int, cat_missing_idx: list) -> dict:
    """Hides fields the way MetadataProcessor marks genuinely-absent ones, so the encoder swaps
    in its learned missing embeddings -- not a zero or an imputed mean."""
    out = {k: v.clone() for k, v in meta.items()}
    n_cat = len(cat_missing_idx)

    num_mask = mask[:, :n_numeric]
    out["numeric"][num_mask] = 0.0
    out["numeric_missing"][num_mask] = 1.0

    cat_mask = mask[:, n_numeric:n_numeric + n_cat]
    for i, missing_idx in enumerate(cat_missing_idx):
        col = cat_mask[:, i]
        out["categorical"][col, i] = missing_idx
        out["categorical_missing"][col, i] = 1.0

    com_mask = mask[:, -1]
    out["comorbidity"][com_mask] = 0.0
    out["comorbidity_missing"][com_mask] = 1.0
    return out


def run_head(model, image_emb, meta, device, batch_size=256):
    """Mirrors DRFusionModel.forward from the metadata branch onward."""
    preds, alphas = [], []
    with torch.no_grad():
        for i in range(0, len(image_emb), batch_size):
            sl = slice(i, i + batch_size)
            img = image_emb[sl].to(device)
            m = {k: v[sl].to(device) for k, v in meta.items()}
            meta_emb = model.meta_encoder(m["numeric"], m["numeric_missing"],
                                          m["categorical"], m["comorbidity"],
                                          m["comorbidity_missing"])
            flags = None
            if model.use_missingness_token:
                flags = torch.cat([m["numeric_missing"], m["categorical_missing"],
                                   m["comorbidity_missing"].unsqueeze(-1)], dim=-1)
            alpha = model.gate(img, meta_emb, flags)
            logits = model.head(alpha * img + (1 - alpha) * meta_emb)
            preds.append(logits.argmax(dim=-1).cpu())
            alphas.append(alpha.mean(dim=-1).cpu())
    return torch.cat(preds).numpy(), torch.cat(alphas).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--rates", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--n-masks", type=int, default=5,
                        help="random masks per rate; the spread across them is the error bar")
    parser.add_argument("--granularity", default="patient", choices=["patient", "field"],
                        help="'patient' hides the whole metadata vector; 'field' hides fields "
                             "independently")
    parser.add_argument("--mode", default="mcar", choices=["mcar", "mnar"],
                        help="'mcar' removes at random; 'mnar' removes only from DR-negative "
                             "examples, reproducing BRSET's real confound")
    parser.add_argument("--out", default=None, help="also write the report to this path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = MetadataProcessor(
        processed_dir=dcfg["processed_dir"], numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    ds = BRSETDataset(
        split_csv=str(Path(dcfg["processed_dir"]) / f"{args.split}.csv"),
        images_dir=dcfg["images_dir"], metadata=metadata, label_col=dcfg["label_col"],
        image_size=dcfg["image_size"], train=False, label_map=dcfg.get("label_map"),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    cardinalities = [metadata.num_categories(f) for f in dcfg["categorical_fields"]]
    model = DRFusionModel(
        retfound_cfg=mcfg, meta_cfg=mcfg["meta_encoder"], gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"], categorical_cardinalities=cardinalities,
        n_numeric=len(dcfg["numeric_fields"]), n_comorbidities=len(metadata.comorbidity_vocab),
        proj_dim=mcfg["proj_dim"], num_classes=mcfg["num_classes"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    lines = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, "
         f"best val QWK {ckpt.get('best_qwk', float('nan')):.4f})")
    emit(f"Config: {args.config}   split: {args.split}   classes: {mcfg['num_classes']}")
    emit(f"Granularity: {args.granularity}   mode: {args.mode.upper()}   "
         f"masks per rate: {args.n_masks}")

    image_emb, meta, labels = cache_forward_inputs(model, loader, device)
    n = len(labels)
    # one mask column per numeric field, per categorical field, plus one for comorbidities
    n_fields = len(dcfg["numeric_fields"]) + len(cardinalities) + 1
    cat_missing_idx = [c - 1 for c in cardinalities]  # reserved unseen/missing index per field
    emit(f"{n} examples, {n_fields} maskable metadata fields "
         f"({len(dcfg['numeric_fields'])} numeric, {len(cardinalities)} categorical, "
         f"1 comorbidity)")

    emit("")
    emit("=" * 86)
    emit("METADATA REMOVAL SWEEP   (mean +/- sd across masks)")
    emit("=" * 86)
    emit(f"{'removed':>9} {'actual':>8} {'accuracy':>16} {'QWK':>16} {'macro-F1':>16} "
         f"{'mean alpha':>11}")
    emit("-" * 86)

    results = []
    for rate in args.rates:
        # rate 0 and rate 1 are deterministic -- no spread to measure, so one mask each.
        n_masks = 1 if rate in (0.0, 1.0) and args.mode == "mcar" else args.n_masks
        accs, qwks, f1s, alphas, actuals = [], [], [], [], []
        for seed in range(n_masks):
            rng = np.random.default_rng(1000 + seed)
            mask = build_mask(n, n_fields, rate, args.granularity, args.mode, labels, rng)
            actuals.append(mask.float().mean().item())
            ablated = apply_mask(meta, mask, len(dcfg["numeric_fields"]), cat_missing_idx)
            preds, alpha = run_head(model, image_emb, ablated, device)
            m = compute_metrics(labels, preds)
            accs.append(m["accuracy"]); qwks.append(m["qwk"]); f1s.append(m["macro_f1"])
            alphas.append(float(alpha.mean()))

        def ms(vals):
            return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}" if len(vals) > 1 \
                else f"{np.mean(vals):.4f}        "

        emit(f"{rate * 100:>8.0f}% {np.mean(actuals) * 100:>7.1f}% {ms(accs):>16} "
             f"{ms(qwks):>16} {ms(f1s):>16} {np.mean(alphas):>11.4f}")
        results.append({"rate": rate, "acc": np.mean(accs), "qwk": np.mean(qwks),
                        "alpha": np.mean(alphas)})

    emit("")
    emit("'actual' is the realised fraction of hidden field-slots, which differs from the")
    emit("requested rate under MNAR (removal is concentrated on DR-negative examples).")
    emit("alpha: 1.0 = read the image, 0.0 = read the metadata.")

    base = next((r for r in results if r["rate"] == 0.0), None)
    full = next((r for r in results if r["rate"] == 1.0), None)
    if base and full:
        emit("")
        emit("=" * 86)
        emit("READING")
        emit("=" * 86)
        d_acc = (full["acc"] - base["acc"]) * 100
        d_qwk = full["qwk"] - base["qwk"]
        d_alpha = full["alpha"] - base["alpha"]
        emit(f"  All metadata hidden vs none: accuracy {d_acc:+.2f} points, QWK {d_qwk:+.4f}")
        emit(f"  Alpha shift over the same range: {d_alpha:+.4f} "
             f"({base['alpha']:.4f} -> {full['alpha']:.4f})")
        emit("")
        if abs(d_acc) < 1.0:
            emit("  => Metadata contributes under 1 accuracy point. The image branch is doing")
            emit("     essentially all the work, and the fusion architecture is not earning its")
            emit("     complexity on this dataset. Report this plainly.")
        else:
            emit(f"  => Metadata is worth {abs(d_acc):.1f} accuracy points. The fusion has real")
            emit("     content, and the curve above says how gracefully it degrades.")
        emit("")
        if abs(d_alpha) < 0.02:
            emit("  => The gate does NOT re-route when metadata disappears (alpha moves less than")
            emit("     0.02 from fully-present to fully-absent). Under controlled removal there is")
            emit("     no confound left to explain this away: the gate is not adaptive to")
            emit("     missingness. This is the clean version of the BRSET experiment-4 result.")
        else:
            emit(f"  => The gate DOES re-route, shifting alpha {d_alpha:+.4f} toward the")
            emit("     " + ("image" if d_alpha > 0 else "metadata") + " as metadata is removed. "
                 "This is the adaptive behaviour BRSET could not show.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
