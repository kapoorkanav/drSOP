"""
Characterises what the fusion gate actually learned, beyond the dataset-wide mean alpha
(~0.52 in every run so far) that hides the interesting structure.

Answers two questions the mean can't:
  1. Is alpha bimodal (near-binary routing -- some dims taken from the image, others from
     the metadata) or genuinely blending (~0.5 everywhere)? A mean of 0.52 is consistent
     with BOTH, and they are very different claims about the architecture.
  2. Is that routing the same for every patient (a static feature-space partition) or
     adapted per patient (a dynamic fusion policy)?

    python scripts/inspect_gate_alpha.py --config configs/balanced_lora.yaml \
        --checkpoint runs/exp1_balanced_lora/best.pt

Prints ASCII histograms (no dependencies beyond what's installed). Also writes a PNG
alongside the checkpoint if matplotlib happens to be available -- optional, skipped silently
otherwise.
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
from drsop.models.fusion_model import DRFusionModel  # noqa: E402


def ascii_histogram(values: np.ndarray, n_bins: int = 20, width: int = 50, title: str = "") -> None:
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0))
    peak = counts.max() if counts.max() else 1
    if title:
        print(title)
    for i in range(n_bins):
        bar = "#" * int(round(width * counts[i] / peak))
        pct = 100 * counts[i] / len(values)
        print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}) {bar:<{width}} {counts[i]:>9,} ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = dcfg["processed_dir"]

    metadata = MetadataProcessor(
        processed_dir=processed_dir, numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"], comorbidity_field=dcfg["comorbidity_field"],
    )
    ds = BRSETDataset(
        split_csv=str(Path(processed_dir) / f"{args.split}.csv"), images_dir=dcfg["images_dir"],
        metadata=metadata, label_col=dcfg["label_col"], image_size=dcfg["image_size"], train=False,
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    model = DRFusionModel(
        retfound_cfg=mcfg, meta_cfg=mcfg["meta_encoder"], gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"],
        categorical_cardinalities=[metadata.num_categories(f) for f in dcfg["categorical_fields"]],
        n_numeric=len(dcfg["numeric_fields"]), n_comorbidities=len(metadata.comorbidity_vocab),
        proj_dim=mcfg["proj_dim"], num_classes=mcfg["num_classes"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    alphas = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"collecting alpha ({args.split})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            _, alpha = model(batch, return_alpha=True)
            alphas.append(alpha.cpu().numpy())
    alpha = np.concatenate(alphas, axis=0)  # [n_examples, proj_dim]
    n_examples, n_dims = alpha.shape
    flat = alpha.ravel()

    print(f"\nCollected alpha for {n_examples:,} examples x {n_dims} dims "
          f"= {flat.size:,} gate values\n")

    print("=" * 78)
    print("1. ALL GATE VALUES -- is the gate blending, or routing near-binary?")
    print("=" * 78)
    ascii_histogram(flat, title="")
    near_meta = (flat < 0.1).mean() * 100
    near_image = (flat > 0.9).mean() * 100
    blending = ((flat >= 0.4) & (flat <= 0.6)).mean() * 100
    print(f"\n  mean {flat.mean():.4f} | median {np.median(flat):.4f} | sd {flat.std():.4f}")
    print(f"  alpha < 0.1  (essentially pure METADATA): {near_meta:5.1f}%")
    print(f"  alpha > 0.9  (essentially pure IMAGE):    {near_image:5.1f}%")
    print(f"  0.4-0.6      (genuinely blending):        {blending:5.1f}%")
    print(f"  => saturated at one extreme or the other: {near_meta + near_image:5.1f}%")
    print("\n  Reference points for sd of a [0,1] quantity: uniform = 0.289,")
    print("  all-mass-at-0-and-1 (perfectly binary) = ~0.500.")

    print("\n" + "=" * 78)
    print("2. PER-DIMENSION ROUTING -- averaged over examples, where does each dim sit?")
    print("=" * 78)
    per_dim_mean = alpha.mean(axis=0)  # [proj_dim]
    ascii_histogram(per_dim_mean, title="")
    print(f"\n  dims with mean alpha < 0.1 (this dim always reads METADATA): "
          f"{(per_dim_mean < 0.1).sum():>4} of {n_dims}")
    print(f"  dims with mean alpha > 0.9 (this dim always reads IMAGE):    "
          f"{(per_dim_mean > 0.9).sum():>4} of {n_dims}")
    print(f"  dims in 0.4-0.6 (this dim genuinely mixes):                  "
          f"{((per_dim_mean >= 0.4) & (per_dim_mean <= 0.6)).sum():>4} of {n_dims}")

    print("\n" + "=" * 78)
    print("3. IS THE ROUTING STATIC ACROSS PATIENTS, OR PATIENT-SPECIFIC?")
    print("=" * 78)
    per_dim_std = alpha.std(axis=0)      # variation of each dim across patients
    per_example_std = alpha.std(axis=1)  # variation across dims within a patient
    print(f"  Mean spread of a given dim ACROSS patients:   {per_dim_std.mean():.4f}")
    print(f"  Mean spread ACROSS dims within one patient:   {per_example_std.mean():.4f}")
    ratio = per_example_std.mean() / per_dim_std.mean() if per_dim_std.mean() else float("inf")
    print(f"  ratio (within-patient / across-patient):      {ratio:.1f}x")
    print("\n  A large ratio means the gate mostly learned a FIXED partition of the feature")
    print("  space (same dims read the image for everyone) rather than deciding per patient.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].hist(flat, bins=50, range=(0, 1), color="#d1602a")
        axes[0].set_title(f"All gate values ({flat.size:,})")
        axes[0].set_xlabel("alpha  (0 = metadata, 1 = image)")
        axes[1].hist(per_dim_mean, bins=50, range=(0, 1), color="#0f9c68")
        axes[1].set_title(f"Per-dimension mean ({n_dims} dims)")
        axes[1].set_xlabel("mean alpha for that dimension")
        for ax in axes:
            ax.set_ylabel("count")
        fig.tight_layout()
        out_png = Path(args.checkpoint).parent / f"gate_alpha_{args.split}.png"
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved figure to {out_png}")
    except ImportError:
        print("\n(matplotlib not installed -- skipped the PNG; the histograms above are the "
              "same data. `pip install matplotlib` if you want the figure for a writeup.)")


if __name__ == "__main__":
    main()
