"""
Is the gate's severity-tracking behaviour actually learned, or an artifact of the embeddings?

inspect_missing_token.py found the gate sits at ~1.0x its initialisation even after 11 epochs.
That matters because fused = alpha*image_emb + (1-alpha)*meta_emb, so a gate that never moved
is a fixed random projection -- and a fixed random readout of severity-informative embeddings
will still produce a severity-correlated alpha, with nothing having been routed on purpose.

The test: keep the trained image encoder, metadata encoder and head exactly as they are, throw
away the trained gate, drop in a freshly random one, and recompute alpha. Run it for several
random gates so the result isn't one lucky draw.

  If the random gates reproduce the trained gate's severity correlation and saturation, the
  behaviour lives in (image_emb - meta_emb), not in anything the gate learned.
  If they don't, the trained gate is doing real work despite its weights barely moving.

    python scripts/gate_ablation.py --config configs/balanced_lora.yaml \
        --checkpoint runs/exp1_balanced_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.labels import apply_label_map  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402
from drsop.models.gate import GateTransformer  # noqa: E402


def collect_alpha(model, loader, device) -> np.ndarray:
    alphas = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("label")
            _, alpha = model(batch, return_alpha=True)
            alphas.append(alpha.cpu().numpy())
    return np.concatenate(alphas, axis=0)


def summarise(alpha: np.ndarray, grades: np.ndarray, label: str) -> dict:
    per_example = alpha.mean(axis=1)
    flat = alpha.ravel()
    rho, p = spearmanr(grades, per_example)
    saturated = ((flat < 0.1) | (flat > 0.9)).mean() * 100
    blending = ((flat >= 0.4) & (flat <= 0.6)).mean() * 100
    print(f"  {label:<26} rho={rho:+.3f} (p={p:.2g})   mean={flat.mean():.4f}  "
          f"sd={flat.std():.4f}  saturated={saturated:5.1f}%  blending={blending:5.1f}%")
    return {"rho": rho, "saturated": saturated, "sd": flat.std()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-random-gates", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    label_col = dcfg["label_col"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = dcfg["processed_dir"]

    split_csv = Path(processed_dir) / f"{args.split}.csv"
    # Mapped the same way BRSETDataset maps it, so these line up with the model's classes.
    grades = apply_label_map(pd.read_csv(split_csv)[label_col], dcfg.get("label_map")).values

    metadata = MetadataProcessor(
        processed_dir=processed_dir, numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"], comorbidity_field=dcfg["comorbidity_field"],
    )
    ds = BRSETDataset(
        split_csv=str(split_csv), images_dir=dcfg["images_dir"], metadata=metadata,
        label_col=label_col, image_size=dcfg["image_size"], train=False, label_map=dcfg.get("label_map"),
        )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    categorical_cardinalities = [metadata.num_categories(f) for f in dcfg["categorical_fields"]]
    model = DRFusionModel(
        retfound_cfg=mcfg, meta_cfg=mcfg["meta_encoder"], gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"], categorical_cardinalities=categorical_cardinalities,
        n_numeric=len(dcfg["numeric_fields"]), n_comorbidities=len(metadata.comorbidity_vocab),
        proj_dim=mcfg["proj_dim"], num_classes=mcfg["num_classes"],
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"\nLoaded {args.checkpoint} (epoch {ckpt['epoch']}, best val QWK {ckpt['best_qwk']:.4f})")

    print("\n" + "=" * 94)
    print("GATE ABLATION -- everything else stays trained; only the gate is replaced")
    print("=" * 94 + "\n")

    trained = summarise(collect_alpha(model, tqdm(loader, desc="trained gate", leave=False),
                                       device), grades, "TRAINED gate")

    # Rebuild a gate the same way DRFusionModel does, but freshly random each time.
    gate_cfg = dict(mcfg["gate"])
    use_tok = gate_cfg.pop("use_missingness_token", False)
    n_flags = (len(dcfg["numeric_fields"]) + len(categorical_cardinalities) + 1) if use_tok else 0

    randoms = []
    for seed in range(args.n_random_gates):
        torch.manual_seed(1000 + seed)
        model.gate = GateTransformer(dim=mcfg["proj_dim"], n_missing_flags=n_flags,
                                      **gate_cfg).to(device).eval()
        randoms.append(summarise(
            collect_alpha(model, tqdm(loader, desc=f"random gate {seed}", leave=False), device),
            grades, f"RANDOM gate (seed {seed})"))

    print("\n" + "=" * 94)
    mean_random_rho = float(np.mean([r["rho"] for r in randoms]))
    mean_random_sat = float(np.mean([r["saturated"] for r in randoms]))
    print(f"  trained rho {trained['rho']:+.3f}   vs   random-gate mean rho {mean_random_rho:+.3f}")
    print(f"  trained saturation {trained['saturated']:.1f}%   vs   random-gate mean "
          f"{mean_random_sat:.1f}%\n")

    retained = abs(mean_random_rho) / abs(trained["rho"]) if trained["rho"] else float("nan")
    if retained > 0.7:
        print(f"  => Random gates reproduce {retained:.0%} of the trained gate's severity")
        print("     correlation. The behaviour is a property of (image_emb - meta_emb), not of")
        print("     learned routing -- the encoders carry the severity signal and ANY readout of")
        print("     them inherits it. 'The gate learned severity-driven routing' does not hold.")
    elif retained > 0.3:
        print(f"  => Random gates reproduce {retained:.0%} of it. Part artifact, part learned --")
        print("     the embeddings supply much of the structure, but the trained gate adds to it.")
    else:
        print(f"  => Random gates reproduce only {retained:.0%} of it. The trained gate is doing")
        print("     real work despite its weights barely moving off their initialisation.")


if __name__ == "__main__":
    main()
