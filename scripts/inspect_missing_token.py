"""
Did the gate's missingness token actually learn anything, or did it stay a no-op?

After training with model.gate.use_missingness_token, the gate came out perfectly
indifferent to metadata completeness (alpha identical to 4 dp between full and partial
groups). Two very different explanations:

  a) the token never escaped its small initialisation (trunc_normal std=0.02), so it
     contributes the same constant vector for every patient regardless of what's missing;
  b) it learned real weights and deliberately settled on missingness-invariant routing.

This reads the checkpoint directly -- no GPU, no data, no RETFound weights needed -- and
compares missing_proj's learned magnitude against both its init scale and the gate's other
parameters (modality_emb started at the same std=0.02, so it's the fair yardstick).

Also prints the per-field column norms, which are the interpretable payoff of choosing a
linear projection: each column is the direction that field's absence pushes the gate.

    python scripts/inspect_missing_token.py --config configs/balanced_missgate_lora.yaml \
        --checkpoint runs/exp1_balanced_missgate_lora/best.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drsop.config import load_config, resolve  # noqa: E402

INIT_STD = 0.02  # what gate.py initialises missing_proj.weight and modality_emb to


def describe(name: str, tensor: torch.Tensor) -> None:
    print(f"  {name:<34} std={tensor.std().item():.5f}  mean|w|={tensor.abs().mean().item():.5f}  "
          f"max|w|={tensor.abs().max().item():.5f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg = resolve(load_config(args.config), root)
    dcfg = cfg["data"]
    field_names = list(dcfg["numeric_fields"]) + list(dcfg["categorical_fields"]) + ["comorbidities"]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd = ckpt["model"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Saved at epoch {ckpt['epoch']} (best val QWK {ckpt['best_qwk']:.4f})\n")

    key = "gate.missing_proj.weight"
    if key not in sd:
        print(f"No {key} in this checkpoint -- it was trained without the missingness token "
              "(model.gate.use_missingness_token was false or absent).")
        return

    w = sd[key]           # [dim, n_flags]
    b = sd["gate.missing_proj.bias"]
    print("=" * 78)
    print("DID THE MISSINGNESS TOKEN LEARN? (all three started at std=0.02)")
    print("=" * 78)
    describe("missing_proj.weight", w)
    describe("missing_proj.bias", b)
    if "gate.modality_emb" in sd:
        describe("modality_emb  (same init, yardstick)", sd["gate.modality_emb"])
    if "gate.alpha_head.weight" in sd:
        describe("alpha_head.weight  (default init)", sd["gate.alpha_head.weight"])

    growth = w.std().item() / INIT_STD
    print(f"\n  missing_proj.weight std is {growth:.2f}x its initialisation ({INIT_STD}).")
    if growth < 1.5:
        print("  => Essentially UNMOVED. The token stayed a no-op: it contributes roughly the")
        print("     same constant vector for every patient, which explains alpha being identical")
        print("     across completeness groups. The signal was available and training had no")
        print("     reason to use it -- evidence the bottleneck is training pressure (-> modality")
        print("     dropout), not information availability.")
    else:
        print("  => It MOVED substantially. The gate had real, trained weights available and")
        print("     still settled on missingness-invariant routing -- a deliberate choice rather")
        print("     than an untrained pathway. That argues the model genuinely doesn't find")
        print("     missingness useful for this objective.")

    print("\n" + "=" * 78)
    print("PER-FIELD DIRECTIONS -- how strongly does each field's absence push the gate?")
    print("=" * 78)
    print("  (column norm of missing_proj.weight; larger = that field's absence matters more)\n")
    col_norms = w.norm(dim=0)  # one per flag
    order = torch.argsort(col_norms, descending=True)
    peak = col_norms.max().item() or 1.0
    for i in order.tolist():
        label = field_names[i] if i < len(field_names) else f"flag_{i}"
        bar = "#" * int(round(34 * col_norms[i].item() / peak))
        print(f"  {label:<20} {bar:<34} {col_norms[i].item():.4f}")


if __name__ == "__main__":
    main()
