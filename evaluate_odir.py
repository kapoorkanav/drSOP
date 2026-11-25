"""
Evaluate an ODIR checkpoint on the held-out test set, per disease.

    python evaluate_odir.py --config configs/odir_lora.yaml --checkpoint runs/odir_lora/best.pt

Reports AUC (threshold-free) plus F1 / sensitivity / specificity at two cut-offs:
  - 0.5, the default
  - a per-disease threshold chosen to maximise F1 on the VALIDATION set, then applied unchanged
    to test. For a disease in ~4% of eyes, 0.5 is often badly placed; tuning on val (never on
    test) gives a fair read of what the model can do once the cut-off is set sensibly.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.odir_dataset import ODIRDataset  # noqa: E402
from drsop.metrics import compute_multilabel_metrics  # noqa: E402
from train_odir import build_odir_model, load_codes  # noqa: E402


def predict(model, ds, device, num_workers: int):
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=num_workers)
    probs, trues = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predicting", leave=False):
            labels = batch.pop("label")
            batch = {k: v.to(device) for k, v in batch.items()}
            probs.append(torch.sigmoid(model(batch).float()).cpu().numpy())
            trues.append(labels.numpy())
    return np.concatenate(trues), np.concatenate(probs)


def best_f1_threshold(t: np.ndarray, p: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(t, (p >= c).astype(int), zero_division=0) for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def report(y_true, y_prob, codes, thresholds, title):
    print(f"\n{title}")
    print(f"  {'disease':>8} {'n_pos':>6} {'AUC':>7} {'thresh':>7} {'F1':>6} {'sens':>6} {'spec':>6}"
          f"   TP   FN   FP    TN")
    for i, c in enumerate(codes):
        t, s = y_true[:, i], y_prob[:, i]
        m = compute_multilabel_metrics(y_true[:, [i]], y_prob[:, [i]], [c], threshold=thresholds[i])
        p = (s >= thresholds[i]).astype(int)
        tp = int(((p == 1) & (t == 1)).sum()); fn = int(((p == 0) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum()); tn = int(((p == 0) & (t == 0)).sum())
        print(f"  {c:>8} {int(t.sum()):>6} {m[f'{c}_auc']:>7.4f} {thresholds[i]:>7.2f} "
              f"{m[f'{c}_f1']:>6.3f} {m[f'{c}_sens']:>6.3f} {m[f'{c}_spec']:>6.3f}"
              f" {tp:>4} {fn:>4} {fp:>4} {tn:>5}")
    m_all = compute_multilabel_metrics(y_true, y_prob, codes)
    print(f"  mean AUC {m_all['mean_auc']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg = cfg["data"], cfg["model"]
    processed_dir = dcfg["processed_dir"]
    codes = load_codes(processed_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_odir_model(mcfg, len(codes)).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)  # frozen backbone isn't in the ckpt
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(best val mean AUC at save time: {ckpt['best_mean_auc']:.4f})")

    def ds(split):
        return ODIRDataset(str(Path(processed_dir) / f"{split}.csv"), dcfg["images_dir"],
                           codes, dcfg["image_size"], train=False)

    val_true, val_prob = predict(model, ds("val"), device, cfg["train"]["num_workers"])
    tuned = [best_f1_threshold(val_true[:, i], val_prob[:, i]) for i in range(len(codes))]

    y_true, y_prob = predict(model, ds(args.split), device, cfg["train"]["num_workers"])
    print(f"\n{args.split} set: {len(y_true)} eye-images")
    report(y_true, y_prob, codes, [0.5] * len(codes), "At threshold 0.5:")
    report(y_true, y_prob, codes, tuned,
           "At per-disease thresholds tuned for F1 on VAL (applied unchanged here):")


if __name__ == "__main__":
    main()
