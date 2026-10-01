"""
Audits the experiments for the mistakes that do not announce themselves.

Every number in the write-ups depends on things no training log would have complained about:
that no patient leaked across splits, that the normalisation statistics were fit on train only,
that the checkpoint actually populated every trainable weight, and that the reported metrics
can be reproduced from the predictions. This checks each of those directly.

The one worth the most attention is the checkpoint check. train.py saves only trainable
parameters, and both evaluate.py and the analysis scripts load with strict=False -- which is
correct, since the frozen backbone is restored from the RETFound checkpoint on construction. But
strict=False also means a trainable weight MISSING from the checkpoint stays at its random
initialisation and is reported as a result anyway. That failure is invisible in the output.

    # one experiment
    python scripts/verify_experiments.py --config configs/balanced_3class_lora.yaml \
        --checkpoint runs/exp1_balanced_3class_lora/best.pt

    # data-only checks, no checkpoint needed
    python scripts/verify_experiments.py --config configs/balanced_lora.yaml

Exits non-zero if any check FAILS, so it can gate a write-up.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.labels import apply_label_map  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.data.text import parse_locale_number, tokenize_comorbidities  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.factory import build_model  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
    RESULTS.append((name, ok))
    return ok


def info(text: str) -> None:
    print(f"         {text}")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def check_splits(dcfg: dict, label_col: str) -> dict:
    section("1. SPLIT INTEGRITY")
    processed = Path(dcfg["processed_dir"])
    splits = {}
    for name in ("train", "val", "test"):
        path = processed / f"{name}.csv"
        if not path.exists():
            check(f"{name}.csv exists", False, str(path))
            return {}
        splits[name] = pd.read_csv(path)

    # A patient in two splits means their other eye's image trains the model that scores them.
    pats = {k: set(v["patient_id"]) for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = pats[a] & pats[b]
        check(f"no patient shared between {a} and {b}", not shared,
              f"{len(shared)} shared" if shared else "")

    imgs = {k: set(v["image_id"]) for k, v in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = imgs[a] & imgs[b]
        check(f"no image shared between {a} and {b}", not shared,
              f"{len(shared)} shared" if shared else "")

    dup = sum(v["image_id"].duplicated().sum() for v in splits.values())
    check("no duplicated image_id within a split", dup == 0, f"{dup} duplicates")

    for name, sdf in splits.items():
        n_null = sdf[label_col].isna().sum()
        check(f"{name}: no null labels", n_null == 0, f"{n_null} null")

    # Both eyes of a patient must carry the same completeness tier, or the stratification key
    # built from "first" tier per patient was lying.
    if "completeness_tier" in splits["train"].columns:
        allrows = pd.concat(splits.values())
        inconsistent = allrows.groupby("patient_id")["completeness_tier"].nunique()
        n_bad = int((inconsistent > 1).sum())
        check("completeness_tier consistent within each patient", n_bad == 0,
              f"{n_bad} patients with mixed tiers")

    total = sum(len(v) for v in splits.values())
    info(f"images: train={len(splits['train'])} val={len(splits['val'])} "
         f"test={len(splits['test'])} total={total}")
    info(f"patients: train={len(pats['train'])} val={len(pats['val'])} test={len(pats['test'])}")
    info(f"test.csv sha256: {hashlib.sha256((processed / 'test.csv').read_bytes()).hexdigest()[:16]}"
         "   (same digest across configs => genuinely the same test set)")
    return splits


def check_no_leakage_in_stats(dcfg: dict, splits: dict) -> None:
    section("2. PREPROCESSING FIT ON TRAIN ONLY")
    processed = Path(dcfg["processed_dir"])
    train = splits["train"]

    with open(processed / "metadata_stats.json") as f:
        stats = json.load(f)
    for field in dcfg["numeric_fields"]:
        vals = train[field].apply(parse_locale_number)
        exp_mean, exp_std = float(vals.mean()), float(vals.std() or 1.0)
        got = stats["numeric"][field]
        ok = (abs(got["mean"] - exp_mean) < 1e-6) and (abs(got["std"] - exp_std) < 1e-6)
        check(f"{field}: mean/std match the TRAIN split", ok,
              f"stored ({got['mean']:.4f}, {got['std']:.4f}) vs train "
              f"({exp_mean:.4f}, {exp_std:.4f})")

    for field in dcfg["categorical_fields"]:
        expected = sorted(train[field].dropna().astype(str).unique().tolist())
        got = stats["categorical"][field]
        check(f"{field}: category vocab is the TRAIN vocab", got == expected,
              f"stored {len(got)} vs train {len(expected)}")
        # A category only ever seen in val/test must fall to the reserved index, not crash or
        # silently alias onto a real category.
        unseen = set()
        for name in ("val", "test"):
            unseen |= set(splits[name][field].dropna().astype(str).unique()) - set(expected)
        if unseen:
            info(f"  {field}: {len(unseen)} category/ies appear only outside train "
                 f"({sorted(unseen)[:5]}) -> reserved missing index, as designed")

    with open(processed / "comorbidity_vocab.json") as f:
        vocab = json.load(f)
    counter = Counter()
    for text in train[dcfg["comorbidity_field"]].fillna(""):
        counter.update(tokenize_comorbidities(text))
    expected = [tok for tok, _ in counter.most_common(dcfg["comorbidity_vocab_size"])]
    check("comorbidity vocab is the TRAIN vocab", vocab == expected,
          f"stored {len(vocab)} tokens")


def check_label_map(dcfg: dict, mcfg: dict, splits: dict) -> None:
    section("3. LABELS AND CLASS COUNT")
    label_col, label_map = dcfg["label_col"], dcfg.get("label_map")
    n_classes = mcfg["num_classes"]
    info(f"label_map: {label_map}")
    for name, sdf in splits.items():
        mapped = apply_label_map(sdf[label_col], label_map)
        lo, hi = int(mapped.min()), int(mapped.max())
        ok = lo >= 0 and hi < n_classes
        check(f"{name}: mapped labels inside [0, {n_classes - 1}]", ok, f"range [{lo}, {hi}]")
        if name == "test":
            counts = mapped.value_counts().sort_index()
            info(f"test class counts: {counts.to_dict()}")
            info(f"majority-class baseline accuracy: {counts.max() / counts.sum():.4f}  "
                 f"(balanced-accuracy baseline {1 / n_classes:.4f})")
            missing = [c for c in range(n_classes) if c not in counts.index]
            check("every class present in test", not missing, f"absent: {missing}")
            tiny = counts[counts < 20]
            if len(tiny):
                info(f"classes with <20 test images (per-class recall not measurable): "
                     f"{tiny.to_dict()}")


def check_checkpoint(cfg: dict, ckpt_path: str, splits: dict) -> None:
    """The important one: strict=False hides trainable weights that never loaded."""
    section("4. CHECKPOINT COMPLETENESS")
    dcfg, mcfg = cfg["data"], cfg["model"]
    device = torch.device("cpu")
    metadata = MetadataProcessor(
        processed_dir=dcfg["processed_dir"], numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    model = build_model(
        mcfg,
        categorical_cardinalities=[metadata.num_categories(f) for f in dcfg["categorical_fields"]],
        n_numeric=len(dcfg["numeric_fields"]),
        n_comorbidities=len(metadata.comorbidity_vocab),
        num_classes=mcfg["num_classes"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    info(f"checkpoint epoch {ckpt.get('epoch')}, best val QWK {ckpt.get('best_qwk')}")
    state = ckpt["model"]

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    ckpt_keys = set(state.keys())
    never_loaded = sorted(trainable - ckpt_keys)
    check("every trainable parameter is present in the checkpoint", not never_loaded,
          f"{len(never_loaded)} would stay RANDOM: {never_loaded[:6]}")

    extra = sorted(ckpt_keys - set(model.state_dict().keys()))
    check("no checkpoint keys the model does not have", not extra,
          f"{len(extra)}: {extra[:6]}")

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    missing, unexpected = model.load_state_dict(state, strict=False)
    unchanged = [n for n, p in model.named_parameters()
                 if p.requires_grad and torch.equal(p.detach(), before[n])]
    # A trainable tensor identical to its init after loading means the load did not reach it.
    check("every trainable parameter actually changed on load", not unchanged,
          f"{len(unchanged)} unchanged: {unchanged[:6]}")
    info(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
         f"(missing should be the frozen backbone only)")
    info(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
         f"of {sum(p.numel() for p in model.parameters()):,}")

    # --- reproduce the reported metrics from scratch ---
    section("5. METRIC REPRODUCTION")
    label_col, n_classes = dcfg["label_col"], mcfg["num_classes"]
    ds = BRSETDataset(
        split_csv=str(Path(dcfg["processed_dir"]) / "test.csv"), images_dir=dcfg["images_dir"],
        metadata=metadata, label_col=label_col, image_size=dcfg["image_size"], train=False,
        label_map=dcfg.get("label_map"),
    )
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=16, shuffle=False, num_workers=2):
            trues.extend(batch.pop("label").tolist())
            preds.extend(model(batch).argmax(dim=-1).tolist())

    truth_from_csv = apply_label_map(
        pd.read_csv(Path(dcfg["processed_dir"]) / "test.csv")[label_col],
        dcfg.get("label_map")).tolist()
    check("dataset labels match the CSV row-for-row", trues == truth_from_csv,
          "the dataloader's order or label mapping differs from the CSV")

    m = compute_metrics(trues, preds)
    info(f"recomputed: qwk {m['qwk']:.4f}  accuracy {m['accuracy']:.4f}  "
         f"macro_f1 {m['macro_f1']:.4f}")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(trues, preds):
        cm[t, p] += 1
    info("confusion (rows=true):")
    for row in cm:
        info("  " + "".join(f"{v:>6}" for v in row))
    recalls = cm.diagonal() / cm.sum(1)
    info("per-class recall: " + "  ".join(f"{r * 100:.1f}%" for r in recalls))
    info(f"balanced accuracy: {recalls.mean():.4f}")
    check("accuracy equals the confusion matrix diagonal",
          abs(cm.diagonal().sum() / cm.sum() - m["accuracy"]) < 1e-9)

    if n_classes == 5:
        collapse = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        c3 = np.zeros((3, 3), dtype=int)
        for t, p in zip(trues, preds):
            c3[collapse[t], collapse[p]] += 1
        info("")
        info("collapsed to 3 classes (none / mid / severe), for comparison with 3-class runs:")
        for row in c3:
            info("  " + "".join(f"{v:>6}" for v in row))
        r3 = c3.diagonal() / c3.sum(1)
        info(f"accuracy {c3.diagonal().sum() / c3.sum():.4f}   "
             f"balanced accuracy {r3.mean():.4f}")
        info("per-class recall: " + "  ".join(f"{r * 100:.1f}%" for r in r3))
        diseased, missed = c3[1:, :].sum(), c3[1:, 0].sum()
        info(f"sensitivity {100 * (1 - missed / diseased):.1f}%  "
             f"({diseased - missed}/{diseased});  severe called healthy {c3[2, 0]}/{c3[2].sum()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="omit to run the data checks only")
    args = parser.parse_args()

    cfg = resolve(load_config(args.config), ROOT)
    dcfg, mcfg = cfg["data"], cfg["model"]
    print(f"config:     {args.config}")
    print(f"checkpoint: {args.checkpoint or '(none -- data checks only)'}")
    print(f"model:      {'image-only' if mcfg.get('image_only') else 'gated fusion'}, "
          f"{mcfg['num_classes']} classes, proj_dim={mcfg.get('proj_dim')}")

    splits = check_splits(dcfg, dcfg["label_col"])
    if not splits:
        sys.exit(1)
    check_no_leakage_in_stats(dcfg, splits)
    check_label_map(dcfg, mcfg, splits)
    if args.checkpoint:
        check_checkpoint(cfg, args.checkpoint, splits)

    failed = [n for n, ok in RESULTS if not ok]
    section("SUMMARY")
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for n in failed:
            print(f"  FAILED: {n}")
        sys.exit(1)
    print("  No problems found.")


if __name__ == "__main__":
    main()
