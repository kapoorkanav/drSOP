"""
Train RETFound -> 4 independent binary heads (D / G / C / A) on ODIR-5K. Image only.

    python scripts/prepare_data_odir.py --config configs/odir_lora.yaml   # once
    python train_odir.py --config configs/odir_lora.yaml

Same backbone, LoRA, checkpointing and early-stopping machinery as train.py; what differs is
the task: multi-label, so BCE with per-disease pos_weight instead of cross-entropy, sigmoid
probabilities instead of argmax, and early stopping on mean per-disease AUC instead of QWK.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.odir_dataset import ODIRDataset  # noqa: E402
from drsop.losses import build_multilabel_loss  # noqa: E402
from drsop.metrics import compute_multilabel_metrics  # noqa: E402
from drsop.models.image_only_model import ImageOnlyModel  # noqa: E402
from train import set_seed, trainable_state_dict  # noqa: E402


def load_codes(processed_dir: str) -> list:
    with open(Path(processed_dir) / "disease_codes.json") as f:
        return json.load(f)


def build_odir_model(mcfg: dict, n_diseases: int, img_size: int = 224) -> ImageOnlyModel:
    # img_size must match data.image_size: the ViT's patch grid / position embeddings are
    # built for exactly that size. Callers pass data.image_size so the two can't drift apart.
    return ImageOnlyModel(retfound_cfg=mcfg, head_cfg=mcfg["head"],
                          proj_dim=mcfg.get("proj_dim"), num_classes=n_diseases,
                          img_size=img_size)


def run_epoch(model, loader, device, criterion, codes, optimizer=None, scaler=None, desc="",
              accum_steps: int = 1):
    # accum_steps > 1: gradient accumulation -- sum gradients over that many small batches
    # before one optimizer step, so a high-resolution run that only fits a batch of 4 in GPU
    # memory still takes steps of the same effective size (4 x 2 = 8) as the 224 run.
    # accum_steps=1 is exactly the original one-step-per-batch behaviour.
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_true, all_prob = 0.0, [], []
    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc=desc, leave=False)
    for step, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("label")

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(batch)
            loss = criterion(logits.float(), labels)

        if train_mode:
            scaled = loss / accum_steps
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_prob.extend(torch.sigmoid(logits.float()).detach().cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_multilabel_metrics(all_true, all_prob, codes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def summary(m: dict, codes: list) -> str:
    per = " ".join(f"{c}={m[f'{c}_auc']:.3f}" for c in codes)
    return f"loss {m['loss']:.4f} | mean AUC {m['mean_auc']:.4f} ({per}) | mean F1 {m['mean_f1']:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    processed_dir = dcfg["processed_dir"]
    codes = load_codes(processed_dir)

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(split: str, train: bool):
        ds = ODIRDataset(str(Path(processed_dir) / f"{split}.csv"), dcfg["images_dir"],
                         codes, dcfg["image_size"], train)
        return DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=train,
                          num_workers=tcfg["num_workers"], pin_memory=True, drop_last=train), ds

    train_loader, train_ds = make_loader("train", train=True)
    val_loader, _ = make_loader("val", train=False)

    model = build_odir_model(mcfg, len(codes), dcfg["image_size"]).to(device)
    accum_steps = tcfg.get("grad_accum_steps", 1)
    print(f"image size {dcfg['image_size']}, batch {tcfg['batch_size']} x accum {accum_steps}"
          f" = effective batch {tcfg['batch_size'] * accum_steps}")
    criterion = build_multilabel_loss(tcfg["class_weighted_loss"], train_ds.labels(), device)
    if tcfg["class_weighted_loss"]:
        pw = criterion.pos_weight.tolist()
        print("pos_weight per disease: " + ", ".join(f"{c}={w:.1f}" for c, w in zip(codes, pw)))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if (tcfg["amp"] and device.type == "cuda") else None

    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, best_auc = 0, -1.0
    patience_left = tcfg["early_stopping_patience"]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)  # frozen backbone isn't in the ckpt
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_auc = ckpt["best_mean_auc"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, tcfg["epochs"]):
        tr = run_epoch(model, train_loader, device, criterion, codes, optimizer, scaler,
                       desc=f"epoch {epoch} train", accum_steps=accum_steps)
        va = run_epoch(model, val_loader, device, criterion, codes, desc=f"epoch {epoch} val")
        print(f"[epoch {epoch}] train: {summary(tr, codes)}")
        print(f"[epoch {epoch}]   val: {summary(va, codes)}")

        improved = va["mean_auc"] > best_auc
        if improved:
            best_auc = va["mean_auc"]
            patience_left = tcfg["early_stopping_patience"]

        ckpt = {"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_mean_auc": best_auc, "codes": codes, "config": cfg}
        torch.save(ckpt, output_dir / "last.pt")

        if improved:
            torch.save(ckpt, output_dir / "best.pt")
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(va, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best val mean AUC={best_auc:.4f})")
                break


if __name__ == "__main__":
    main()
