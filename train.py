"""
Train the RETFound + metadata gated-fusion DR severity grading model.

    python train.py --config configs/default.yaml

Expects scripts/prepare_data.py to have already been run (produces data/processed/{train,val,test}.csv
plus metadata_stats.json / comorbidity_vocab.json), and RETFound_MAE cloned + checkpoint downloaded
per README.md.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from drsop.config import load_config, resolve  # noqa: E402
from drsop.data.brset_dataset import BRSETDataset  # noqa: E402
from drsop.data.metadata import MetadataProcessor  # noqa: E402
from drsop.losses import build_loss  # noqa: E402
from drsop.metrics import compute_metrics  # noqa: E402
from drsop.models.fusion_model import DRFusionModel  # noqa: E402


def trainable_state_dict(model: torch.nn.Module) -> dict:
    """Excludes the frozen RETFound backbone (~1.2GB of unchanging weights) from checkpoints --
    it gets reloaded fresh from the original RETFound checkpoint on every model construction
    anyway, so saving it every epoch is pure wasted disk I/O."""
    return {k: v for k, v in model.state_dict().items() if not k.startswith("image_encoder.backbone.")}


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_true, all_pred = 0.0, [], []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch.pop("label")

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(batch)
            loss = criterion(logits, labels)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(logits.argmax(dim=-1).cpu().tolist())

    metrics = compute_metrics(all_true, all_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cfg = resolve(load_config(args.config), root)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    label_col = dcfg["label_col"]
    num_classes = mcfg["num_classes"]

    set_seed(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = dcfg["processed_dir"]
    metadata = MetadataProcessor(
        processed_dir=processed_dir,
        numeric_fields=dcfg["numeric_fields"],
        categorical_fields=dcfg["categorical_fields"],
        comorbidity_field=dcfg["comorbidity_field"],
    )
    n_comorbidities = len(metadata.comorbidity_vocab)
    categorical_cardinalities = [metadata.num_categories(f) for f in dcfg["categorical_fields"]]

    def make_loader(split: str, train: bool):
        ds = BRSETDataset(
            split_csv=str(Path(processed_dir) / f"{split}.csv"),
            images_dir=dcfg["images_dir"],
            metadata=metadata,
            label_col=label_col,
            image_size=dcfg["image_size"],
            train=train,
        )
        return DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=train,
                           num_workers=tcfg["num_workers"], pin_memory=True, drop_last=train), ds

    train_loader, train_ds = make_loader("train", train=True)
    val_loader, _ = make_loader("val", train=False)

    model = DRFusionModel(
        retfound_cfg=mcfg,
        meta_cfg=mcfg["meta_encoder"],
        gate_cfg=mcfg["gate"],
        head_cfg=mcfg["head"],
        categorical_cardinalities=categorical_cardinalities,
        n_numeric=len(dcfg["numeric_fields"]),
        n_comorbidities=n_comorbidities,
        proj_dim=mcfg["proj_dim"],
        num_classes=num_classes,
    ).to(device)

    train_labels = train_ds.df[label_col].values
    criterion = build_loss(tcfg["class_weighted_loss"], train_labels, num_classes, device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if (tcfg["amp"] and device.type == "cuda") else None

    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_qwk, patience_left = -1.0, tcfg["early_stopping_patience"]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # strict=False: checkpoints only contain trainable params (frozen RETFound backbone is
        # excluded, see trainable_state_dict below) -- it gets reloaded fresh from the original
        # RETFound checkpoint during model construction above, so this is expected, not an error.
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_qwk = ckpt["best_qwk"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, tcfg["epochs"]):
        train_metrics = run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        val_metrics = run_epoch(model, val_loader, device, criterion)
        print(f"[epoch {epoch}] train: {train_metrics} | val: {val_metrics}")

        if val_metrics["qwk"] > best_qwk:
            best_qwk = val_metrics["qwk"]
            patience_left = tcfg["early_stopping_patience"]
            improved = True
        else:
            improved = False

        # Built after the best_qwk update above, so this field always reflects the true
        # current best -- not the pre-update value from before this epoch's comparison.
        ckpt = {
            "model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_qwk": best_qwk, "config": cfg,
        }
        torch.save(ckpt, output_dir / "last.pt")

        if improved:
            torch.save(ckpt, output_dir / "best.pt")
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best val QWK={best_qwk:.4f})")
                break


if __name__ == "__main__":
    main()
