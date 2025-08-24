#!/usr/bin/env python3
"""
train.py — Ultralytics YOLOv8 + Weights & Biases (CPU/WSL-safe)
- Uses python3
- workers=0 and device=cpu by default
- Always uploads best.pt as a W&B model Artifact ('yolo-best')
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import time

# Tame thread-hungry BLAS on laptops/WSL
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("WANDB__SERVICE_WAIT", "300")

try:
    from ultralytics import YOLO
except Exception as e:
    print("Ultralytics is not installed. Run: python3 -m pip install ultralytics", file=sys.stderr)
    raise

try:
    import wandb
except Exception as e:
    print("wandb is not installed. Run: python3 -m pip install wandb", file=sys.stderr)
    raise


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLOv8 and upload best.pt as a W&B model artifact")
    # W&B
    ap.add_argument("--entity", type=str, default=None, help="W&B entity/org (e.g., hocmemini-openinnovate-org)")
    ap.add_argument("--project", type=str, default="yolo-demo", help="W&B project name")
    ap.add_argument("--run_name", type=str, default=None, help="Optional W&B run name")
    ap.add_argument("--tags", nargs="*", default=[], help="Optional W&B tags")
    # Data/Model
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model or path")
    ap.add_argument("--data", type=str, default="coco128.yaml", help="Ultralytics data yaml")
    ap.add_argument("--epochs", type=int, default=5, help="Training epochs")
    ap.add_argument("--imgsz", type=int, default=640, help="Image size")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--optimizer", type=str, default="auto", help="auto/SGD/Adam/AdamW")
    ap.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 safest on WSL)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu or GPU index like '0'")
    # Optional hyperparams (sweep-friendly)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    # Misc
    ap.add_argument("--exist_ok", action="store_true", help="Allow reusing run dir name")
    ap.add_argument("--artifact_name", type=str, default="yolo-best")
    ap.add_argument("--artifact_description", type=str, default="Best YOLOv8 weights from training")
    return ap.parse_args()


def build_train_kwargs(a: argparse.Namespace) -> dict:
    kw = dict(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        optimizer=a.optimizer, workers=a.workers, device=a.device,
        project="runs/train", name=a.run_name or "wb_yolo_demo",
        exist_ok=a.exist_ok, save=True, seed=a.seed, verbose=True
    )
    if a.lr0 is not None: kw["lr0"] = a.lr0
    if a.weight_decay is not None: kw["weight_decay"] = a.weight_decay
    if a.momentum is not None: kw["momentum"] = a.momentum
    if a.patience is not None: kw["patience"] = a.patience
    return kw


def find_best_from_results(results) -> Path | None:
    save_dir = Path(getattr(results, "save_dir", ""))
    candidate = save_dir / "weights" / "best.pt"
    if candidate.exists():
        return candidate
    # fallback: newest best under runs/train
    cands = sorted(Path("runs/train").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def main():
    a = parse_args()
    run = wandb.init(
        project=a.project, entity=a.entity, name=a.run_name, tags=a.tags,
        config={
            "model": a.model, "data": a.data, "epochs": a.epochs, "imgsz": a.imgsz,
            "batch": a.batch, "optimizer": a.optimizer, "workers": a.workers,
            "device": a.device, "lr0": a.lr0, "weight_decay": a.weight_decay,
            "momentum": a.momentum, "patience": a.patience, "seed": a.seed
        },
        settings=wandb.Settings(start_method="thread"),
    )

    start = time.time()
    model = YOLO(a.model)
    print("Starting training…")
    results = model.train(**build_train_kwargs(a))

    best = find_best_from_results(results)
    if best and best.exists():
        print(f"[OK] best.pt: {best}")
        art = wandb.Artifact(a.artifact_name, type="model", description=a.artifact_description)
        art.add_file(str(best))
        run.log_artifact(art)
        print(f"[OK] Uploaded model artifact '{a.artifact_name}'")
    else:
        print("[WARN] best.pt not found — check Ultralytics output directories.")

    # Always finish cleanly
    run.summary["runtime_seconds"] = float(time.time() - start)
    wandb.finish()


if __name__ == "__main__":
    main()
