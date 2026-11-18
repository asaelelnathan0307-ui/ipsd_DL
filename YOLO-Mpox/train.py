# %%
import argparse
import sys
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import torch

from Models.YOLO import buildYOLO
from Utils.getData import downloadDataset, buildFoldDataset


def parseArgs(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo11n-cls.pt")
    p.add_argument("--scratch", action="store_true", help="tanpa bobot pretrained")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--preprocess", default="clahe", choices=["none", "clahe", "clahe_denoise"])
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--patience", type=int, default=10)
    if argv is None and 'ipykernel' in sys.modules:
        argv = []  # dijalankan sebagai sel: pakai nilai default
    return p.parse_args(argv)


def plotLoss(csv_path, out_png):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    plt.figure()
    plt.plot(df["epoch"], df["train/loss"], color="#931a00", label="Training")
    plt.plot(df["epoch"], df["val/loss"], color="#3399e6", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.legend()
    plt.savefig(out_png)
    plt.close()


def main(argv=None):
    args = parseArgs(argv)
    device = 0 if torch.cuda.is_available() else "cpu"
    orig_path, aug_path = downloadDataset()
    out_dir = os.path.abspath("Results/train")

    for fold in args.folds:
        print(f"\n--- Training Fold {fold} ---")
        fold_dir = buildFoldDataset(fold, orig_path, aug_path, mode=args.preprocess, size=args.imgsz)

        model = buildYOLO(args.model, scratch=args.scratch)
        model.train(
            data=os.path.abspath(fold_dir),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr,
            optimizer="Adam",
            patience=args.patience,
            device=device,
            project=out_dir,
            name=f"fold{fold}",
            exist_ok=True,
            seed=fold,
            plots=True,
        )

        save_dir = str(model.trainer.save_dir)
        plotLoss(f"{save_dir}/results.csv", f"Results/train/training_fold{fold}.png")
        # bobot (*.pt) tetap lokal & diabaikan git (lihat .gitignore)
        print(f"Bobot fold {fold}: {save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
# %%
