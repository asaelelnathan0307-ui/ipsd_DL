# %%
import argparse
import sys
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from Models.YOLO import buildYOLO
from Utils.getData import downloadDataset, buildFoldDataset, getImageLabel


def parseArgs(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=224)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--preprocess", default="clahe", choices=["none", "clahe", "clahe_denoise"])
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    if argv is None and 'ipykernel' in sys.modules:
        argv = []  # dijalankan sebagai sel: pakai nilai default
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    device = 0 if torch.cuda.is_available() else "cpu"
    orig_path, aug_path = downloadDataset()
    os.makedirs("Results/test", exist_ok=True)

    names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    all_scores = {n: [] for n in names}

    for fold in args.folds:
        fold_dir = buildFoldDataset(fold, orig_path, aug_path, mode=args.preprocess, size=args.imgsz)
        items, classes = getImageLabel(f"{fold_dir}/test")
        paths = [p for p, _ in items]
        ground_truth = [t for _, t in items]

        model = buildYOLO(f"Results/train/fold{fold}/weights/best.pt")
        prediction, probs = [], []
        for r in model.predict(paths, imgsz=args.imgsz, batch=args.batch, device=device,
                               stream=True, verbose=False):
            p = r.probs.data.detach().cpu().numpy()
            probs.append(p)
            prediction.append(int(np.argmax(p)))
        probs = np.array(probs)

        cf_matrix = confusion_matrix(ground_truth, prediction)
        print(f"\n=== Fold {fold} ===")
        print(cf_matrix)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                             index=classes, columns=classes)
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f"Results/test/confusion_matrix_fold{fold}.png")
        plt.close()

        scores = {
            "accuracy": accuracy_score(ground_truth, prediction),
            "precision": precision_score(ground_truth, prediction, average="weighted", zero_division=0),
            "recall": recall_score(ground_truth, prediction, average="weighted", zero_division=0),
            "f1": f1_score(ground_truth, prediction, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(ground_truth, probs, multi_class="ovr", average="weighted"),
        }
        for n in names:
            print(f"{n} score = ", scores[n])
            all_scores[n].append(scores[n])
        with open(f"Results/test/metrics_fold{fold}.json", "w") as f:
            json.dump(scores, f, indent=2)

    print("\n=== RERATA 5-FOLD CROSS VALIDATION ===")
    mean = {n: float(np.mean(v)) for n, v in all_scores.items()}
    for n in names:
        print(f"Mean {n} score = ", mean[n])
    with open("Results/test/metrics_mean.json", "w") as f:
        json.dump(mean, f, indent=2)


if __name__ == "__main__":
    main()
# %%
