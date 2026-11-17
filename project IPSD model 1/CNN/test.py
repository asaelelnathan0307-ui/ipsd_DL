# %%
import os

from zmq import device
import kagglehub
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    BATCH_SIZE = 32
    EPOCH = 100
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_path = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")
    orig_path = [os.path.join(r, "FOLDS") for r, d, _ in os.walk(base_path) if "FOLDS" in d][0]

    classes = ('Chickenpox', 'Cowpox', 'HFMD', 'Healthy', 'Measles', 'Monkeypox')
    acc_all, prec_all, rec_all, f1_all, roc_all = [], [], [], [], []

    for fold in range(1, 6):
        test_loader = DataLoader(getImageLabel(folder=f"{orig_path}/fold{fold}/Test"), batch_size=BATCH_SIZE, shuffle=False)

        model = SimpleCNN(input_dim=200, input_c=3, output=6, hidden_dim=128, dropout=0.5, device=device).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        checkpoint = torch.load(f'SimpleCNN_fold{fold}_50.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss_batch = checkpoint['loss']

        prediction, ground_truth, probs = [], [], []
        with torch.no_grad():
            model.eval()
            for batch, (src, trg) in enumerate(test_loader):
                src = torch.permute(src, (0, 3, 1, 2)).to(device)

                pred = model(src)
                probs.extend(torch.exp(pred).detach().cpu().numpy())
                prediction.extend(torch.argmax(pred, dim=1).detach().cpu().numpy())
                ground_truth.extend(torch.argmax(trg, dim=1).detach().cpu().numpy())

        # Build confusion matrix
        cf_matrix = confusion_matrix(ground_truth, prediction)
        print(f"\n=== Fold {fold} ===")
        print(cf_matrix)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                             columns=[i for i in classes])
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'confusion_matrix_fold{fold}.png')
        plt.close()

        acc = accuracy_score(ground_truth, prediction)
        prec = precision_score(ground_truth, prediction, average='weighted')
        rec = recall_score(ground_truth, prediction, average='weighted')
        f1 = f1_score(ground_truth, prediction, average='weighted')
        roc = roc_auc_score(ground_truth, np.array(probs), multi_class='ovr', average='weighted')

        print("accuracy score = ", acc)
        print("precision score = ", prec)
        print("recall score = ", rec)
        print("f1 score = ", f1)
        print("roc auc score = ", roc)

        acc_all.append(acc); prec_all.append(prec); rec_all.append(rec); f1_all.append(f1); roc_all.append(roc)

    print("\n=== RERATA 5-FOLD CROSS VALIDATION ===")
    print("Mean accuracy score = ", np.mean(acc_all))
    print("Mean precision score = ", np.mean(prec_all))
    print("Mean recall score = ", np.mean(rec_all))
    print("Mean f1 score = ", np.mean(f1_all))
    print("Mean roc auc score = ", np.mean(roc_all))

if __name__ == "__main__":
    main()

# %%
# %%
import glob
import base64
import zipfile
from IPython.display import HTML, Image, display

# 0. Buat file hasil_grafik.zip secara otomatis dari semua gambar PNG yang ada
with zipfile.ZipFile("hasil_grafik.zip", "w") as zipf:
    for img_path in glob.glob("*.png"):
        zipf.write(img_path)
print("File hasil_grafik.zip berhasil dibuat!")

# 1. Buat link download langsung yang bisa diklik di VS Code
with open("hasil_grafik.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
display(HTML(f'<a download="hasil_grafik.zip" href="data:application/zip;base64,{b64}" style="font-size:16px; font-weight:bold;">KLIK DI SINI UNTUK DOWNLOAD hasil_grafik.zip</a>'))

# 2. Tampilkan gambar langsung di layar kanan VS Code
for img_path in sorted(glob.glob("*.png")):
    print(f"=== {img_path} ===")
    display(Image(filename=img_path))

# %%
