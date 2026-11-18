# %%
import os
import kagglehub
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Adam
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN

def main():
    BATCH_SIZE = 64
    EPOCH = 35
    LEARNING_RATE = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Menggunakan perangkat:", device)
    print("CUDA Available:", torch.cuda.is_available())
    print("Device Count:", torch.cuda.device_count())
    # Download dataset langsung di server Colab
    base_path = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")
    orig_path = [os.path.join(r, "FOLDS") for r, d, _ in os.walk(base_path) if "FOLDS" in d][0]
    aug_path = [os.path.join(r, "FOLDS_AUG") for r, d, _ in os.walk(base_path) if "FOLDS_AUG" in d][0]

    for fold in range(1, 6):
        print(f"\n--- Training Fold {fold} ---")
        train_data = getImageLabel(folder=f"{orig_path}/fold{fold}/Train") + getImageLabel(folder=f"{aug_path}/fold{fold}_AUG/Train")
        valid_data = getImageLabel(folder=f"{orig_path}/fold{fold}/Valid")

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        model = SimpleCNN(input_dim=200, input_c=3, output=6, hidden_dim=128, dropout=0.5, device=device).to(device)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        loss_function = nn.CrossEntropyLoss()

        loss_train_all, loss_test_all = [], []
        for epoch in range(EPOCH):
            train_loss = 0
            test_loss = 0
            model.train()
            for batch, (src, trg) in enumerate(train_loader):
                src = torch.permute(src, (0, 3, 1, 2)).to(device)
                trg = trg.to(device)
                pred = model(src)
                loss = loss_function(pred, trg)
                train_loss += loss.detach().cpu().numpy()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            for batch, (src, trg) in enumerate(test_loader):
                src = torch.permute(src, (0, 3, 1, 2)).to(device)
                trg = trg.to(device)
                pred = model(src)
                loss = loss_function(pred, trg)
                test_loss += loss.detach().cpu().numpy()

            loss_train_all.append(train_loss / len(train_loader))
            loss_test_all.append(test_loss / len(test_loader))
            print("epoch = ", epoch+1, "train loss = ", train_loss / len(train_loader), ", test loss = ", test_loss / len(test_loader))

            if (epoch+1) == EPOCH:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss / len(train_loader),
                }, f"./SimpleCNN_fold{fold}_" + str(epoch+1) + ".pt")

        plt.figure()
        plt.plot(range(EPOCH), loss_train_all, color="#931a00", label='Training')
        plt.plot(range(EPOCH), loss_test_all, color="#3399e6", label='Testing')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./training_fold{fold}.png")
        plt.close()

if __name__ == "__main__":
    main()
# %%