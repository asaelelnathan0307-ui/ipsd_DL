# %%
import os
import cv2 as cv
import torch
import numpy as np
from os import listdir
from torch.utils.data import Dataset

class getImageLabel(Dataset):
    def __init__(self, folder='./Dataset'):
        self.dataset = []
        to_one_hot = np.eye(6)
        for i, dino in enumerate(sorted(listdir(folder))):
            for image_name in listdir(folder + "/" + dino):
                image = cv.resize(cv.imread(folder + "/" + dino + "/" + image_name), (100, 100)) / 255
                self.dataset.append([image, to_one_hot[i]])

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)

# file ini tersimpan di folder Utils/ pada Server Colab
if __name__ == "__main__" and "In" in globals():
    os.makedirs("Utils", exist_ok=True)
    with open("Utils/getData.py", "w") as f:
        f.write(In[-1])
    print("Utils/getData.py berhasil disimpan!")
# %%
