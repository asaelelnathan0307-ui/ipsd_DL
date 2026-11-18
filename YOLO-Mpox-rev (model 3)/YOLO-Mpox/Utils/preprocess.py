# %%
import os
import cv2 as cv
import numpy as np

PREPROCESS_MODES = ("none", "clahe", "clahe_denoise")


def apply_clahe(img, clip_limit=2.0, tile=8):
    """CLAHE pada kanal L (ruang warna LAB): memperjelas kontras lesi kulit."""
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)


def preprocess_image(img, mode="clahe", size=224):
    """Pre-processing satu citra BGR -> citra BGR ukuran (size, size).

    none          : hanya resize
    clahe         : resize + CLAHE
    clahe_denoise : resize + Gaussian blur ringan + CLAHE
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"mode harus salah satu dari {PREPROCESS_MODES}")
    img = cv.resize(img, (size, size), interpolation=cv.INTER_AREA)
    if mode == "clahe_denoise":
        img = cv.GaussianBlur(img, (3, 3), 0)
    if mode in ("clahe", "clahe_denoise"):
        img = apply_clahe(img)
    return img.astype(np.uint8)

# file ini tersimpan di folder Utils/ pada Server Colab
if __name__ == "__main__" and "In" in globals():
    os.makedirs("Utils", exist_ok=True)
    with open("Utils/preprocess.py", "w") as f:
        f.write(In[-1])
    print("Utils/preprocess.py berhasil disimpan!")
# %%
