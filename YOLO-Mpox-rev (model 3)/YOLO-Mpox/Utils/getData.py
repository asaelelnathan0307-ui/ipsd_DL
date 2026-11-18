# %%
import os
import shutil
from os import listdir

import cv2 as cv
import kagglehub

from Utils.preprocess import preprocess_image

DATASET = "joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def downloadDataset():
    """Download dataset (cache kagglehub) dan kembalikan path FOLDS & FOLDS_AUG."""
    base_path = kagglehub.dataset_download(DATASET)
    orig_path = [os.path.join(r, "FOLDS") for r, d, _ in os.walk(base_path) if "FOLDS" in d][0]
    aug_path = [os.path.join(r, "FOLDS_AUG") for r, d, _ in os.walk(base_path) if "FOLDS_AUG" in d][0]
    return orig_path, aug_path


def _copyFolder(src_folder, dst_folder, mode, size, prefix=""):
    """Preprocess semua citra di src_folder/<kelas>/* ke dst_folder/<kelas>/*."""
    for cls in sorted(listdir(src_folder)):
        os.makedirs(f"{dst_folder}/{cls}", exist_ok=True)
        for name in listdir(f"{src_folder}/{cls}"):
            if not name.lower().endswith(IMG_EXT):
                continue
            img = cv.imread(f"{src_folder}/{cls}/{name}")
            if img is None:
                continue
            img = preprocess_image(img, mode=mode, size=size)
            stem = os.path.splitext(name)[0]
            cv.imwrite(f"{dst_folder}/{cls}/{prefix}{stem}.jpg", img)


def buildFoldDataset(fold, orig_path, aug_path, out_root="Data", mode="clahe", size=224):
    """Susun dataset satu fold dalam format yang dibaca YOLO-cls.

    Data/<mode>_<size>/fold{n}/
        train/<kelas>/  = original Train + augmented Train  (SESUAI KETENTUAN)
        val/<kelas>/    = original Valid
        test/<kelas>/   = original Test
    Valid & Test TIDAK pernah masuk ke train/.
    """
    fold_dir = f"{out_root}/{mode}_{size}/fold{fold}"
    if os.path.isdir(f"{fold_dir}/test"):  # sudah pernah dibuat
        return fold_dir
    if os.path.isdir(fold_dir):
        shutil.rmtree(fold_dir)

    _copyFolder(f"{orig_path}/fold{fold}/Train", f"{fold_dir}/train", mode, size)
    _copyFolder(f"{aug_path}/fold{fold}_AUG/Train", f"{fold_dir}/train", mode, size, prefix="aug_")
    _copyFolder(f"{orig_path}/fold{fold}/Valid", f"{fold_dir}/val", mode, size)
    _copyFolder(f"{orig_path}/fold{fold}/Test", f"{fold_dir}/test", mode, size)
    return fold_dir


def getImageLabel(folder):
    """Daftar (path_citra, indeks_kelas) dari folder/<kelas>/*, kelas diurut alfabet.

    Urutan kelas sama dengan yang dipakai YOLO-cls (alfabetis).
    """
    classes = sorted(listdir(folder))
    items = []
    for i, cls in enumerate(classes):
        for name in sorted(listdir(f"{folder}/{cls}")):
            if name.lower().endswith(IMG_EXT):
                items.append((f"{folder}/{cls}/{name}", i))
    return items, classes

# file ini tersimpan di folder Utils/ pada Server Colab
if __name__ == "__main__" and "In" in globals():
    os.makedirs("Utils", exist_ok=True)
    with open("Utils/getData.py", "w") as f:
        f.write(In[-1])
    print("Utils/getData.py berhasil disimpan!")
# %%
