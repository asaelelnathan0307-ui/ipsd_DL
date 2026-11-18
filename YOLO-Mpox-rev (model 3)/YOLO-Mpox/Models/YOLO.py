# %%
import os
"""Wrapper model YOLO (Ultralytics, varian klasifikasi `-cls`).

YOLO-cls memakai loss CrossEntropy untuk klasifikasi 6 kelas
(Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox).
"""
from ultralytics import YOLO


def buildYOLO(weights="yolo11n-cls.pt", scratch=False):
    """Buat model YOLO klasifikasi.

    weights : nama bobot (mis. yolo11n-cls.pt, yolov8s-cls.pt).
    scratch : True -> pakai arsitektur (.yaml) tanpa bobot pretrained,
              training murni dari nol hanya dengan data tugas ini.
    """
    if scratch:
        weights = weights.replace(".pt", ".yaml")
    return YOLO(weights)

# file ini tersimpan di folder Models/ pada Server Colab
if __name__ == "__main__" and "In" in globals():
    os.makedirs("Models", exist_ok=True)
    with open("Models/YOLO.py", "w") as f:
        f.write(In[-1])
    print("Models/YOLO.py berhasil disimpan!")
# %%
