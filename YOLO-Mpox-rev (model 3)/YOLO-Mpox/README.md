# Klasifikasi Lesi Kulit Mpox dengan YOLO (Ultralytics YOLO-cls)

Tugas Minggu 10 – Studi Kasus Pembelajaran Mesin.
Klasifikasi 6 kelas citra kulit: **Chickenpox, Cowpox, HFMD, Healthy, Measles, Monkeypox**
menggunakan dataset [Mpox Skin Lesion Dataset (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20).

## Struktur Proyek
```
YOLO-Mpox/
├── Models/YOLO.py        # pembuat model YOLO-cls
├── Utils/getData.py      # download dataset & susun data per fold
├── Utils/preprocess.py   # pre-processing citra (resize, CLAHE, denoise)
├── colab_setup.py        # sel setup Colab: GPU, pip, kredensial, unduh hasil
├── train.py              # training 5-fold
├── test.py               # evaluasi 5-fold + rerata
├── Results/train/        # grafik loss training per fold
├── Results/test/         # confusion matrix & metrik per fold
├── requirements.txt
└── .gitignore            # mengecualikan *.pt dan dataset
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
# butuh kredensial Kaggle (KAGGLE_USERNAME / KAGGLE_KEY) untuk kagglehub
python train.py --epochs 35 --preprocess clahe
python test.py --preprocess clahe
```
Opsi: `--model yolo11s-cls.pt`, `--scratch` (tanpa bobot pretrained), `--preprocess none|clahe|clahe_denoise`, `--folds 1 2`.

## Metode
- **Data training**: original `Train` + augmented `Train` tiap fold. **Validasi**: original `Valid`. **Testing**: original `Test`. Valid/Test tidak pernah dipakai untuk training.
- **Pre-processing**: resize 224×224 (INTER_AREA) + CLAHE pada kanal L (LAB) untuk memperjelas kontras lesi. Pre-processing yang sama diterapkan ke semua split.
- **Model**: YOLO11n-cls (Ultralytics), optimizer Adam, lr 0.001, batch 64, 35 epoch (early stopping patience 10). Loss: CrossEntropy.
- **Evaluasi**: confusion matrix, accuracy, precision, recall, F1 (weighted), ROC-AUC (one-vs-rest, weighted); dirata-rata dari 5-fold.

## Hasil
> Isi setelah menjalankan `test.py` (lihat `Results/test/metrics_mean.json`).

| Metrik | Rerata 5-fold |
|---|---|
| Accuracy | – |
| Precision | – |
| Recall | – |
| F1 Score | – |
| ROC-AUC | – |

### Kurva Training
![fold1](Results/train/training_fold1.png)

### Confusion Matrix
![cm1](Results/test/confusion_matrix_fold1.png)

## Analisis
> Tulis pembahasan: kelas yang sering tertukar, pengaruh pre-processing (bandingkan `none` vs `clahe`), dan kekurangan model.
