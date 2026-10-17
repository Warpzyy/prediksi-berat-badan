# Prediksi Berat Badan Berdasarkan Tinggi Badan

Regresi Linear Sederhana (scikit-learn) + aplikasi web Flask.

- Dataset: Weight-Height (Kaggle) - https://www.kaggle.com/mustafaali96/weight-height
- X: Height (cm, dikonversi dari inci) | y: Weight (kg, dikonversi dari pon)

## Menjalankan lokal
```
pip install -r requirements.txt
python train_model.py     # melatih model, menyimpan model/model.pkl
python app.py             # buka http://127.0.0.1:5000
```

## Deploy (Render)
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`
- Pastikan folder `model/` (model.pkl dan metrics.json) dan `static/` ikut di-push ke GitHub.
