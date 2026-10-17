"""
Prediksi Berat Badan Berdasarkan Tinggi Badan (Regresi Linear Sederhana)

Jalankan:  python train_model.py
Output  :  model/model.pkl, model/metrics.json, static/scatter.png, static/pred_vs_actual.png
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa layar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 1. Muat data
# ---------------------------------------------------------------
df = pd.read_csv("data/weight-height.csv")
print("--- 5 Baris Pertama ---")
print(df.head())
print("\n--- Info ---")
print(df.info())

# ---------------------------------------------------------------
# 2. Pembersihan data
# ---------------------------------------------------------------
print("\nMissing values per kolom:\n", df.isnull().sum())
print("Jumlah duplikat:", df.duplicated().sum())

# Dataset asli: tinggi dalam inci, berat dalam pon -> konversi ke cm dan kg
df["Height_cm"] = df["Height"] * 2.54
df["Weight_kg"] = df["Weight"] * 0.45359237

# Cek outlier dengan metode IQR (hanya dicek & didokumentasikan, tidak dibuang)
for col in ["Height_cm", "Weight_kg"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    print(f"Outlier (IQR) pada {col}: {n_out} dari {len(df)} baris")

print("\n--- Statistik Deskriptif ---")
print(df[["Height_cm", "Weight_kg"]].describe().round(2))

# ---------------------------------------------------------------
# 3. EDA: validasi asumsi linear dengan scatterplot
# ---------------------------------------------------------------
corr = df["Height_cm"].corr(df["Weight_kg"])
print(f"\nKorelasi Height_cm vs Weight_kg: {corr:.3f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Height_cm", y="Weight_kg", hue="Gender", alpha=0.4, s=15)
plt.title("Tinggi Badan vs Berat Badan")
plt.xlabel("Tinggi Badan (cm)")
plt.ylabel("Berat Badan (kg)")
plt.tight_layout()
plt.savefig("static/scatter.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 4. Model: pisah X dan y, bagi data latih/uji
# ---------------------------------------------------------------
X = df[["Height_cm"]].values   # satu variabel independen
y = df["Weight_kg"].values     # variabel dependen

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------------
# 5. Evaluasi
# ---------------------------------------------------------------
pred = model.predict(X_test)
mae = metrics.mean_absolute_error(y_test, pred)
mse = metrics.mean_squared_error(y_test, pred)
rmse = float(np.sqrt(mse))
r2 = metrics.r2_score(y_test, pred)

print("\n--- Evaluasi (data uji) ---")
print(f"MAE  : {mae:.3f} kg")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f} kg")
print(f"R2   : {r2:.3f}")

print("\n--- Parameter Model ---")
print(f"Intercept (b0)  : {model.intercept_:.4f}")
print(f"Koefisien (b1)  : {model.coef_[0]:.4f}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, pred, alpha=0.3, s=12)
lo, hi = float(min(y_test)), float(max(y_test))
plt.plot([lo, hi], [lo, hi], color="red", lw=2, label="Garis ideal")
plt.xlabel("Berat Aktual (kg)")
plt.ylabel("Berat Prediksi (kg)")
plt.title("Berat Aktual vs Berat Prediksi (Data Uji)")
plt.legend()
plt.tight_layout()
plt.savefig("static/pred_vs_actual.png", dpi=120)
plt.close()

# ---------------------------------------------------------------
# 6. Simpan model + metrik (dipakai aplikasi Flask)
# ---------------------------------------------------------------
joblib.dump(model, "model/model.pkl")

info = {
    "intercept": float(model.intercept_),
    "coef": float(model.coef_[0]),
    "r2": float(r2),
    "rmse": rmse,
    "mae": float(mae),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "height_min_cm": float(df["Height_cm"].min()),
    "height_max_cm": float(df["Height_cm"].max()),
}
with open("model/metrics.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2)

print("\nModel tersimpan di model/model.pkl")
