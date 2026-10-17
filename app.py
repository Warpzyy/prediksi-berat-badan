import json
import os

import joblib
from flask import Flask, render_template, request

app = Flask(__name__)
import base64

def _gambar(nama):
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", nama), "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except OSError:
        return ""

IMG_SCATTER = _gambar("scatter.png")
IMG_PRED = _gambar("pred_vs_actual.png")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model", "model.pkl"))
with open(os.path.join(BASE_DIR, "model", "metrics.json"), encoding="utf-8") as f:
    info = json.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None
    error = None
    peringatan = None
    tinggi = ""

    if request.method == "POST":
        tinggi = request.form.get("tinggi", "").strip().replace(",", ".")
        try:
            nilai = float(tinggi)
            if not 100 <= nilai <= 250:
                error = "Masukkan tinggi badan antara 100 dan 250 cm."
            else:
                hasil = float(model.predict([[nilai]])[0])
                if not info["height_min_cm"] <= nilai <= info["height_max_cm"]:
                    peringatan = (
                        f"Tinggi ini di luar rentang data latih "
                        f"({info['height_min_cm']:.0f}-{info['height_max_cm']:.0f} cm), "
                        f"sehingga prediksi kurang dapat diandalkan."
                    )
        except ValueError:
            error = "Input harus berupa angka, misalnya 170."

    return render_template(
        "index.html",
        hasil=hasil,
        error=error,
        peringatan=peringatan,
        tinggi=tinggi,
        info=info,
        img_scatter=IMG_SCATTER,
        img_pred=IMG_PRED,
    )


if __name__ == "__main__":
    app.run(debug=True)
