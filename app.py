from flask import Flask, request, jsonify, render_template, send_file
import numpy as np
import joblib
import os
import io

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

def load(name):
    path = os.path.join(MODELS, name)
    if os.path.exists(path):
        return joblib.load(path)
    return None

model_stunting  = load("model_knn_stunting(1).pkl")
scaler_stunting = load("scaler(2).pkl")        # fitur: Umur, Jenis Kelamin, Tinggi Badan (3 fitur)
model_diabetes  = load("model_knn_diabetes.pkl")
scaler_diabetes = load("scaler_1.pkl")         # fitur: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level (8 fitur)
le_smoking      = load("le.pkl")               # LabelEncoder untuk smoking_history

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/stunting", methods=["POST"])
def predict_stunting():
    try:
        data   = request.get_json()
        age    = float(data["age"])     # bulan
        height = float(data["height"]) # cm
        sex    = int(data["sex"])       # 0=laki-laki, 1=perempuan

        # Validasi batas input
        if age < 0 or age > 32:
            return jsonify({"status": "error", "message": "Usia harus antara 0–32 bulan."}), 400
        if height < 45:
            return jsonify({"status": "error", "message": "Tinggi badan minimal 45 cm."}), 400
        if height > 100:
            return jsonify({"status": "error", "message": "Tinggi badan maksimal 100 cm."}), 400
        if sex not in (0, 1):
            return jsonify({"status": "error", "message": "Jenis kelamin tidak valid (0=laki-laki, 1=perempuan)."}), 400

        if not model_stunting or not scaler_stunting:
            return jsonify({
                "status": "error",
                "message": "Model stunting belum dimuat. Pastikan file model_knn_stunting(1).pkl dan scaler(2).pkl ada di folder models/."
            }), 503

        import pandas as pd

        # ── WHO Height-for-Age: median & SD per bulan ────────────────────────
        # Sumber: WHO Child Growth Standards
        who_boys = {
            0:  (49.9, 1.9), 1:  (54.7, 2.0), 2:  (58.4, 2.1), 3:  (61.4, 2.1),
            4:  (63.9, 2.2), 5:  (65.9, 2.2), 6:  (67.6, 2.3), 7:  (69.2, 2.3),
            8:  (70.6, 2.3), 9:  (72.0, 2.4), 10: (73.3, 2.4), 11: (74.5, 2.5),
            12: (75.7, 2.6), 13: (76.9, 2.6), 14: (78.0, 2.7), 15: (79.1, 2.7),
            16: (80.2, 2.8), 17: (81.2, 2.8), 18: (82.3, 2.8), 19: (83.2, 2.9),
            20: (84.2, 2.9), 21: (85.1, 3.0), 22: (86.0, 3.0), 23: (86.9, 3.1),
            24: (87.8, 3.1), 25: (88.6, 3.2), 26: (89.4, 3.2), 27: (90.3, 3.3),
            28: (91.1, 3.3), 29: (91.9, 3.4), 30: (92.7, 3.4), 31: (93.4, 3.5),
            32: (94.2, 3.5),
        }
        who_girls = {
            0:  (49.1, 1.9), 1:  (53.7, 2.0), 2:  (57.1, 2.1), 3:  (59.8, 2.1),
            4:  (62.1, 2.2), 5:  (64.0, 2.2), 6:  (65.7, 2.3), 7:  (67.3, 2.3),
            8:  (68.7, 2.3), 9:  (70.1, 2.4), 10: (71.5, 2.4), 11: (72.8, 2.5),
            12: (74.0, 2.5), 13: (75.2, 2.6), 14: (76.4, 2.6), 15: (77.5, 2.7),
            16: (78.6, 2.7), 17: (79.7, 2.8), 18: (80.7, 2.8), 19: (81.7, 2.9),
            20: (82.7, 2.9), 21: (83.7, 2.9), 22: (84.6, 3.0), 23: (85.5, 3.0),
            24: (86.4, 3.1), 25: (87.3, 3.1), 26: (88.1, 3.2), 27: (89.0, 3.2),
            28: (89.8, 3.3), 29: (90.6, 3.3), 30: (91.4, 3.4), 31: (92.2, 3.4),
            32: (93.0, 3.5),
        }

        # Hitung z-score WHO
        age_int = int(round(age))
        who_table = who_boys if sex == 0 else who_girls
        median, sd = who_table.get(age_int, (None, None))

        who_label  = None
        who_zscore = None
        if median and sd:
            who_zscore = (height - median) / sd
            if who_zscore < -3:
                who_label = "Sangat Pendek"
            elif who_zscore < -2:
                who_label = "Pendek"
            elif who_zscore <= 2:
                who_label = "Normal"
            else:
                who_label = "Tinggi"

        # ── Prediksi KNN ─────────────────────────────────────────────────────
        # Urutan fitur sesuai scaler: Umur, Jenis Kelamin, Tinggi Badan
        features = pd.DataFrame([[age, sex, height]],
                                columns=["Umur", "Jenis Kelamin", "Tinggi Badan"])
        features_scaled = scaler_stunting.transform(features)
        pred = int(model_stunting.predict(features_scaled)[0])
        try:
            proba_arr = model_stunting.predict_proba(features_scaled)[0]
            proba = float(proba_arr[pred])
        except Exception:
            proba = None

        # LabelEncoder urutan alfabetis scikit-learn dari nama kelas asli dataset:
        # 'normal'=0, 'severely stunted'=1, 'stunted'=2, 'tinggi'=3
        labels = {
            0: "Normal",
            1: "Sangat Pendek",
            2: "Pendek",
            3: "Tinggi"
        }
        knn_label = labels.get(pred, str(pred))

        # ── LOGIKA FINAL: WHO override jika z-score ekstrem ──────────────────
        # Jika WHO dan KNN tidak sepakat DAN z-score sangat ekstrem → pakai WHO
        if who_label and who_label != knn_label:
            if who_zscore is not None and (who_zscore < -2.5 or who_zscore > 2.5):
                final_label = who_label  # WHO menang untuk kasus ekstrem
            else:
                final_label = knn_label  # KNN menang untuk kasus borderline
        else:
            final_label = knn_label

        tips_map = {
            "Sangat Pendek": [
                "Segera bawa ke dokter atau ahli gizi anak.",
                "Tingkatkan asupan protein: telur, ikan, daging, tahu/tempe.",
                "Berikan suplemen zinc dan vitamin A sesuai anjuran dokter.",
                "Pantau berat dan tinggi badan setiap bulan di Posyandu.",
                "Pastikan sanitasi dan kebersihan lingkungan.",
                "Ikuti program PMT (Pemberian Makanan Tambahan) di Puskesmas.",
            ],
            "Pendek": [
                "Konsultasikan ke dokter atau ahli gizi.",
                "Tingkatkan asupan protein dan kalsium harian.",
                "Berikan suplemen zinc sesuai anjuran.",
                "Pantau pertumbuhan secara rutin di Posyandu.",
                "Pastikan imunisasi anak lengkap.",
            ],
            "Normal": [
                "Pertahankan pola makan bergizi seimbang.",
                "Berikan ASI eksklusif hingga 6 bulan.",
                "Pantau pertumbuhan secara rutin di Posyandu.",
                "Pastikan imunisasi anak lengkap.",
            ],
            "Tinggi": [
                "Pertahankan pola makan dan gaya hidup sehat.",
                "Pastikan asupan nutrisi tetap seimbang.",
                "Lanjutkan pemantauan pertumbuhan rutin.",
            ],
        }

        return jsonify({
            "status": "ok",
            "prediction": pred,
            "label": final_label,
            "confidence": round(proba * 100, 1) if proba is not None else None,
            "who_zscore": round(who_zscore, 2) if who_zscore is not None else None,
            "who_label": who_label,
            "tips": tips_map.get(final_label, []),
            "inputs": {"age_bulan": age, "height_cm": height, "sex": "Laki-laki" if sex == 0 else "Perempuan"},
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        # Fitur sesuai scaler_1.pkl: gender, age, hypertension, heart_disease, smoking_history, bmi, HbA1c_level, blood_glucose_level
        gender           = int(data["gender"])              # 0=female, 1=male
        age              = float(data["age"])
        hypertension     = int(data["hypertension"])        # 0/1
        heart_disease    = int(data["heart_disease"])       # 0/1
        smoking_raw      = data["smoking_history"]          # string: 'never','former','current','ever','not current'
        bmi              = float(data["bmi"])
        hba1c            = float(data["hba1c"])
        blood_glucose    = float(data["blood_glucose"])

        # Validasi input
        if age < 1 or age > 120:
            return jsonify({"status": "error", "message": "Usia harus antara 1–120 tahun."}), 400
        if bmi < 10 or bmi > 80:
            return jsonify({"status": "error", "message": "BMI harus antara 10–80."}), 400
        if hba1c < 3 or hba1c > 15:
            return jsonify({"status": "error", "message": "HbA1c harus antara 3–15%."}), 400
        if blood_glucose < 0 or blood_glucose > 600:
            return jsonify({"status": "error", "message": "Glukosa darah harus antara 0–600 mg/dL."}), 400
        if hypertension not in (0, 1):
            return jsonify({"status": "error", "message": "Hipertensi: nilai 0 atau 1."}), 400
        if heart_disease not in (0, 1):
            return jsonify({"status": "error", "message": "Penyakit jantung: nilai 0 atau 1."}), 400
        if gender not in (0, 1):
            return jsonify({"status": "error", "message": "Gender tidak valid."}), 400

        if not model_diabetes or not scaler_diabetes:
            return jsonify({
                "status": "error",
                "message": "Model diabetes belum dimuat. Pastikan file model_knn_diabetes.pkl dan scaler_1.pkl ada di folder models/."
            }), 503

        # Encode smoking history menggunakan le.pkl
        # classes_: ['current' 'ever' 'former' 'never' 'not current']
        if le_smoking:
            try:
                smoking_encoded = int(le_smoking.transform([smoking_raw])[0])
            except Exception:
                smoking_encoded = 3  # default 'never'
        else:
            smoking_map = {"current": 0, "ever": 1, "former": 2, "never": 3, "not current": 4}
            smoking_encoded = smoking_map.get(smoking_raw, 3)

        import pandas as pd
        features = pd.DataFrame(
            [[gender, age, hypertension, heart_disease, smoking_encoded, bmi, hba1c, blood_glucose]],
            columns=["gender", "age", "hypertension", "heart_disease", "smoking_history",
                     "bmi", "HbA1c_level", "blood_glucose_level"]
        )

        features_scaled = scaler_diabetes.transform(features)
        pred = int(model_diabetes.predict(features_scaled)[0])
        try:
            proba = float(model_diabetes.predict_proba(features_scaled)[0][1])
        except Exception:
            proba = None

        labels = {0: "Tidak Diabetes", 1: "Diabetes"}
        label  = labels.get(pred, str(pred))

        tips_map = {
            "Tidak Diabetes": [
                "Pertahankan gaya hidup sehat dan aktif.",
                "Batasi konsumsi gula dan karbohidrat sederhana.",
                "Olahraga minimal 30 menit per hari.",
                "Periksa kadar gula darah secara berkala.",
            ],
            "Diabetes": [
                "Segera konsultasikan ke dokter endokrinologi.",
                "Ikuti diet diabetes: rendah gula, tinggi serat.",
                "Pantau kadar gula darah setiap hari.",
                "Minum obat sesuai resep dokter.",
                "Hindari merokok dan alkohol.",
                "Olahraga teratur seperti jalan kaki 30 menit/hari.",
            ],
        }

        gender_label = "Laki-laki" if gender == 1 else "Perempuan"
        return jsonify({
            "status": "ok",
            "prediction": pred,
            "label": label,
            "confidence": round(proba * 100, 1) if proba is not None else None,
            "tips": tips_map.get(label, []),
            "inputs": {
                "gender": gender_label,
                "age": age,
                "hypertension": "Ya" if hypertension else "Tidak",
                "heart_disease": "Ya" if heart_disease else "Tidak",
                "smoking": smoking_raw,
                "bmi": bmi,
                "HbA1c": hba1c,
                "blood_glucose": blood_glucose,
            },
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/download-pdf", methods=["POST"])
def download_pdf():
    """Generate a simple PDF result report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import cm
        import datetime

        data = request.get_json()
        buf  = io.BytesIO()
        doc  = SimpleDocTemplate(buf, pagesize=A4,
                                 leftMargin=2*cm, rightMargin=2*cm,
                                 topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = styles["Title"]
        story.append(Paragraph("🩺 Health Predictor — Laporan Hasil", title_style))
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(
            f"Digenerate pada: {datetime.datetime.now().strftime('%d %B %Y, %H:%M WIB')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.6*cm))

        # Result badge
        label     = data.get("label", "-")
        conf      = data.get("confidence", 0)
        pred_type = data.get("type", "Prediksi")
        story.append(Paragraph(f"<b>Jenis Prediksi:</b> {pred_type}", styles["Normal"]))
        story.append(Paragraph(f"<b>Hasil:</b> {label}", styles["Normal"]))
        story.append(Paragraph(f"<b>Tingkat Kepercayaan:</b> {conf}%", styles["Normal"]))
        story.append(Spacer(1, 0.4*cm))

        # Input table
        inputs = data.get("inputs", {})
        if inputs:
            story.append(Paragraph("<b>Data Input:</b>", styles["Normal"]))
            story.append(Spacer(1, 0.2*cm))
            table_data = [["Parameter", "Nilai"]] + [[k, str(v)] for k, v in inputs.items()]
            t = Table(table_data, colWidths=[8*cm, 8*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR",      (0, 0), (-1, 0), colors.white),
                ("ALIGN",          (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f9ff")]),
                ("GRID",           (0, 0), (-1, -1), 0.5, colors.HexColor("#bae6fd")),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.4*cm))

        # Tips
        tips = data.get("tips", [])
        if tips:
            story.append(Paragraph("<b>Rekomendasi Kesehatan:</b>", styles["Normal"]))
            for tip in tips:
                story.append(Paragraph(f"• {tip}", styles["Normal"]))

        story.append(Spacer(1, 0.8*cm))
        story.append(Paragraph(
            "<i>Hasil ini bukan pengganti diagnosis medis profesional. "
            "Selalu konsultasikan dengan tenaga kesehatan.</i>",
            styles["Normal"]
        ))

        doc.build(story)
        buf.seek(0)
        return send_file(
            buf, as_attachment=True,
            download_name="health_predictor_result.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)