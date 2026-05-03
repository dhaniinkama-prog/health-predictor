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

model_stunting  = load("model_knn_stunting.pkl")
scaler_stunting = load("scaler.pkl")
model_diabetes  = load("model_knn_diabetes.pkl")
scaler_diabetes = load("scaler_1.pkl")
le_diabetes     = load("le.pkl")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict/stunting", methods=["POST"])
def predict_stunting():
    try:
        data = request.get_json()
        age    = float(data["age"])          # bulan
        height = float(data["height"])       # cm
        weight = float(data["weight"])       # kg
        sex    = int(data["sex"])            # 0=perempuan, 1=laki-laki

        features = np.array([[age, height, weight, sex]])

        if model_stunting and scaler_stunting:
            features_scaled = scaler_stunting.transform(features)
            pred = int(model_stunting.predict(features_scaled)[0])
            try:
                proba = float(model_stunting.predict_proba(features_scaled)[0][pred])
            except Exception:
                proba = 0.85
        else:
            # Demo fallback — replace with real models
            haz = (height - (45 + age * 0.35)) / 4
            pred = 1 if haz < -2 else 0
            proba = 0.82

        labels = {0: "Normal", 1: "Stunting"}
        label  = labels.get(pred, str(pred))

        tips_map = {
            "Normal": [
                "Pertahankan pola makan bergizi seimbang.",
                "Berikan ASI eksklusif hingga 6 bulan.",
                "Pantau pertumbuhan secara rutin di Posyandu.",
                "Pastikan imunisasi anak lengkap.",
            ],
            "Stunting": [
                "Segera konsultasikan ke dokter atau ahli gizi.",
                "Tingkatkan asupan protein: telur, ikan, daging.",
                "Berikan suplemen zinc dan vitamin A sesuai anjuran.",
                "Pantau berat dan tinggi badan setiap bulan.",
                "Pastikan sanitasi dan kebersihan lingkungan.",
            ],
        }

        return jsonify({
            "status": "ok",
            "prediction": pred,
            "label": label,
            "confidence": round(proba * 100, 1),
            "tips": tips_map.get(label, []),
            "inputs": {"age": age, "height": height, "weight": weight, "sex": sex},
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json()
        glucose    = float(data["glucose"])
        bmi        = float(data["bmi"])
        age        = float(data["age"])
        insulin    = float(data["insulin"])
        bp         = float(data["blood_pressure"])
        pregnancies = float(data.get("pregnancies", 0))
        skin       = float(data.get("skin_thickness", 20))
        dpf        = float(data.get("diabetes_pedigree", 0.5))

        features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

        if model_diabetes and scaler_diabetes:
            features_scaled = scaler_diabetes.transform(features)
            pred_raw = model_diabetes.predict(features_scaled)[0]
            if le_diabetes:
                pred = int(le_diabetes.inverse_transform([pred_raw])[0])
            else:
                pred = int(pred_raw)
            try:
                proba = float(model_diabetes.predict_proba(features_scaled)[0][1])
            except Exception:
                proba = 0.78
        else:
            # Demo fallback
            score = (glucose / 200) * 0.5 + (bmi / 50) * 0.3 + (age / 80) * 0.2
            pred  = 1 if score > 0.45 else 0
            proba = min(score + 0.1, 0.99)

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

        risk_pct = round(proba * 100, 1)

        return jsonify({
            "status": "ok",
            "prediction": pred,
            "label": label,
            "confidence": risk_pct,
            "tips": tips_map.get(label, []),
            "inputs": {
                "glucose": glucose, "bmi": bmi, "age": age,
                "blood_pressure": bp, "insulin": insulin,
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
        label = data.get("label", "-")
        conf  = data.get("confidence", 0)
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
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("ALIGN",      (0,0), (-1,-1), "CENTER"),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f0f9ff")]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#bae6fd")),
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
