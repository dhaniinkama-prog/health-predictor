# 🩺 HealthAI Predictor

Website prediksi kesehatan berbasis Machine Learning (KNN) untuk deteksi **Stunting Balita** dan **Diabetes**.

---

## 📂 Struktur Folder

```
health-predictor/
├── app.py                  ← Backend Flask utama
├── requirements.txt        ← Dependensi Python
├── Procfile                ← Untuk Render.com
├── vercel.json             ← Untuk Vercel (opsional)
├── .gitignore
├── models/                 ← 🔴 Taruh file .pkl di sini!
│   ├── model_knn_stunting.pkl
│   ├── scaler.pkl
│   ├── model_knn_diabetes.pkl
│   ├── scaler_1.pkl
│   └── le.pkl
└── templates/
    └── index.html          ← Frontend (auto-served oleh Flask)
```

---

## ⚙️ File PKL yang Dibutuhkan

Taruh semua file `.pkl` ke folder `models/`:

| File | Fungsi |
|------|--------|
| `model_knn_stunting.pkl` | Model KNN untuk prediksi stunting |
| `scaler.pkl` | Scaler untuk fitur stunting |
| `model_knn_diabetes.pkl` | Model KNN untuk prediksi diabetes |
| `scaler_1.pkl` | Scaler untuk fitur diabetes |
| `le.pkl` | Label Encoder untuk output diabetes |

> ⚠️ Tanpa file PKL, app tetap berjalan dalam **mode demo** (fallback logika manual).

---

## 🚀 Cara Deploy ke Render.com (PALING MUDAH)

### Langkah 1 — Upload ke GitHub
```bash
# Buka terminal/command prompt di folder health-predictor
git init
git add .
git commit -m "Initial commit - HealthAI Predictor"

# Buat repo di github.com lalu:
git remote add origin https://github.com/USERNAME/health-predictor.git
git push -u origin main
```

### Langkah 2 — Deploy di Render.com
1. Buka **https://render.com** → Login/Daftar gratis
2. Klik **"New +"** → pilih **"Web Service"**
3. Hubungkan akun GitHub kamu
4. Pilih repo `health-predictor`
5. Isi pengaturan:
   - **Name**: `health-predictor` (bebas)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
6. Klik **"Create Web Service"**
7. Tunggu ±3 menit → dapat link publik seperti `https://health-predictor.onrender.com`

---

## 🌐 Alternatif: Deploy ke Vercel

```bash
npm install -g vercel
cd health-predictor
vercel
```
> ⚠️ Vercel kurang optimal untuk Flask dengan file pkl besar. Gunakan Render untuk Flask.

---

## 💻 Jalankan Lokal (Testing)

```bash
# Install dependensi
pip install -r requirements.txt

# Jalankan
python app.py

# Buka browser
# http://localhost:5000
```

---

## 🔧 Teknologi

- **Backend**: Python Flask + scikit-learn (KNN)
- **Frontend**: HTML5 + Tailwind CSS + Chart.js + Lucide Icons
- **Design**: Glassmorphism + Dark Mode + Gauge Animation
- **Export**: jsPDF (client-side PDF download)

---

## ⚠️ Disclaimer

Hasil prediksi ini bersifat indikatif dan **tidak menggantikan diagnosis medis profesional**.
Selalu konsultasikan dengan dokter atau tenaga kesehatan untuk diagnosis yang tepat.
