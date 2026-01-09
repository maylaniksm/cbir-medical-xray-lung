import os
import shutil
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# 1. KONFIGURASI DASAR
# ==========================================================
app = Flask(__name__)

STATIC_PATH = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_PATH, 'uploads')
RETRIEVED_FOLDER = os.path.join(STATIC_PATH, 'retrieved')
ASSETS_PATH = 'assets'
FEATURES_FILE = os.path.join(ASSETS_PATH, 'features.npy')
PATHS_FILE = os.path.join(ASSETS_PATH, 'filenames.npy')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRIEVED_FOLDER, exist_ok=True)
os.makedirs(ASSETS_PATH, exist_ok=True)

# ==========================================================
# 2. MODEL EKSTRAKTOR FITUR (PAKAI PYTORCH)
# ==========================================================
def load_feature_extractor_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])  # hapus layer klasifikasi terakhir
    model.eval()
    return model

def extract_feature(image_path, model, transform):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor).squeeze().numpy()
    return features

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================================
# 3. MUAT MODEL DAN DATABASE FITUR
# ==========================================================
print("🔹 Memuat model ResNet18 (PyTorch)...")
model = load_feature_extractor_model()
print("✅ Model berhasil dimuat.")

try:
    all_features = np.load(FEATURES_FILE)
    all_paths = np.load(PATHS_FILE)
    print(f"✅ Database fitur dimuat ({len(all_paths)} gambar).")
except Exception as e:
    print(f"⚠️ Tidak ditemukan database fitur: {e}")
    all_features = np.array([])
    all_paths = np.array([])

# ==========================================================
# 4. ROUTE UTAMA
# ==========================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file diunggah.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Nama file kosong.')

    if file and all_features.any():
        filename = file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        # Ekstraksi fitur dari gambar query
        query_feature = extract_feature(upload_path, model, transform)
        similarities = cosine_similarity([query_feature], all_features)[0]
        top_indices = np.argsort(similarities)[::-1][:9]

        results = []

        # Ambil gambar hasil retrieval
        for i in top_indices:
            img_path = str(all_paths[i])

            # Jika path tidak valid, perbaiki
            if not os.path.exists(img_path):
                img_name = os.path.basename(img_path)
                # coba cari di beberapa folder dataset umum
                for folder in ['NORMAL', 'PNEUMONIA', 'COVID19']:
                    possible_path = os.path.join('dataset', folder, img_name)
                    if os.path.exists(possible_path):
                        img_path = possible_path
                        break
                else:
                    print(f"⚠️ File tidak ditemukan: {img_path}")
                    continue

            # Salin hasil ke static/retrieved/
            dest_path = os.path.join(RETRIEVED_FOLDER, os.path.basename(img_path))
            shutil.copy(img_path, dest_path)

            results.append({
                'path': f"/static/retrieved/{os.path.basename(img_path)}",
                'score': round(similarities[i], 3)
            })

        # Ambil hasil terbaik (skor tertinggi)
        best_result_index = np.argmax(similarities)
        best_image_path = str(all_paths[best_result_index])

        # Kelas diambil dari nama folder induk (NORMAL / PNEUMONIA / COVID19)
        predicted_class = "Tidak diketahui"
        if os.sep in best_image_path:
            predicted_class = best_image_path.split(os.sep)[-2]

        return render_template('results.html',
                               query_image=f"/static/uploads/{filename}",
                               results=results,
                               predicted_class=predicted_class)

    return render_template('index.html', error='Database fitur belum dimuat.')

# ==========================================================
# 5. JALANKAN SERVER
# ==========================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Menjalankan Server Flask Image Retrieval (PyTorch)")
    print("🌐 Akses di: http://127.0.0.1:5000/")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)
