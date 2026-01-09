import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# ==========================================================
# 1. KONFIGURASI DASAR
# ==========================================================
DATASET_PATH = "dataset"          # Folder dataset utama
ASSETS_PATH = "assets"            # Folder penyimpanan hasil ekstraksi fitur
FEATURES_FILE = os.path.join(ASSETS_PATH, "features.npy")
PATHS_FILE = os.path.join(ASSETS_PATH, "filenames.npy")

os.makedirs(ASSETS_PATH, exist_ok=True)

# ==========================================================
# 2. MODEL RESNET18 EKSTRAKTOR FITUR
# ==========================================================
def load_feature_extractor():
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # buang FC layer
    model.eval()
    return model

def extract_feature(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = model(tensor).squeeze().numpy()
        return feature
    except Exception as e:
        print(f"⚠️ Error pada {image_path}: {e}")
        return None

# ==========================================================
# 3. TRANSFORMASI GAMBAR
# ==========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ==========================================================
# 4. FUNGSI UTAMA EKSTRAKSI
# ==========================================================
def process_dataset():
    model = load_feature_extractor()
    print("✅ Model ResNet18 siap digunakan.\n")

    # Cari semua gambar dari subfolder: COVID19/, NORMAL/, PNEUMONIA/
    image_files = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("❌ Tidak ada gambar ditemukan di folder dataset/")
        return

    print(f"📁 Menemukan total {len(image_files)} gambar. Memulai ekstraksi...\n")

    features = []
    paths = []

    for img_path in tqdm(image_files, desc="Ekstraksi Fitur"):
        f = extract_feature(img_path, model, transform)
        if f is not None:
            features.append(f)
            paths.append(img_path)

    # Simpan hasil
    np.save(FEATURES_FILE, np.array(features))
    np.save(PATHS_FILE, np.array(paths))

    print("\n✅ Ekstraksi selesai.")
    print(f"📦 features.npy → {FEATURES_FILE}")
    print(f"📂 filenames.npy → {PATHS_FILE}")
    print(f"📊 Total fitur tersimpan: {len(paths)}")

# ==========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MEMULAI EKSTRAKSI FITUR (PYTORCH)")
    print("=" * 60)
    process_dataset()
