# build_database.py
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tqdm import tqdm

# --- Konfigurasi Path ---
DATASET_PATH = 'dataset'
IMAGE_SIZE = (224, 224)
ASSETS_PATH = 'assets'
FEATURES_FILE = os.path.join(ASSETS_PATH, 'features.npy')
PATHS_FILE = os.path.join(ASSETS_PATH, 'image_paths.npy')

# Pastikan folder 'assets' ada
os.makedirs(ASSETS_PATH, exist_ok=True)

def load_model():
    """Memuat model ResNet50 pre-trained."""
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.trainable = False
    return model

def extract_feature(img_path, model):
    """Mengekstrak satu feature vector dari sebuah gambar."""
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    feature = model.predict(preprocessed_img, verbose=0)
    return feature.flatten()

# --- Main Execution ---
if __name__ == "__main__":
    print("Memulai proses pembangunan aset (features.npy & image_paths.npy)...")

    # 1. Muat Model
    model = load_model()
    print("Model ResNet50 berhasil dimuat.")

    # 2. Temukan SEMUA gambar (JPG, JPEG, PNG)
    print(f"Mencari semua file gambar di {DATASET_PATH}...")
    extensions = ('*.jpg', '*.jpeg', '*.png')
    all_image_paths = []
    for ext in extensions:
        all_image_paths.extend(glob.glob(os.path.join(DATASET_PATH, '**', ext), recursive=True))

    print(f"Total ditemukan {len(all_image_paths)} gambar.")

    # 3. Loop dan Ekstrak Fitur
    all_features = []

    for img_path in tqdm(all_image_paths, desc="Mengekstrak Fitur"):
        try:
            feature = extract_feature(img_path, model)
            all_features.append(feature)
        except Exception as e:
            print(f"\nError memproses {img_path}: {e}")
            all_image_paths.remove(img_path) # Hapus path jika gambar error

    # 4. Simpan hasilnya ke folder 'assets'
    np.save(FEATURES_FILE, np.array(all_features))
    np.save(PATHS_FILE, np.array(all_image_paths))

    print("\n Pembangunan aset selesai.")
    print(f"Fitur disimpan di: '{FEATURES_FILE}'")
    print(f"Path gambar disimpan di: '{PATHS_FILE}'")