import os
import glob
import random
import shutil

# --- Ayarlar ---
SOURCE_DIRS = {
    'train': {'images': 'NEU-DET/train/images', 'labels': 'labels/train'},
    'validation': {'images': 'NEU-DET/validation/images', 'labels': 'labels/validation'}
}
TARGET_BASE_DIR = "dataset"

# Bölme oranları
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
TEST_RATIO = 0.1  # (Toplamı 1.0 olmalı)

# MLOps için kritik: Tekrar üretilebilirlik (Reproducibility)
# Bu sayıyı sabit tuttuğumuz sürece, script her çalıştığında aynı dosyaları
# train/valid/test'e atacaktır.
RANDOM_SEED = 42 
random.seed(RANDOM_SEED)

# --- 1. Çıktı Klasör Yapısını Oluşturma ---
def create_dirs():
    """
    dataset/images/[train,valid,test] ve dataset/labels/[train,valid,test]
    klasör yapılarını oluşturur.
    """
    print(f"'{TARGET_BASE_DIR}' içinde hedef klasör yapısı oluşturuluyor...")
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(TARGET_BASE_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(TARGET_BASE_DIR, 'labels', split), exist_ok=True)
    print("Klasör yapısı hazır.")

# --- 2. Veri Bölme ve Kopyalama ---
def split_and_copy_files():
    """
    NEU-DET/train ve NEU-DET/validation klasörlerindeki dosyaları okur, karıştırır ve 
    hem resimleri hem de ilgili etiketleri hedef klasörlere kopyalar.
    """
    print("Kaynak veriler okunuyor...")
    
    # Tüm resim dosyalarını topla (train ve validation'dan)
    all_image_data = []  # (image_path, label_path, source_subset) tuples
    
    for subset, dirs in SOURCE_DIRS.items():
        image_dir = dirs['images']
        label_dir = dirs['labels']
        
        print(f"  {subset}: {image_dir}")
        
        # Bu subset'teki tüm resim dosyalarını bul (defekt türü alt klasörlerinde)
        image_paths = glob.glob(os.path.join(image_dir, "*", "*.jpg"))
        
        for image_path in image_paths:
            # Resim dosyasının adını al (örn: crazing_1.jpg)
            image_filename = os.path.basename(image_path)
            
            # Etiket dosyasının adını türet (örn: crazing_1.txt)
            label_filename = os.path.splitext(image_filename)[0] + ".txt"
            label_path = os.path.join(label_dir, label_filename)
            
            # Etiket dosyası varsa listeye ekle
            if os.path.exists(label_path):
                all_image_data.append((image_path, label_path, subset))
            else:
                print(f"UYARI: {image_path} için etiket dosyası ({label_path}) bulunamadı. Bu dosya atlanıyor.")
    
    if not all_image_data:
        print("HATA: Hiç resim-etiket çifti bulunamadı.")
        print("Lütfen önce xml_to_yolo.py script'ini çalıştırarak etiketleri oluşturun.")
        return

    print(f"Toplam {len(all_image_data)} adet resim-etiket çifti bulundu.")
    
    # Listeyi karıştır (RANDOM_SEED sayesinde hep aynı sıra)
    random.shuffle(all_image_data)
    
    # Veri setlerini bölmek için indeksleri hesapla
    total_count = len(all_image_data)
    train_count = int(total_count * TRAIN_RATIO)
    valid_count = int(total_count * VALID_RATIO)
    # Geriye kalanlar test seti olacak
    
    splits = {
        'train': all_image_data[:train_count],
        'valid': all_image_data[train_count : train_count + valid_count],
        'test': all_image_data[train_count + valid_count :]
    }
    
    print(f"Veri bölündü: Train={len(splits['train'])}, Valid={len(splits['valid'])}, Test={len(splits['test'])}")

    # --- 3. Dosyaları Kopyalama ---
    print("Dosyalar ilgili klasörlere kopyalanıyor...")
    copied_files = 0
    for split_name, file_data in splits.items():
        for image_path, label_path, source_subset in file_data:
            # 1. Resim dosyasının adını al (örn: crazing_1.jpg)
            image_filename = os.path.basename(image_path)
            label_filename = os.path.basename(label_path)
            
            # 2. Hedef yolları belirle
            target_image_path = os.path.join(TARGET_BASE_DIR, 'images', split_name, image_filename)
            target_label_path = os.path.join(TARGET_BASE_DIR, 'labels', split_name, label_filename)
            
            # 3. Kopyalama işlemi
            shutil.copy(image_path, target_image_path)
            shutil.copy(label_path, target_label_path)
            copied_files += 1

    print(f"Veri bölme ve kopyalama işlemi tamamlandı. Toplam {copied_files} resim ve etiket kopyalandı.")

# --- Ana Script ---
if __name__ == "__main__":
    # Önce klasörleri oluştur
    create_dirs()
    # Sonra dosyaları böl ve kopyala
    split_and_copy_files()