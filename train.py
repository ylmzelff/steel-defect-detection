from ultralytics import YOLO
import os
import torch

def main():
    # --- 1. Model Seçimi ---
    # 'yolov8n.pt', en hızlı ve en küçük modeldir. 
    # '.pt' uzantısı, COCO veri setinde önceden eğitilmiş (pre-trained)
    # ağırlıkları yükleyeceğimizi belirtir. Bu, 'transfer learning' yapmamızı sağlar.
    model_name = 'yolov8n.pt'
    
    # --- 2. Eğitim Parametreleri ---
    # YAML dosyamızın adı (Sizin oluşturduğunuz isim) - DÜZELTİLDİ
    data_config_path = 'neu_defect.yaml'  # defect.yaml -> neu_defect.yaml
    
    # Epoch: Veri setinin tamamının kaç kez "görüleceği"
    epochs = 50 
    
    # Batch Size: GPU'nuza (RTX 3050) tek seferde kaç resim gönderileceği.
    # Hafıza hatası (Out of Memory) alırsanız bu sayıyı 8'e düşürün.
    batch_size = 16 
    
    # Görüntü Boyutu: Resimler eğitimden önce bu boyuta getirilecek.
    img_size = 640 
    
    # Cihaz: '0' -> ilk GPU'yu (RTX 3050'niz) kullan demek.
    # 'cpu' yazarsanız CPU'da eğitir (çok yavaş olur).
    device = 0 
    
    # Kayıt Adı: Eğitim sonuçları bu isimde bir klasöre kaydedilecek.
    run_name = 'steel_defect_run_1'

    # --- 3. GPU Kontrolü ---
    print("=== GPU ve Sistem Kontrolü ===")
    if torch.cuda.is_available():
        print(f"✅ CUDA (GPU) bulundu: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Belleği: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠️  UYARI: CUDA (GPU) bulunamadı. Eğitim CPU üzerinde başlayacak, bu çok yavaş olabilir.")
        device = 'cpu'
        batch_size = 8  # CPU için daha küçük batch size

    # --- 4. YAML Dosyası Kontrolü ---
    if not os.path.exists(data_config_path):
        print(f"❌ HATA: '{data_config_path}' dosyası bulunamadı!")
        print("Lütfen dosya adını kontrol edin.")
        return
    
    print(f"✅ YAML config dosyası bulundu: {data_config_path}")

    # --- 5. Modeli Yükleme ---
    print(f"\n=== Model Yükleme ===")
    print(f"Model yükleniyor: {model_name}")
    model = YOLO(model_name)

    # --- 6. Eğitim Başlatma ---
    print(f"\n=== Eğitim Başlatılıyor ===")
    print(f"Dataset: {data_config_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"Run Name: {run_name}")
    
    try:
        results = model.train(
            data=data_config_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            name=run_name,
            patience=10,  # 10 epoch boyunca gelişme olmazsa eğitimi durdur (overfitting önlemi)
            save=True,    # Model ağırlıklarını kaydet
            plots=True,   # Eğitim grafikleri oluştur
            verbose=True  # Detaylı çıktı
        )
        
        print(f"\n=== Eğitim Tamamlandı! ===")
        print(f"✅ Sonuçlar: 'runs/detect/{run_name}' klasörüne kaydedildi.")
        print(f"✅ En iyi model: 'runs/detect/{run_name}/weights/best.pt'")
        print(f"✅ Son model: 'runs/detect/{run_name}/weights/last.pt'")
        
    except Exception as e:
        print(f"❌ Eğitim sırasında hata oluştu: {e}")
        print("💡 Batch size'ı azaltmayı deneyin (8 veya 4)")

if __name__ == '__main__':
    main()