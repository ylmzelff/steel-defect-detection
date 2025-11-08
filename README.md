# Steel Defect Detection MLOps

Bu proje, çelik yüzey defektlerini tespit etmek için YOLOv8 kullanarak makine öğrenmesi modeli geliştirir ve MLOps best practices uygular.

## 🎯 Proje Özeti

- **Veri Seti**: NEU Steel Surface Defect Database
- **Model**: YOLOv8 (Transfer Learning)
- **Defekt Türleri**: 6 sınıf (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches)
- **Pipeline**: Otomatik veri hazırlama, eğitim ve değerlendirme

## 📁 Proje Yapısı

```
steel-defect-detection-mlops/
├── xml_to_yolo.py          # XML annotations → YOLO format dönüşümü
├── split_data.py           # Train/Valid/Test veri bölme
├── train.py                # YOLOv8 model eğitimi
├── neu_defect.yaml         # Dataset konfigürasyonu
└── README.md
```

## 🚀 Kullanım

### 1. Veri Hazırlama
```bash
# XML'leri YOLO formatına dönüştür
python xml_to_yolo.py

# Veriyi train/valid/test olarak böl
python split_data.py
```

### 2. Model Eğitimi
```bash
# YOLOv8 ile eğitim başlat
python train.py
```

## 🔧 Gereksinimler

```bash
pip install ultralytics torch torchvision
```

## 📊 Veri Seti İstatistikleri

- **Toplam Görüntü**: 1800
- **Train**: 1259 (%70)
- **Validation**: 359 (%20)  
- **Test**: 181 (%10)

## 🏷️ Sınıflar

| ID | Sınıf | Açıklama |
|---|---|---|
| 0 | crazing | Çatlak benzeri defekt |
| 1 | inclusion | İç heterojenlik |
| 2 | patches | Yama şeklinde defekt |
| 3 | pitted_surface | Çukurlu yüzey |
| 4 | rolled-in_scale | Hadde izi |
| 5 | scratches | Çizikler |

## 📈 MLOps Özellikleri

- ✅ Otomatik veri doğrulama
- ✅ Reproducible training (SEED=42)
- ✅ Error handling ve logging
- ✅ GPU/CPU otomatik algılama
- ✅ Model versiyonlama

## 🤝 Katkıda Bulunma

Pull request'ler hoş geldinir. Büyük değişiklikler için önce issue açınız.

## 📄 Lisans

MIT License