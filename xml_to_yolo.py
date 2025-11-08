import xml.etree.ElementTree as ET
import os
import glob
import argparse

# 1. Adım: Sınıf listemizi tanımlıyoruz
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 2. Adım: Matematiksel dönüşümü yapacak fonksiyon
def convert_voc_to_yolo(image_width, image_height, box):
    """
    PASCAL VOC formatındaki (xmin, ymin, xmax, ymax) koordinatlarını
    YOLO formatındaki (x_center_norm, y_center_norm, w_norm, h_norm) koordinatlarına dönüştürür.

    Args:
        image_width (int): görüntü genişliği piksel
        image_height (int): görüntü yüksekliği piksel
        box (tuple): (xmin, ymin, xmax, ymax)

    Returns:
        tuple: (x_center_norm, y_center_norm, w_norm, h_norm)
    """
    # Mutlak piksel değerlerini al
    xmin, ymin, xmax, ymax = box
    
    # Kutu genişliği ve yüksekliği (piksel)
    box_w = float(xmax) - float(xmin)
    box_h = float(ymax) - float(ymin)
    
    # Kutu merkez koordinatları (piksel)
    x_center = float(xmin) + (box_w / 2)
    y_center = float(ymin) + (box_h / 2)
    
    # Normalizasyon (0-1 arası)
    x_center_norm = x_center / float(image_width)
    y_center_norm = y_center / float(image_height)
    w_norm = box_w / float(image_width)
    h_norm = box_h / float(image_height)
    
    return (x_center_norm, y_center_norm, w_norm, h_norm)


# 3. Adım: Ana dönüşüm süreci
def process_annotations(xml_dir, output_dir):
    """
    xml_dir içindeki tüm .xml dosyalarını okur, dönüştürür
    ve output_dir içine .txt olarak kaydeder.
    """
    # Çıktı klasörü yoksa oluştur
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # YENİ KOD:
    # XML klasöründeki ve TÜM ALT KLASÖRLERİNDEKİ .xml dosyalarını bul
    xml_files = glob.glob(os.path.join(xml_dir, "**", '*.xml'), recursive=True)

    if not xml_files:
        print(f"Uyarı: '{xml_dir}' klasöründe hiç .xml dosyası bulunamadı.")
        return

    print(f"Toplam {len(xml_files)} adet XML dosyası işleniyor: {xml_dir}")

    for xml_file in xml_files:
        try:
            # XML dosyasını parse et
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"Hata: '{xml_file}' parse edilemedi: {e}")
            continue

        # Görüntü boyutlarını al
        size = root.find('size')
        if size is None:
            print(f"Uyarı: '{xml_file}' içinde 'size' bilgisi yok. Atlanıyor.")
            continue

        try:
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)
        except Exception as e:
            print(f"Uyarı: '{xml_file}' içinde geçersiz 'size' bilgisi: {e}. Atlanıyor.")
            continue

        # Çıktı .txt dosyasının adını belirle
        base_filename = os.path.basename(xml_file)
        txt_filename = os.path.splitext(base_filename)[0] + '.txt'
        txt_filepath = os.path.join(output_dir, txt_filename)

        # .txt dosyasını yazma modunda aç
        with open(txt_filepath, 'w', encoding='utf-8') as f_out:
            # Resimdeki her bir 'object' (hata) için döngü
            for obj in root.iter('object'):
                class_name = obj.find('name').text

                # Sınıf adının bizim listemizde olduğundan emin ol
                if class_name not in CLASSES:
                    print(f"Uyarı: Bilinmeyen sınıf '{class_name}' - {xml_file} dosyasında bulundu. Atlanıyor.")
                    continue

                # Sınıf ID'sini al (listenin indeksi)
                class_id = CLASSES.index(class_name)

                # Kutu koordinatlarını al
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    print(f"Uyarı: '{xml_file}' içinde 'bndbox' yok. Atlanıyor.")
                    continue

                try:
                    box = (
                        float(bndbox.find('xmin').text),
                        float(bndbox.find('ymin').text),
                        float(bndbox.find('xmax').text),
                        float(bndbox.find('ymax').text)
                    )
                except Exception as e:
                    print(f"Uyarı: '{xml_file}' içinde geçersiz bndbox: {e}. Atlanıyor.")
                    continue

                # Dönüşümü yap
                yolo_box = convert_voc_to_yolo(img_width, img_height, box)

                # .txt dosyasına YOLO formatında yaz
                # format: class_id x_center y_center width height
                # floatları sabit basmak için format kullanıyoruz
                f_out.write(f"{class_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}\n")

    print(f"Dönüştürme tamamlandı. Dosyalar '{output_dir}' klasörüne kaydedildi.")

# --- Script'i Çalıştırma ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PASCAL VOC XML to YOLO txt labels for train/validation folders.')
    parser.add_argument('--root', type=str, default='NEU-DET', help='Root dataset folder that contains train/ and validation/ (default: NEU-DET)')
    parser.add_argument('--out', type=str, default='labels', help='Output labels base folder (default: labels)')
    parser.add_argument('--subsets', type=str, default='train,validation', help='Comma-separated subsets to process (default: train,validation)')
    args = parser.parse_args()

    root = args.root
    out_base = args.out
    subsets = [s.strip() for s in args.subsets.split(',') if s.strip()]

    any_processed = False
    for subset in subsets:
        xml_annotations_dir = os.path.join(root, subset, 'annotations')
        output_dir = os.path.join(out_base, subset)

        if not os.path.isdir(xml_annotations_dir):
            print(f"Bilgi: '{xml_annotations_dir}' bulunamadı. Atlanıyor.")
            continue

        print(f"İşleniyor: {xml_annotations_dir} -> {output_dir}")
        process_annotations(xml_annotations_dir, output_dir)
        any_processed = True

    if not any_processed:
        print("Hiçbir klasör işlenmedi. Lütfen --root ve --subsets parametrelerini kontrol edin veya gerekli klasörleri oluşturun.")