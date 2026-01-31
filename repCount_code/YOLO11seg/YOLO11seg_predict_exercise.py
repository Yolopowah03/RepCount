import os
from ultralytics import YOLO  # type: ignore
import cv2  # type: ignore
import random
import numpy as np
import xml.etree.ElementTree as ET
import time

CONF = 0.2
MODEL_PATH = '/datatmp2/joan/repCount/models_exercise_seg/YOLO11seg_exercise_batch1.pt'
IMAGE_DIR = '/datatmp2/joan/repCount/LSTM_dataset/test/images/bench_press/bench_press_2'
OUT_IMAGE_PATH = '/datatmp2/joan/repCount/results/seg/test_bench_press2'
OUTPUT_XML = '/datatmp2/joan/repCount/results/seg/test_bench_press2/test_bench_press_seg.xml'

os.makedirs(OUT_IMAGE_PATH, exist_ok=True)

model = YOLO(MODEL_PATH)

image_files = []
for root, _, files in sorted(os.walk(IMAGE_DIR), reverse=True):
    for f in sorted(files, reverse=True):
        if f.lower().endswith('.jpg'):
            image_files.append(os.path.join(root, f))

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

def mask_to_rle(submask: np.ndarray) -> str:
    pixels = submask.flatten(order='C')
    rle = []
    prev = pixels[0]
    count = 1
    for pix in pixels[1:]:
        if pix == prev:
            count += 1
        else:
            rle.append(count)
            count = 1
            prev = pix
    rle.append(count)
    if pixels[0] == 1:
        rle = [0] + rle
    return ', '.join(str(x) for x in rle)

with open(OUTPUT_XML, 'w', encoding='utf-8') as xml_file:
    xml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    xml_file.write('<annotation>\n')
    
    for img_id, img_path in enumerate(image_files):
        time1 = time.time()
        save_path = os.path.join(OUT_IMAGE_PATH, os.path.basename(img_path))
        # if os.path.exists(save_path):
        #     print(f"WARNING: {save_path} ya existe. Saltando.")
        #     continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"ERROR: No se pudo leer {img_path}")
            continue
        h, w = img.shape[:2]

        results = model(img_path, conf=CONF)
        for result in results:
            if result.masks is None or len(result.masks) == 0:
                print(f"WARNING: No se encontraron máscaras en {img_path}")
                cv2.imwrite(save_path, img)
                continue

            img_elem = ET.Element('image', {
                'id': str(img_id),
                'name': os.path.basename(img_path),
                'width': str(w),
                'height': str(h)
            })

            masks = result.masks.data
            for mask in masks:
                mask_np = mask.cpu().numpy()
                m_uint8 = (mask_np.astype(np.uint8) * 255)
                
                # Bounding box
                ys, xs = np.where(m_uint8 > 0)
                if ys.size == 0:
                    continue
                top, left = int(ys.min()), int(xs.min())
                height_bb = int(ys.max() - ys.min() + 1)
                width_bb = int(xs.max() - xs.min() + 1)
                submask = mask_np[top:top+height_bb, left:left+width_bb].astype(np.uint8)
                
                rle_str = mask_to_rle(submask)
                ET.SubElement(img_elem, 'mask', {
                    'label': 'contenedor',
                    'source': 'manual',
                    'occluded': '0',
                    'rle': rle_str,
                    'left': str(left),
                    'top': str(top),
                    'width': str(width_bb),
                    'height': str(height_bb),
                    'z_order': '0'
                })

                mask_bool = cv2.resize(m_uint8, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                color = random_color()
                alpha = 0.5
                for c in range(3):
                    img[:, :, c] = np.where(
                        mask_bool,
                        (img[:, :, c] * (1 - alpha) + alpha * color[c]).astype(np.uint8),
                        img[:, :, c]
                    )

            xml_str = ET.tostring(img_elem, encoding='utf-8').decode('utf-8')
            xml_file.write(xml_str + '\n')
            
            cv2.imwrite(save_path, img)
            print(f"[+] Guardada imagen con máscaras: {save_path}")

            time2 = time.time()
            print(f"[+] Tiempo de inferencia {img_path}: {time2 - time1:.2f}s")

            del img_elem

    xml_file.write('</annotation>\n')

print(f"[+] XML incremental escrito en: {OUTPUT_XML}")
