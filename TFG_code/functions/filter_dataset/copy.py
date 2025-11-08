import os
import shutil

# Ruta base donde están las subcarpetas
base_dir = "/datatmp2/joan/tfg_joan/LSTM_dataset/train/images/pull_up"

# Carpeta de destino donde se copiarán las primeras imágenes
output_dir = "/datatmp2/joan/tfg_joan/images/pull_up"
os.makedirs(output_dir, exist_ok=True)

# Recorre cada subcarpeta dentro del directorio base
for subdir in sorted(os.listdir(base_dir)):
    subdir_path = os.path.join(base_dir, subdir)

    if os.path.isdir(subdir_path):
        # Obtiene todas las imágenes del subdirectorio
        images = sorted([
            f for f in sorted(os.listdir(subdir_path))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if images:
            first_image = images[0]
            src = os.path.join(subdir_path, first_image)
            dst = os.path.join(output_dir, f"{subdir}_{first_image}")

            shutil.copy2(src, dst)
            print(f"✅ Copiada: {src} → {dst}")
        else:
            print(f"⚠️ No se encontraron imágenes en: {subdir_path}")
