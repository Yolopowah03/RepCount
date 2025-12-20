from huggingface_hub import hf_hub_download # type: ignore
import os
import shutil

repo_id = "noahcao/sapiens-pose-coco"
filename = "sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2"
output_dir = "/datatmp2/joan/tfg_joan/sapiens/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b"

# --- CAMBIO IMPORTANTE ---

# 1. Descarga el archivo REAL a la ubicación de la CACHÉ por defecto de Hugging Face.
#    El path_in_cache será la ruta al archivo real.
path_in_cache = hf_hub_download(repo_id=repo_id, filename=filename)

# 2. Define la ruta final deseada
final_path = os.path.join(output_dir, os.path.basename(filename))

# 3. Mueve (o copia) el archivo REAL desde la caché a la ubicación deseada
try:
    # Asegúrate de que el directorio de destino exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Mueve el archivo. shutil.move también funciona si está en el mismo sistema de archivos.
    # Usaremos copy para ser más seguros y mantener el archivo en la caché.
    shutil.copy2(path_in_cache, final_path)
    
    print(f"Archivo copiado exitosamente a: {final_path}")
    print(f"El archivo original REAL en caché es: {path_in_cache}")
    
except Exception as e:
    print(f"Error al mover/copiar el archivo: {e}")