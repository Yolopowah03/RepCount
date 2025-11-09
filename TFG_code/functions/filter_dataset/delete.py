import os
import re
import shutil

# Directorios
dir1 = "/datatmp2/joan/tfg_joan/dataset/squat"
dir2 = "/datatmp2/joan/tfg_joan/LSTM_dataset/train/images/squat"

# Extraer todos los números de 3 dígitos de los archivos en dir1
pattern = re.compile(r"\d{3}")
nums_dir1 = set()

for fname in sorted(os.listdir(dir1)):
    match = pattern.findall(fname)
    nums_dir1.update(match)

print(f"📋 Números de 3 dígitos encontrados en dir1: {sorted(nums_dir1)}")

# Recorremos los subdirectorios de dir2
for subdir in sorted(os.listdir(dir2)):
    subdir_path = os.path.join(dir2, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # Buscar si el nombre del subdir contiene alguno de los números detectados
    if any(num in subdir for num in nums_dir1):
        print(f"✅ Conservando {subdir_path} (coincide con número de dir1)")
    else:
        print(f"🗑️ Eliminando {subdir_path} (sin coincidencias)")
        # shutil.rmtree(subdir_path)

# for file in sorted(os.listdir(dir2)):
#     file_path = os.path.join(dir2, file)
#     if os.path.isfile(file_path):
#         match = pattern.findall(file)
#         if match and any(num in nums_dir1 for num in match):
#             print(f"✅ Conservando {file_path} (coincide con número de dir1)")
#         else:
#             print(f"🗑️ Eliminando {file_path} (sin coincidencias)")
#             os.remove(file_path)