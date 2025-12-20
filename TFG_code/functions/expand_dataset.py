import os
import shutil
import random

dir_imgs = '/datatmp2/joan/tfg_joan/images_to_train/Squat1/images/Train'
dir_labels = '/datatmp2/joan/tfg_joan/images_to_train/Squat1/labels/Train'
out_dir_images_train = '/datatmp2/joan/tfg_joan/exercise_dataset_seg/train/images'
out_dir_labels_train = '/datatmp2/joan/tfg_joan/exercise_dataset_seg/train/labels'
out_dir_images_test = '/datatmp2/joan/tfg_joan/exercise_dataset_seg/test/images'
out_dir_labels_test = '/datatmp2/joan/tfg_joan/exercise_dataset_seg/test/labels'

os.makedirs(out_dir_images_train, exist_ok=True)
os.makedirs(out_dir_labels_train, exist_ok=True)
os.makedirs(out_dir_images_test, exist_ok=True)
os.makedirs(out_dir_labels_test, exist_ok=True)

images = [f for f in os.listdir(dir_imgs) if f.endswith('.jpg')]
labels = [f.replace('.jpg', '.txt') for f in images if os.path.isfile(os.path.join(dir_labels, f.replace('.jpg', '.txt')))]

images_true = []

for image in images:
    label = image.replace('.jpg', '.txt')
    if not os.path.isfile(os.path.join(dir_labels, label)):
        print(f"WARNING: La etiqueta {label} no existe para la imagen {image}. Se omitirá esta imagen.")
    else:
        images_true.append(image)

print('Len images: ', len(images_true))
print('Len labels: ', len(labels))

# Filtrar para mantener solo los pares de imagen y etiqueta que existen ambos
data_pairs = [(img, lbl) for img, lbl in zip(images_true, labels) if os.path.isfile(os.path.join(dir_labels, lbl))]
print('Len data_pairs: ', len(data_pairs))

# Mezclar aleatoriamente los pares de datos
random.shuffle(data_pairs)

# Calcular el número de muestras para test y test
total_data = len(data_pairs)
test_count = int(total_data * 0.15)
train_count = total_data - test_count

# Separar los datos en conjuntos de train, test, y test
train_data = data_pairs[:train_count]
test_data = data_pairs[train_count:train_count + test_count]

# Función para copiar archivos a sus respectivas carpetas
def copy_files(data, img_dest, lbl_dest):
    for img, lbl in data:
        shutil.copy(os.path.join(dir_imgs, img), os.path.join(img_dest, img))
        shutil.copy(os.path.join(dir_labels, lbl), os.path.join(lbl_dest, lbl))

# Copiar archivos a los directorios de train, test y test
copy_files(train_data, out_dir_images_train, out_dir_labels_train)
copy_files(test_data, out_dir_images_test, out_dir_labels_test)

print(fr"¡Dataset distribuido en train ({len(train_data)} imágenes) y test ({len(test_data)} imágenes) exitosamente!")