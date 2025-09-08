from ultralytics import YOLO #type: ignore
import os
import shutil

os.environ["WANDB_DISABLED"] = "true"

#Para cambiar parámetros de entrenamiento hacerlo en la propia función
    
DATASET_PATH = "/datatmp2/joan/tfg_joan/exercise_dataset_seg/exercise_dataset_seg.yaml"

MODEL_PATH = '/datatmp2/joan/tfg_joan/models_exercise_seg/yolo11m-seg.pt'

MODEL_NEW_PATH = '/datatmp2/joan/tfg_joan/models_exercise_seg/YOLO11seg_exercise_batch1.pt'

TMP_DIR = '/datatmp2/joan/tfg_joan/models'

OUT_MODEL_NAME = 'YOLO11seg_exercise_batch1'

def training(dataset_path, model_path, model_new_path, out_dir, out_name):

    model = YOLO(model_path)

    model.train(
        task='seg',
        data=dataset_path,
        project=out_dir, 
        name=out_name,
        save=True, 
        val=True,  
        imgsz=1216, 
        epochs=300, 
        patience=25,
        batch=4,
        lr0=0.001, #Se ajusta automáticamente
        lrf = 0.0003, 
        momentum=0.95, #Se ajusta automáticamente
        weight_decay=0.0001
    
        # fliplr=0.5,  # Flip horizontal
        # flipud=0.5,  # Flip vertical
        # hsv_h=0.035,  # Color (tono)
        # hsv_s=0.7,  # Saturación
        # hsv_v=0.4,  # Valor
        # degrees=5.0,  # Rotación
        # translate=0.1,  # Traslación
        # scale=0.4,  # Escalado
        # shear=0.0,  # Inclinación
        # perspective=0.001,  # Perspectiva
        # mixup= 0.0,  # Mezcla avanzada de imágenes
        # mosaic=0.5  # Activar aumento tipo Mosaic
        )   
    
    tmp_path = os.path.join(out_dir, out_name, 'weights', 'best.pt')

    shutil.copy(tmp_path, model_new_path)

def validate(dataset_path, model_path):
    YOLO(model_path).val(
       data=dataset_path, 
       split="val", 
       task="seg")

training(DATASET_PATH, MODEL_PATH, MODEL_NEW_PATH, TMP_DIR, OUT_MODEL_NAME)

validate(DATASET_PATH, MODEL_NEW_PATH)