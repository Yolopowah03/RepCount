import os
import cv2 as cv  # type: ignore
from natsort import natsorted  # type: ignore

IMAGE_FOLDER = '/datatmp2/joan/SWIR_joan/results/YOLO11_obb/images_simulation_booth_conference'
OUTPUT_VIDEO = '/datatmp2/joan/SWIR_joan/results/simulation_booth_conference_YOLO11.avi'
FPS = 25

def images_to_video(image_dir: str, output_path: str, fps: float = 30.0):
    # 1. Listar y ordenar las imágenes
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]
    images = natsorted(images) 

    if not images:
        raise ValueError(f"No se encontraron imágenes en {image_dir!r}")

    # 2. Leer la primera imagen para saber resolución
    first_frame = cv.imread(os.path.join(image_dir, images[0]))
    if first_frame is None:
        raise RuntimeError(f"No se pudo leer la imagen {images[0]!r}")
    height, width = first_frame.shape[:2]

    # 3. Crear el VideoWriter
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # o 'MJPG', 'X264'...
    writer = cv.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"No se pudo crear el vídeo de salida {output_path!r}")

    print(f"Creando vídeo {output_path!r} a {fps} fps, resolución {width}×{height}")

    # 4. Iterar y escribir cada frame
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        frame = cv.imread(img_path)
        if frame is None:
            print(f"Advertencia: no pude leer {img_name}, lo salto.")
            continue
        # Asegurar tamaño idéntico
        h, w = frame.shape[:2]
        if (w, h) != (width, height):
            frame = cv.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print("Vídeo generado con éxito.")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    images_to_video(IMAGE_FOLDER, OUTPUT_VIDEO, fps=FPS)