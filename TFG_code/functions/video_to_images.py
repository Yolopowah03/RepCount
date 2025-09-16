import os
import cv2 as cv #type: ignore

VIDEO_PATH = '/datatmp2/joan/tfg_joan/videos/test/deadlift/deadlift_2.mp4'
OUTPUT_FOLDER = '/datatmp2/joan/tfg_joan/images/test/deadlift2'

if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def images_from_video(video_path, output_folder):
    
    video_name = os.path.basename(video_path)

    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return
        
    # true_fps = cap.get(cv.CAP_PROP_FPS)

    # print(f"FPS: {true_fps}")

    frame_number = -1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        if frame_number % 10 == 0:
            # timestamp = frame_number / fps
            image_name = f"frame_{frame_number:04d}.jpg"
            image_path = os.path.join(output_folder, video_name+image_name)
            cv.imwrite(image_path, frame)

    cap.release()
    print(f'Video {video_name} processed')

if __name__ == "__main__": 
    images_from_video(VIDEO_PATH, OUTPUT_FOLDER)