import os
import cv2 as cv #type: ignore

video_folder = '/datatmp2/joan/tfg_joan/videos/test/pull_Up'
output_folder = '/datatmp2/joan/tfg_joan/images/test/pull_up_1'

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def images_from_video(VIDEO_FOLDER, IMAGE_SAVE_DIR):

    for video_name in sorted(os.listdir(VIDEO_FOLDER)):
        
        if video_name != 'pull_up_1.mp4':
            continue
        
        video_path = os.path.join(VIDEO_FOLDER, video_name)

        video_name = video_name.replace('.avi', '_')

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
                image_path = os.path.join(IMAGE_SAVE_DIR, video_name+image_name)
                cv.imwrite(image_path, frame)

        cap.release()
        print(f'Video {video_name} processed')

if __name__ == "__main__": 
    images_from_video(video_folder, output_folder)