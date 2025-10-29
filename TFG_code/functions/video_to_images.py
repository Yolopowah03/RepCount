import os
import cv2 as cv #type: ignore

VIDEO_FOLDER = '/datatmp2/joan/tfg_joan/videos/test/bench_press'
OUTPUT_FOLDER = '/datatmp2/joan/tfg_joan/LSTM_dataset/test/images/bench_press'

if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

def images_from_video(video_folder, output_folder):
    
    for video_file in sorted(os.listdir(video_folder)):
        
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.basename(video_path)[0:-4]

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
            if frame_number % 2 == 0:
                # timestamp = frame_number / fps
                image_name = f"{video_name}_{frame_number:03d}.jpg"
                image_path = os.path.join(output_folder, video_name, image_name)
                
                if not os.path.exists(os.path.join(output_folder, video_name)):
                    os.makedirs(os.path.join(output_folder, video_name))
                    
                cv.imwrite(image_path, frame)

        cap.release()
        print(f'Video {video_name} processed')

if __name__ == "__main__": 
    images_from_video(VIDEO_FOLDER, OUTPUT_FOLDER)