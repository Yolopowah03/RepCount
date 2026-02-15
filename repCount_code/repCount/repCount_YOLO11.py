import os
import cv2 as cv # type: ignore
import sys
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import subprocess

time_count = time.time()

PYTHON_LSTM_PATH = '/datatmp2/joan/repCount/repCount_code/LSTM'
PYTHON_YOLO_PATH = '/datatmp2/joan/repCount/repCount_code/YOLO_pose'
PYTHON_HOMOGRAPHY_PATH = '/datatmp2/joan/repCount/repCount_code/Homography'

for p in (PYTHON_LSTM_PATH, PYTHON_YOLO_PATH, PYTHON_HOMOGRAPHY_PATH):
    if p not in sys.path:
        sys.path.append(p)
        
import predict_LSTM_mod # type: ignore
import predict_YOLO11_mod # type: ignore
import homography_mod # type: ignore

VIDEO_PATH = '/datatmp2/joan/repCount/videos/train/pull_up/train_pull_up_013.mp4'
OUTPUT_DIR_POSE = "/datatmp2/joan/repCount/results/pose/pull_up/pull_up_test_013"
OUTPUT_PATH_VIDEO = "/datatmp2/joan/repCount/results/repCount/repcount_pull_up_test_013.mp4"

CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']
N_KEYPOINTS_TOTAL = 17
N_KEYPOINTS_SHORTENED = 13
SKIP_FRAMES = 1
FPS_REDUCTION = 1

# PRE_LOADED_JSON_DIR = '/datatmp2/joan/repCount/results/repcount'
PRE_LOADED_JSON_DIR = '/datatmp2/joan/repCount/results/YOLO_pose/repcount_bench_press11'

#YOLO
YOLO_MODEL_PATH='/datatmp2/joan/repCount/models_YOLO11_pose/yolo11m-pose.pt'
VALID_GPU_ID = 3
SAVE_FRAMES = False
SAVE_JSON = False
MIN_CONF = 0.15

#LSTM 
LSTM_MODEL_PATH = '/datatmp2/joan/repCount/models_LSTM/LSTM_17_RepCount2.pth'
VEL = True
SEQ_LEN = 80

COCO_KPTS_COLORS = [
    [51, 153, 255],   # 0: nose
    [51, 153, 255],   # 1: left_eye
    [51, 153, 255],   # 2: right_eye
    [51, 153, 255],   # 3: left_ear
    [51, 153, 255],   # 4: right_ear
    [0, 255, 0],      # 5: left_shoulder
    [255, 128, 0],    # 6: right_shoulder
    [0, 255, 0],      # 7: left_elbow
    [255, 128, 0],    # 8: right_elbow
    [0, 255, 0],      # 9: left_wrist
    [255, 128, 0],    # 10: right_wrist
    [0, 255, 0],      # 11: left_hip
    [255, 128, 0],    # 12: right_hip
    [0, 255, 0],      # 13: left_knee
    [255, 128, 0],    # 14: right_knee
    [0, 255, 0],      # 15: left_ankle
    [255, 128, 0],    # 16: right_ankle
]

COCO_SKELETON_INFO_17 = {
    0: dict(link=(15, 13), id=0, color=[0, 255, 0]),
    1: dict(link=(13, 11), id=1, color=[0, 255, 0]),
    2: dict(link=(16, 14), id=2, color=[255, 128, 0]),
    3: dict(link=(14, 12), id=3, color=[255, 128, 0]),
    4: dict(link=(11, 12), id=4, color=[51, 153, 255]),
    5: dict(link=(5, 11), id=5, color=[51, 153, 255]),
    6: dict(link=(6, 12), id=6, color=[51, 153, 255]),
    7: dict(link=(5, 6), id=7, color=[51, 153, 255]),
    8: dict(link=(5, 7), id=8, color=[0, 255, 0]),
    9: dict(link=(6, 8), id=9, color=[0, 255, 0]),
    10: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    11: dict(link=(8, 10), id=11, color=[0, 255, 0]),
    12: dict(link=(1, 2), id=12, color=[51, 153, 255]),
    13: dict(link=(0, 1), id=13, color=[51, 153, 255]),
    14: dict(link=(0, 2), id=14, color=[51, 153, 255]),
    15: dict(link=(1, 3), id=15, color=[51, 153, 255]),
    16: dict(link=(2, 4), id=16, color=[51, 153, 255]),
    17: dict(link=(3, 5), id=17, color=[51, 153, 255]),
    18: dict(link=(4, 6), id=18, color=[51, 153, 255])
}

COCO_SKELETON_INFO_13 = {
    0: dict(link=(11, 12), id=4, color=[51, 153, 255]),
    1: dict(link=(5, 11), id=5, color=[51, 153, 255]),
    2: dict(link=(6, 12), id=6, color=[51, 153, 255]),
    3: dict(link=(5, 6), id=7, color=[51, 153, 255]),
    4: dict(link=(5, 7), id=8, color=[0, 255, 0]),
    5: dict(link=(6, 8), id=9, color=[0, 255, 0]),
    6: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    7: dict(link=(8, 10), id=11, color=[0, 255, 0]),
    8: dict(link=(1, 2), id=12, color=[51, 153, 255]),
    9: dict(link=(0, 1), id=13, color=[51, 153, 255]),
    10: dict(link=(0, 2), id=14, color=[51, 153, 255]),
    11: dict(link=(1, 3), id=15, color=[51, 153, 255]),
    12: dict(link=(2, 4), id=16, color=[51, 153, 255]),
    13: dict(link=(3, 5), id=17, color=[51, 153, 255]),
    14: dict(link=(4, 6), id=18, color=[51, 153, 255])
}

#KEYPOINT INDEX:

# 0: nose
# 1: left_eye
# 2: right_eye
# 3: left_ear
# 4: right_ear
# 5: left_shoulder
# 6: right_shoulder
# 7: left_elbow
# 8: right_elbow
# 9: left_wrist
# 10: right_wrist
# 11: left_hip
# 12: right_hip
# 13: left_knee
# 14: right_knee
# 15: left_ankle
# 16: right_ankle

def extract_images_from_video(video_path, skip_frames):
    
    images = []
    idx = 0
    
    cap1 = cv.VideoCapture(video_path)
    fps_og = cap1.get(cv.CAP_PROP_FPS)
    
    success, frame = cap1.read()
    
    if not success:
        raise ValueError('Error reading video file at path:', video_path)

    while success:

        if idx % skip_frames != 1:
            images.append(frame)

        idx += 1
        
        success, frame = cap1.read()

    cap1.release()

    return images, fps_og

def draw_keypoints(image, kpts, kpt_colors, skeleton_info, kpt_thr=0.3, radius=3, thickness=2):
    
    for kid, kpt in sorted(enumerate(kpts)):
        # print(kpt)

        color = kpt_colors[kid]
        if not isinstance(color, str):
            color = tuple(int(c) for c in color[::-1])
        image = cv.circle(image, (int(kpt[0]), int(kpt[1])), int(radius), color, -1)
        
        # draw skeleton
        for skid, link_info in skeleton_info.items():
            pt1_idx, pt2_idx = link_info['link']
            color = link_info['color'][::-1] # BGR

            pt1 = kpts[pt1_idx]
            pt2 = kpts[pt2_idx]

            x1_coord = int(pt1[0]); y1_coord = int(pt1[1])
            x2_coord = int(pt2[0]); y2_coord = int(pt2[1])
            cv.line(image, (x1_coord, y1_coord), (x2_coord, y2_coord), color, thickness=thickness)
            
    return image

def draw_counter(image, count):
    font = cv.FONT_ITALIC
    text_position = (100, 100)
    fontScale = 2
    fontColor = (255, 0, 0)
    lineType = 3

    cv.putText(image, f'Rep Count: {count}', 
        text_position, 
        font, 
        fontScale,
        fontColor,
        lineType)
    
    return image

# def get_angle(x0, y0, x1, y1, x2, y2):
    
#     # x0: Intersecció
    
#     v1 = (x1-x0, y1-y0)
#     v2 = (x2-x0, y2-y0)
    
#     dot = v1[0]*v2[0] + v1[1]*v2[1]
    
#     mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
#     mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
#     cos_theta = dot / (mag_v1 * mag_v2)
    
#     theta = math.acos(cos_theta)
#     theta_deg = math.degrees(theta)
    
#     return theta_deg

def create_histogram(history, history_swiftened, save_path, counts, counts_end, pred_label):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), dpi=100)
    plt.subplots_adjust(hspace=0.4)
    
    ax[0].plot(history, color='red', linewidth=2, label='Original')
    ax[1].plot(history_swiftened, color='blue', linewidth=2, label='Swiftened')
    
    ax[0].set_xlim(0, len(history))
    ax[1].set_xlim(0, len(history))
    ax[0].set_ylim(0, max(history))
    ax[1].set_ylim(0, max(history))
    
    for count in counts:
        ax[0].axvline(x=count, color='green', linewidth=2)
        ax[1].axvline(x=count, color='green', linewidth=2)
        
    for count_end in counts_end:
        ax[0].axvline(x=count_end, color='orange', linewidth=2, linestyle='--')
        ax[1].axvline(x=count_end, color='orange', linewidth=2, linestyle='--')
        
    ax[0].set_title("Anàlisi de l'exercici")
    ax[0].set_xlabel('Frame')
    ax[0].set_ylabel('Moviment')
    ax[0].grid(True, linestyle=':', alpha=0.7)
    ax[1].set_title("Anàlisi de l'exercici (suavitzat)")
    ax[1].set_xlabel('Frame')
    ax[1].set_ylabel('Moviment')   
    ax[1].grid(True, linestyle=':', alpha=0.7)
    
    fig.suptitle(f'Anàlisi de pose {pred_label}', fontsize=16)
    
    plt.savefig(save_path)

def repcount_bench_press(keypoints):

    #PLA: Canells, espatlles
    
    # Initial position: Mans esteses cap amunt
    # end position: Barra toca el pit
    
    timestamps = []
    timestamps_end = []
    initialPosition = None
    end_rep = False
    timer = 0
    distance_history_swiftened = []
    distance_history = []
    first_dist = 0
    
    for i, frame_keypoints in enumerate(keypoints):
        
        if (i % SKIP_FRAMES) == 1:
            continue
        
        if timer > 0:
            timer = timer - 1

        wrist_shoulder_distL = np.sqrt((frame_keypoints['left_wrist'][0]-frame_keypoints['left_shoulder'][0])**2 + (frame_keypoints['left_wrist'][1]-frame_keypoints['left_shoulder'][1])**2)
        wrist_shoulder_distR = np.sqrt((frame_keypoints['right_wrist'][0]-frame_keypoints['right_shoulder'][0])**2 + (frame_keypoints['right_wrist'][1]-frame_keypoints['right_shoulder'][1])**2)
        wrist_shoulder_dist = (wrist_shoulder_distL + wrist_shoulder_distR) / 2
        
        distance_history.append(wrist_shoulder_dist)
        
        if len(distance_history_swiftened) < 5:
            distance_history_swiftened.append(wrist_shoulder_dist)
            
        elif len(distance_history_swiftened) >= 5 and first_dist == 0 and wrist_shoulder_dist < 300:
            first_dist = wrist_shoulder_dist
            distance_history_swiftened.append(wrist_shoulder_dist)
            
        elif wrist_shoulder_dist > (first_dist*2.25):
            distance_history_swiftened.append(distance_history_swiftened[-1])
            continue
        else:
            distance_history_swiftened.append(wrist_shoulder_dist)
        
        # Trobar primera InitialPosition 
        if initialPosition is None:
            initialPosition = {}
            initialPosition['wrist_shoulder_dist'] = wrist_shoulder_dist
            # print('Initial Position arm_dist:')
            # print(initialPosition)
            
        else:
           # Actualitzar initial positions si es troben millors
            if wrist_shoulder_dist > initialPosition['wrist_shoulder_dist'] and end_rep == False:
                initialPosition['wrist_shoulder_dist'] = wrist_shoulder_dist
                # print('Initial Position:')
                # print(initialPosition)
                
            #Trobar quan es retorna a la posició inicial
            if wrist_shoulder_dist >= (initialPosition['wrist_shoulder_dist'] * 0.9):
                if end_rep == True:
                    if timer <= 0 and i > 30:
                        timestamps.append(i)
                        timer += 15
                    
                end_rep = False

            # print('Extended arm dist:')
            # print(wrist_shoulder_dist)
                
            #Trobar fi de repetició
            
            if end_rep == False and wrist_shoulder_dist <= (initialPosition['wrist_shoulder_dist'] * 0.65):
                timestamps_end.append(i)
                end_rep = True
                    
    # print('timestamps_initial', timestamps_initial)
    return timestamps, timestamps_end, distance_history, distance_history_swiftened
    
def repcount_deadlift(keypoints):

    #PLA: Mans, peus
    # Initial Position: Mans agafant la barra del terra
    # end Position: Esquena recta, genolls estirats   

    timestamps = []
    timestamps_end = []
    initialPosition = None
    end_rep = False
    timer = 0
    distance_history_swiftened = []
    distance_history = []
    first_dist = 0
    
    for i, frame_keypoints in enumerate(keypoints):
        
        if (i % SKIP_FRAMES) == 1:
            continue
        
        if timer > 0:
            timer = timer - 1
        
        wrist_ankle_distL = np.sqrt((frame_keypoints['left_wrist'][0]-frame_keypoints['left_ankle'][0])**2 + (frame_keypoints['left_wrist'][1]-frame_keypoints['left_ankle'][1])**2)
        wrist_ankle_distR = np.sqrt((frame_keypoints['right_wrist'][0]-frame_keypoints['right_ankle'][0])**2 + (frame_keypoints['right_wrist'][1]-frame_keypoints['right_ankle'][1])**2)
        wrist_ankle_dist = (wrist_ankle_distL + wrist_ankle_distR) / 2
        
        distance_history.append(wrist_ankle_dist)
        
        if len(distance_history_swiftened) < 5:
            distance_history_swiftened.append(wrist_ankle_dist)
            
        elif len(distance_history_swiftened) >= 5 and first_dist == 0 and wrist_ankle_dist < 400:
            first_dist = wrist_ankle_dist
            distance_history_swiftened.append(wrist_ankle_dist)
            
        elif wrist_ankle_dist < (first_dist*0.4):
            distance_history_swiftened.append(distance_history_swiftened[-1])
            continue
        else:
            distance_history_swiftened.append(wrist_ankle_dist)
        
        # Trobar primera InitialPosition 
        if initialPosition is None:
            initialPosition = {}
            initialPosition['wrist_ankle_dist'] = wrist_ankle_dist
            # print('Initial Position arm_dist:')
            # print(initialPosition)

            
        else: 
            # Actualitzar initial positions si es troben millors
            if wrist_ankle_dist < initialPosition['wrist_ankle_dist'] and end_rep == False:
                initialPosition['wrist_ankle_dist'] = wrist_ankle_dist
                # print('Initial Position:')
                # print(initialPosition)                
            #Trobar quan es retorna a la posició inicial
            if wrist_ankle_dist <= (initialPosition['wrist_ankle_dist'] * 1.15):
                if end_rep == True:
                    if timer <= 0 and i > 30:
                        timestamps.append(i)
                        timer += 15
                    
                end_rep = False

            # print('Wrist-hip dist:')
            # print(wrist_hip_dist)
                
            #Trobar fi de repetició
            
            if end_rep == False and wrist_ankle_dist >= (initialPosition['wrist_ankle_dist'] * 1.4):
                timestamps_end.append(i)
                end_rep = True
                    
    # print('timestamps_initial', timestamps_initial)
    return timestamps, timestamps_end, distance_history, distance_history_swiftened
    
def repcount_squat(keypoints):
    
    #PLA: Espatlles, genolls
    
    # Initial Position: De peu, genolls estirats
    # end Position: Genolls flexionats
    
    timestamps = []
    timestamps_end = []
    initialPosition = None
    end_rep = False
    timer = 0
    distance_history_swiftened = []
    distance_history = []
    first_dist = 0
    
    for i, frame_keypoints in enumerate(keypoints):
        
        if (i % SKIP_FRAMES) == 1:
            continue
        
        if timer > 0:
            timer = timer - 1

        knee_shoulder_distL = np.sqrt((frame_keypoints['left_knee'][0]-frame_keypoints['left_shoulder'][0])**2 + (frame_keypoints['left_knee'][1]-frame_keypoints['left_shoulder'][1])**2)
        knee_shoulder_distR = np.sqrt((frame_keypoints['right_knee'][0]-frame_keypoints['right_shoulder'][0])**2 + (frame_keypoints['right_knee'][1]-frame_keypoints['right_shoulder'][1])**2)
        knee_shoulder_dist = (knee_shoulder_distL + knee_shoulder_distR) / 2
        
        distance_history.append(knee_shoulder_dist)
        
        if len(distance_history_swiftened) < 5:
            distance_history_swiftened.append(knee_shoulder_dist)
            
        elif len(distance_history_swiftened) >= 5 and first_dist == 0 and knee_shoulder_dist < 300:
            first_dist = knee_shoulder_dist
            distance_history_swiftened.append(knee_shoulder_dist)
            
        elif knee_shoulder_dist > (first_dist*2.25):
            distance_history_swiftened.append(distance_history_swiftened[-1])
            continue
        else:
            distance_history_swiftened.append(knee_shoulder_dist)
        
        # Trobar primera InitialPosition 
        if initialPosition is None:
            initialPosition = {}
            initialPosition['frame'] = i
            initialPosition['knee_shoulder_dist'] = knee_shoulder_dist
            # print('Initial Position arm_dist:')
            # print(initialPosition)
            
        else:
           # Actualitzar initial positions si es troben millors
            if knee_shoulder_dist > initialPosition['knee_shoulder_dist'] and end_rep == False:
                initialPosition['frame'] = i
                initialPosition['knee_shoulder_dist'] = knee_shoulder_dist
                # print('Initial Position:')
                # print(initialPosition)
                
            #Trobar quan es retorna a la posició inicial
            if knee_shoulder_dist >= (initialPosition['knee_shoulder_dist'] * 0.875):
                initialPosition['frame'] = i
                if end_rep == True:
                    if timer <= 0 and i > 30:
                        timestamps.append(i)
                        timer += 15
                    
                end_rep = False

            # print('Extended arm dist:')
            # print(wrist_shoulder_dist)
                
            #Trobar fi de repetició
            
            if end_rep == False and knee_shoulder_dist <= (initialPosition['knee_shoulder_dist'] * 0.75):
                timestamps_end.append(i)
                end_rep = True
                    
    # print('timestamps_initial', timestamps_initial)
    return timestamps, timestamps_end, distance_history, distance_history_swiftened

def repcount_pull_up(keypoints):
    
    #PLA: Canells, espatlles
    
    # Initial position: Cos estirat, braços estesos cap amunt
    # end position: Barana a l'altura del pit
    
    timestamps = []
    timestamps_end = []
    initialPosition = None
    end_rep = False
    timer = 0
    distance_history_swiftened = []
    distance_history = []
    first_dist = 0
    
    for i, frame_keypoints in enumerate(keypoints):
        
        if (i % SKIP_FRAMES) == 1:
            continue
        
        if timer > 0:
            timer = timer - 1

        wrist_shoulder_distL = np.sqrt((frame_keypoints['left_wrist'][0]-frame_keypoints['left_shoulder'][0])**2 + (frame_keypoints['left_wrist'][1]-frame_keypoints['left_shoulder'][1])**2)
        wrist_shoulder_distR = np.sqrt((frame_keypoints['right_wrist'][0]-frame_keypoints['right_shoulder'][0])**2 + (frame_keypoints['right_wrist'][1]-frame_keypoints['right_shoulder'][1])**2)
        wrist_shoulder_dist = (wrist_shoulder_distL + wrist_shoulder_distR) / 2
        distance_history.append(wrist_shoulder_dist)
        
        if len(distance_history_swiftened) < 5:
            distance_history_swiftened.append(wrist_shoulder_dist)
        elif wrist_shoulder_dist < (max(distance_history_swiftened)*1.2):
            distance_history_swiftened.append(wrist_shoulder_dist)
        else:
            continue
        
        # Trobar primera InitialPosition 
        if initialPosition is None:
            initialPosition = {}
            initialPosition['wrist_shoulder_dist'] = wrist_shoulder_dist
            # print('Initial Position arm_dist:')
            # print(initialPosition)
            
        else:
           # Actualitzar initial positions si es troben millors
            if wrist_shoulder_dist > initialPosition['wrist_shoulder_dist'] and end_rep == False:
                initialPosition['wrist_shoulder_dist'] = wrist_shoulder_dist
                # print('Initial Position dist:')
                # print(wrist_shoulder_dist)
                
                
            #Trobar quan es retorna a la posició inicial
            if wrist_shoulder_dist >= (initialPosition['wrist_shoulder_dist'] * 0.9):
                if end_rep == True:
                    if timer <= 0 and i > 30:
                        timestamps.append(i)
                        timer += 15
                    
                end_rep = False

            # print('Extended arm dist:')
            # print(wrist_shoulder_dist)
                
            #Trobar fi de repetició
            
            if end_rep == False and wrist_shoulder_dist <= (initialPosition['wrist_shoulder_dist'] * 0.65):
                timestamps_end.append(i)
                end_rep = True
                    
    # print('timestamps_initial', timestamps_initial)
    return timestamps, timestamps_end, distance_history, distance_history_swiftened

def save_video(image_list, output_path, fps):
    
    height, width, layers = image_list[0].shape
    frame_size = (width, height)
    
    temp_path = output_path + '.temp.mp4'
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(temp_path, fourcc, fps, frame_size)

    for img in image_list:
        out.write(img)

    out.release()
    
    ffmpeg_command = [
        'ffmpeg',
        '-i', temp_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        '-movflags', 'faststart',
        output_path
    ]
    
    try:
        subprocess.run(ffmpeg_command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("Error during ffmpeg execution:")
        print(e.stderr.decode())
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    
    print('Processing ', VIDEO_PATH)
    
    images, fps_og = extract_images_from_video(VIDEO_PATH, SKIP_FRAMES)

    pre_loaded = False

    if PRE_LOADED_JSON_DIR is not None and os.path.exists(PRE_LOADED_JSON_DIR):
        for json_file in os.listdir(PRE_LOADED_JSON_DIR):
            if json_file.endswith('.json'):
                pre_loaded = True
                break

    if not pre_loaded:
        args_pose = {}

        args_pose["model_path"] = YOLO_MODEL_PATH
        args_pose["input_video_path"] = VIDEO_PATH
        args_pose["output_dir"] = OUTPUT_DIR_POSE
        args_pose["device"] = fr"cuda:{VALID_GPU_ID}"
        args_pose["kpt_thr"] = 0.3
        args_pose["kpts_colors"] = COCO_KPTS_COLORS
        args_pose["skeleton_info"] = COCO_SKELETON_INFO_17
        args_pose['n_keypoints'] = N_KEYPOINTS_TOTAL
        args_pose['save_json'] = SAVE_JSON
        args_pose['save_frames'] = SAVE_FRAMES
        args_pose['skip_frames'] = SKIP_FRAMES
        args_pose['min_conf'] = MIN_CONF
        out_pose = predict_YOLO11_mod.yolo_mod(args_pose)
        
        total_time_pose = out_pose['total_time']
        resulting_kpts = out_pose['resulting_kpts']
        
    else:
        resulting_kpts = []
        for file in os.listdir(PRE_LOADED_JSON_DIR):
            if file.endswith('.json'):
                file_path = os.path.join(PRE_LOADED_JSON_DIR, file)
                with open(file_path, 'r') as f:
                    
                    data = json.load(f)
                    
                    instance_info = data.get('instance_info', {})
                    keypoints = instance_info[0]['keypoints']
                    
                resulting_kpts.append(np.array(keypoints[0: N_KEYPOINTS_TOTAL], dtype=np.float32))
                      
    resulting_kpts = np.array(resulting_kpts)
    resulting_kpts = resulting_kpts.reshape(-1, N_KEYPOINTS_TOTAL, 2)
    
    args_lstm = {}
      
    args_lstm['keypoints'] = resulting_kpts
    args_lstm["vel"] = VEL
    args_lstm["seq_len"] = SEQ_LEN
    args_lstm["classes"] = CLASSES
    args_lstm["n_keypoints"] = N_KEYPOINTS_TOTAL
    args_lstm["model_path"] = LSTM_MODEL_PATH

    pred_label, probs = predict_LSTM_mod.lstm_main(args_lstm)
    
    if pred_label == 'bench_press' or pred_label == 'pull_up':
        n_keypoints_used = N_KEYPOINTS_SHORTENED
        resulting_kpts = resulting_kpts[:, :N_KEYPOINTS_SHORTENED, :]
    else:
        n_keypoints_used = N_KEYPOINTS_TOTAL 
    
    args_homography = {}
    
    args_homography['keypoints'] = resulting_kpts
    args_homography['n_keypoints'] = n_keypoints_used
    args_homography['class_name'] = pred_label
    args_homography['img_shape'] = images[0].shape[:2]

    corrected_kpts, reference_kp_distr, indices = homography_mod.homography_main(args_homography)

    # print('probs:', probs)
    
    cls_idx = CLASSES.index(pred_label)
    
    kpts_dict_list = []
    for frame_kpts in corrected_kpts:
        
        kpts_dict = {}

        kpts_dict['nose'] = frame_kpts[0]
        kpts_dict['left_eye'] = frame_kpts[1]
        kpts_dict['right_eye'] = frame_kpts[2]
        kpts_dict['left_ear'] = frame_kpts[3]
        kpts_dict['right_ear'] = frame_kpts[4]
        kpts_dict['left_shoulder'] = frame_kpts[5]
        kpts_dict['right_shoulder'] = frame_kpts[6]
        kpts_dict['left_elbow'] = frame_kpts[7]
        kpts_dict['right_elbow'] = frame_kpts[8]
        kpts_dict['left_wrist'] = frame_kpts[9]
        kpts_dict['right_wrist'] = frame_kpts[10]
        kpts_dict['left_hip'] = frame_kpts[11]
        kpts_dict['right_hip'] = frame_kpts[12]
        
        if n_keypoints_used == N_KEYPOINTS_TOTAL:
            kpts_dict['left_knee'] = frame_kpts[13]
            kpts_dict['right_knee'] = frame_kpts[14]
            kpts_dict['left_ankle'] = frame_kpts[15]
            kpts_dict['right_ankle'] = frame_kpts[16]

        kpts_dict_list.append(kpts_dict)
    
    match(cls_idx): # type: ignore
        case 0:
            print('Predicted exercise: Bench Press')
            
            time_labels, time_labels_end, distance_history, distance_history_swiftened = repcount_bench_press(kpts_dict_list)
            print('RepCount:', time_labels)
            
        case 1:
            print('Predicted exercise: Deadlift')
            
            time_labels, time_labels_end, distance_history, distance_history_swiftened = repcount_deadlift(kpts_dict_list)
            
        case 2:
            print('Predicted exercise: Squat')

            time_labels, time_labels_end, distance_history, distance_history_swiftened = repcount_squat(kpts_dict_list)

        case 3:
            print('Predicted exercise: Pull Up')

            time_labels, time_labels_end, distance_history, distance_history_swiftened = repcount_pull_up(kpts_dict_list)
            
        case _:
            print('Error: Unknown class index')
            
    counter_images = []
    count = 0
    i = 0

    for img, kps, warped_kpts in zip(images, resulting_kpts, corrected_kpts):
        
        if n_keypoints_used == N_KEYPOINTS_SHORTENED:
            drawn_img = draw_keypoints(img.copy(), kps, COCO_KPTS_COLORS, COCO_SKELETON_INFO_13)
        else:
            drawn_img = draw_keypoints(img.copy(), kps, COCO_KPTS_COLORS, COCO_SKELETON_INFO_17)
        counter_img = draw_counter(drawn_img, count)
        i += 1

        if (kps == 0).all():
            counter_images.append(counter_img)
            continue
    
        for ref_kp in reference_kp_distr:
            counter_img = cv.circle(counter_img, (int(ref_kp[0]), int(ref_kp[1])), 5, (255, 0, 0), -1)
            
        for j, warped in enumerate(warped_kpts):
            if j in indices:
                counter_img = cv.circle(counter_img, (int(warped[0]), int(warped[1])), 3, (0, 0, 255), -1)
    
        if time_labels is not None and i in time_labels:
            count += 1
            
        counter_images.append(counter_img)
        
    video_save_dir = os.path.dirname(OUTPUT_PATH_VIDEO)
    video_save_name = os.path.basename(OUTPUT_PATH_VIDEO).split('.mp4')[0]
    hist_save_path = os.path.join(video_save_dir, video_save_name + '_hist.jpg')
    
    if not os.path.exists(video_save_dir):
        os.makedirs(video_save_dir)
        
    histogram = create_histogram(distance_history, distance_history_swiftened, hist_save_path, time_labels, time_labels_end, pred_label)
    
    print('Saved histogram at:', hist_save_path)
            
    print('Total processed frames:', len(counter_images))
    save_video(counter_images, OUTPUT_PATH_VIDEO, fps=fps_og/(SKIP_FRAMES*FPS_REDUCTION))
    print('Saved output at:', OUTPUT_PATH_VIDEO)
    
    print('Total time pose estimation:', np.round(total_time_pose, 3), ' seconds')
    print('Total time:', np.round(time.time() - time_count, 3), ' seconds')
    
    print('Total reps counted:', count)