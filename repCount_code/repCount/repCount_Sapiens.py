import os
import cv2 as cv # type: ignore
import sys
import json
import numpy as np
import math
PYTHON_POSE_PATH = "/datatmp2/joan/repCount/sapiens/lite/demo"
sys.path.append(PYTHON_POSE_PATH)
PYTHON_LSTM_PATH = '/datatmp2/joan/repCount/repCount_code/LSTM'
sys.path.append(PYTHON_LSTM_PATH)
import vis_pose_mod # type: ignore
from vis_pose_mod import vis_pose_main # type: ignore
import predict_LSTM_mod # type: ignore
from predict_LSTM_mod import lstm_main # type: ignore

# VIDEO_PATH = '/datatmp2/joan/repCount/videos/test/squat/bench_press_1.mp4'
VIDEO_PATH = '/datatmp2/joan/repCount/videos/test/bench_press/bench_press_1.mp4'

# OUTPUT_DIR="/datatmp2/joan/repCount/results/repcount"
OUTPUT_DIR="/datatmp2/joan/repCount/results/repcount_bench_press_1"

POSE_CHECKPOINT='/datatmp2/joan/repCount/sapiens/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2'
DETECTION_CONFIG_FILE='/datatmp2/joan/repCount/sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py'
DETECTION_CHECKPOINT='/datatmp2/joan/repCount/sapiens/sapiens_lite_host/torchscript/detector/checkpoints/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

LSTM_MODEL_PATH = '/datatmp2/joan/repCount/models_LSTM/LSTM_RepCount1.pth'

# PRE_LOADED_JSON_DIR = '/datatmp2/joan/repCount/results/repcount'
PRE_LOADED_JSON_DIR = '/datatmp2/joan/repCount/results/repcount_bench_press_1'

#Sapiens
LINE_THICKNESS=2
RADIUS=6
KPT_THRES=0.3
NUM_KEYPOINTS = 17
JOBS_PER_GPU=1; TOTAL_GPUS=1; VALID_GPU_ID=3
BATCH_SIZE=8
SHAPE = [1024, 768]
BBOX_THR = 0.3
NMS_THR = 0.3
KPT_THR = 0.3
RADIUS = 9
THICKNESS = -1
HEATMAP_SCALE = 4
FLIP = False

#LSTM 
N_KEYPOINTS = 17
VEL = True
SEQ_LEN = 80

def extract_images_from_video(video_path):
    
    images = []
    idx = 0
    
    cap1 = cv.VideoCapture(video_path)

    while cap1.isOpened():
        ret, frame = cap1.read()
        if not ret:
            break

        if idx % 2 == 1:
            continue
        
        idx += 1

        images.append(frame)

    cap1.release()

    return images

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

def get_angle(x0, y0, x1, y1, x2, y2):
    
    # x0: Intersecció
    
    v1 = (x1-x0, y1-y0)
    v2 = (x2-x0, y2-y0)
    
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    
    mag_v1 = np.sqrt(v1[0]**2 + v1[1]**2)
    mag_v2 = np.sqrt(v2[0]**2 + v2[1]**2)
    
    cos_theta = dot / (mag_v1 * mag_v2)
    
    theta = math.acos(cos_theta)
    theta_deg = math.degrees(theta)
    
    return theta_deg

def repcount_bench_press(keypoints):
    
    # 1: Distancia entre manos (9, 10) y hombros (5, 6) (brazos extendidos)
    # 2: Ángulo perpendicular entre hombros (5, 6), codos (7, 8) y muñecas (9, 10) (brazos flexionados) + Distancia 1 en un 20%
    # 3: Retorno a la posicion inicial, a un 90% mínimo de distancia inicial (brazos extendidos)
    
    timestamps = []
    timestamps_initial = []
    initialPosition = None
    end_rep = False
    
    for i, frame_keypoints in enumerate(keypoints):
        
        if i % 2:
            continue

        hand_vector = (frame_keypoints['left_wrist'][0]-frame_keypoints['right_wrist'][0], frame_keypoints['left_wrist'][1] - frame_keypoints['right_wrist'][1])
        extended_arm_distL = np.sqrt((frame_keypoints['left_wrist'][0]-frame_keypoints['left_shoulder'][0])**2 + (frame_keypoints['left_wrist'][1]-frame_keypoints['left_shoulder'][1])**2)
        extended_arm_distR = np.sqrt((frame_keypoints['right_wrist'][0]-frame_keypoints['right_shoulder'][0])**2 + (frame_keypoints['right_wrist'][1]-frame_keypoints['right_shoulder'][1])**2)
        extended_arm_dist = (extended_arm_distL + extended_arm_distR) / 2
        
        # Trobar InitialPosition i actualitzar-la si es troben millors posicions
        
        if initialPosition is None:
            initialPosition = {}
            initialPosition['frame'] = i
            initialPosition['extended_arm_dist'] = extended_arm_dist
            end_rep = False
            print('Initial Position arm_dist:')
            print(initialPosition)
            timestamps_initial.append(i)
            
        else:
            if extended_arm_dist > initialPosition['extended_arm_dist']:
                initialPosition['extended_arm_dist'] = extended_arm_dist
                initialPosition['frame'] = i
                end_rep = False
                print('Initial Position:')
                print(initialPosition)
                timestamps_initial.append(i)
                
            if initialPosition is not None:
                #Trobar quan es retorna a la posició inicial
                if extended_arm_dist >= (initialPosition['extended_arm_dist'] * 0.9):
                    initialPosition['frame'] = i
                    end_rep = False
                    
                print('Extended arm dist:')
                print(extended_arm_dist)
                    
                #Trobar fi de repetició
                if end_rep == False and extended_arm_dist <= (initialPosition['extended_arm_dist'] * 0.5):
                    end_rep = True
                    timestamps.append(i)
                    
    print('timestamps_initial', timestamps_initial)
    return timestamps
    
def repcount_deadlift(keypoints):
    
    # 1: 
    # 2: 
    
    # return timestamps
    pass
    
def repcount_squat(keypoints):
    # return timestamps
    pass

def repcount_pull_up(keypoints):
    # 1: Distancia entre manos y hombros (brazos extendidos)
    # 2: Punto medio entre linea orejas y linea hombros pasa linea manos (brazos flexionados)
    # 3: Retorno a la posicion inicial, a un 90% mínimo de distancia inicial (brazos extendidos)
    
    # return timestamps
    pass
    
if __name__ == "__main__":
    
    # images = extract_images_from_video(VIDEO_PATH)
    
    pre_loaded = False

    if PRE_LOADED_JSON_DIR is not None and os.path.exists(PRE_LOADED_JSON_DIR):
        for json_file in os.listdir(PRE_LOADED_JSON_DIR):
            if json_file.endswith('.json'):
                pre_loaded = True
                break

    if not pre_loaded:
        args_pose = {}

        args_pose["pose_checkpoint"] = POSE_CHECKPOINT
        args_pose["det_config"] = DETECTION_CONFIG_FILE
        args_pose["det_checkpoint"] = DETECTION_CHECKPOINT
        args_pose["input"] = VIDEO_PATH
        args_pose["num_keypoints"] = NUM_KEYPOINTS
        args_pose["shape"] = SHAPE
        args_pose["output_root"] = OUTPUT_DIR
        args_pose["batch_size"] = BATCH_SIZE
        args_pose["fp16"] = False
        args_pose["device"] = fr"cuda:{VALID_GPU_ID}"
        args_pose["det_cat_id"] = 0
        args_pose["bbox_thr"] = BBOX_THR
        args_pose["nms_thr"] = NMS_THR
        args_pose["kpt_thr"] = KPT_THR
        args_pose["radius"] = RADIUS
        args_pose["thickness"] = THICKNESS
        args_pose["heatmap_scale"] = HEATMAP_SCALE
        args_pose["flip"] = False

        out_pose = vis_pose_main(args_pose)
        
        total_time = out_pose['total_time']
        resulting_kpts = out_pose['resulting_kpts']
    
        print('Total time:', total_time)
        print('Total images:', len(os.listdir(OUTPUT_DIR)) / 2)
    
        
    else:
        resulting_kpts = []
        for file in os.listdir(PRE_LOADED_JSON_DIR):
            if file.endswith('.json'):
                file_path = os.path.join(PRE_LOADED_JSON_DIR, file)
                with open(file_path, 'r') as f:
                    
                    data = json.load(f)
                    
                    instance_info = data.get('instance_info', {})
                    keypoints = instance_info[0]['keypoints'][0]
                    
                resulting_kpts.append(np.array(keypoints))
                
    resulting_kpts = np.array(resulting_kpts)
    
    resulting_kpts.reshape(-1, 17, 2)
                  
    args_lstm = {}
      
    args_lstm['keypoints'] = resulting_kpts
    args_lstm["vel"] = VEL
    args_lstm["seq_len"] = SEQ_LEN
    args_lstm["classes"] = CLASSES
    args_lstm["n_keypoints"] = N_KEYPOINTS
    args_lstm["model_path"] = LSTM_MODEL_PATH

    pred_label, probs = lstm_main(args_lstm)
    
    # print('probs:', probs)
    
    cls_idx = CLASSES.index(pred_label)
    
    kpts_dict_list = []
    for frame_kpts in resulting_kpts:
        
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
        kpts_dict['left_knee'] = frame_kpts[13]
        kpts_dict['right_knee'] = frame_kpts[14]
        kpts_dict['left_ankle'] = frame_kpts[15]
        kpts_dict['right_ankle'] = frame_kpts[16]

        kpts_dict_list.append(kpts_dict)
    
    print(len(kpts_dict_list))
    
    match(cls_idx):
        case 0:
            print('Predicted exercise: Bench Press')
            
            time_labels = repcount_bench_press(kpts_dict_list)
            print(time_labels)
            
        case 1:
            print('Predicted exercise: Deadlift')
            
            time_labels = repcount_deadlift(kpts_dict_list)
            
        case 2:
            print('Predicted exercise: Squat')

            time_labels = repcount_squat(kpts_dict_list)

        case 3:
            print('Predicted exercise: Pull Up')

            time_labels = repcount_pull_up(kpts_dict_list)
            
        case _:
            print('Error: Unknown class index')