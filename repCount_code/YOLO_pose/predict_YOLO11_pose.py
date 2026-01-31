from ultralytics import YOLO #type: ignore
import os
import cv2 as cv #type: ignore
import numpy as np
import json

MODEL_PATH = '/datatmp2/joan/repCount/models_YOLO11_pose/yolo11m-pose.pt'
INPUT_DIR = '/datatmp2/joan/repCount/LSTM_dataset/test/images/bench_press'
OUTPUT_DIR = '/datatmp2/joan/repCount/LSTM_dataset/test/labels/bench_press'

CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

# Load a model
model = YOLO(MODEL_PATH)

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

COCO_SKELETON_INFO = {
    0: dict(link=(15, 13), id=0, color=[0, 255, 0]),
    1: dict(link=(13, 11), id=1, color=[0, 255, 0]),
    2: dict(link=(16, 14), id=2, color=[255, 128, 0]),
    3: dict(link=(14, 12), id=3, color=[255, 128, 0]),
    4: dict(link=(11, 12), id=4, color=[51, 153, 255]),
    5: dict(link=(5, 11), id=5, color=[51, 153, 255]),
    6: dict(link=(6, 12), id=6, color=[51, 153, 255]),
    7: dict(link=(5, 6), id=7, color=[51, 153, 255]),
    8: dict(link=(5, 7), id=8, color=[0, 255, 0]),
    9: dict(link=(6, 8), id=9, color=[255, 128, 0]),
    10: dict(link=(7, 9), id=10, color=[0, 255, 0]),
    11: dict(link=(8, 10), id=11, color=[255, 128, 0]),
    12: dict(link=(1, 2), id=12, color=[51, 153, 255]),
    13: dict(link=(0, 1), id=13, color=[51, 153, 255]),
    14: dict(link=(0, 2), id=14, color=[51, 153, 255]),
    15: dict(link=(1, 3), id=15, color=[51, 153, 255]),
    16: dict(link=(2, 4), id=16, color=[51, 153, 255]),
    17: dict(link=(3, 5), id=17, color=[51, 153, 255]),
    18: dict(link=(4, 6), id=18, color=[51, 153, 255])
}

# Predict with the model

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

def save_json(keypoints, scores, output_path, KPT_THR):

    pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")
    
    if np.mean(scores.tolist()) <= KPT_THR:
        keypoints = np.zeros((17, 2))
        scores = np.zeros((17,))
        
    with open(pred_save_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": scores.tolist(),
                    }
                ]
            ),
            f,
            indent="\t",
        )

# Access the results
# for class_name in CLASSES:
# for sub_dir, _, files in os.walk(os.path.join(INPUT_DIR, class_name)):
for sub_dir, _, files in os.walk(INPUT_DIR):
    print(os.path.join(INPUT_DIR, sub_dir))
    for file in sorted(os.listdir(os.path.join(INPUT_DIR, sub_dir))):
        if file.endswith('.jpg'):
            file_path = os.path.join(INPUT_DIR, sub_dir, file)
            save_dir = os.path.join(OUTPUT_DIR, os.path.basename(sub_dir))
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            save_path = os.path.join(save_dir, file)
            save_path_json = save_path.replace(".jpg", ".json").replace(".png", ".json")
            
            print(save_path)
            
            img = cv.imread(file_path)
            
            results = model(img)  # predict on an image
            
            for result in results:
                xy = result.keypoints.xy  # x and y coordinates
                xyn = result.keypoints.xyn  # normalized
                kpts = result.keypoints.data  # x, y, visibility (if available)
                scores = result.keypoints.conf  # confidence scores
                
                xy = xy.cpu()
                xy = np.asarray(xy, dtype=np.float32)
                
                if xy.shape[0] != 17 and xy.shape[0] > 0:
                    xy = xy[0]
                elif xy.shape[0] == 0:
                    xy = np.zeros((17, 2))
                    
            save_img = draw_keypoints(img, xy, COCO_KPTS_COLORS, COCO_SKELETON_INFO)
            save_json(xy, scores, save_path_json, KPT_THR=0.6)
                
            print(save_path)
            # cv.imwrite(save_path, save_img)