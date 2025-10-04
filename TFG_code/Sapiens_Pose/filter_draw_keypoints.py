import os
import json
import numpy as np
import cv2 as cv # type: ignore

INPUT_IMAGE_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/images/squat'
OUTPUT_IMAGE_DIR = '/datatmp2/joan/tfg_joan/results/pose/train_LSTM/squat'
JSON_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels/squat'

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

json_files_array = []

for subdir, _, json_files in sorted(os.walk(JSON_DIR)):
    for file in sorted(json_files):
        if file.endswith('.json'):
            json_files_array.append(os.path.join(subdir, file))

image_files_array = []

for subdir, _, img_files in sorted(os.walk(INPUT_IMAGE_DIR)):
    for file in sorted(img_files):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_files_array.append(os.path.join(subdir, file))

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def rewrite_json(file_path, data):

    #remove previous json
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'w') as f:
        json.dump(data, f)
        
def redraw_image(image_path, output_path, kpts, kpt_colors, skeleton_info, kpt_thr=0.3, radius=3, thickness=2):

    image = cv.imread(image_path)

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

    cv.imwrite(output_path, image)


for json_file, image_file in sorted(zip(json_files_array, image_files_array)):

    # print(json_file)
    data = load_json(os.path.join(JSON_DIR, json_file))
    # Process the JSON data as needed
    instance_info = data.get('instance_info')
    
    for prediction in instance_info:
        keypoints = prediction['keypoints']
        diff_array = []
        
        min_x = 9999
        min_y = 9999
        max_x = 0
        max_y = 0

        for keypoint in keypoints:
            x, y, = keypoint

            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        diff_x = max_x - min_x
        diff_y = max_y - min_y
        
        total_diff = diff_x + diff_y
        
        diff_array.append(total_diff)
        
    selected_body = np.argmax(diff_array)

    filtered_prediction = instance_info[selected_body]

    new_data = {
        "instance_info": [filtered_prediction]
    }

    rewrite_json(os.path.join(JSON_DIR, json_file), new_data)

    output_path = image_file.replace(INPUT_IMAGE_DIR, OUTPUT_IMAGE_DIR)
    
    print('redrawn ', output_path)

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    redraw_image(image_file, output_path, filtered_prediction['keypoints'], COCO_KPTS_COLORS, COCO_SKELETON_INFO)
    
    data.clear()
    instance_info.clear()
    diff_array.clear()
    keypoints.clear()
    filtered_prediction.clear()
    new_data.clear()