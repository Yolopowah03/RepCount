import numpy as np
import cv2 # type: ignore
import os
import json

LABEL_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels'
OUTPUT_IMAGE_DIR = '/datatmp2/joan/tfg_joan/results/homography'
CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

# Mapeo COCO 17: intercambiar índices left<->right
# COCO order: 0:nose,1:l_eye,2:r_eye,3:l_ear,4:r_ear,5:l_sh,6:r_sh,7:l_elb,8:r_elb,9:l_wri,10:r_wri,
# 11:l_hip,12:r_hip

BENCH_PRESS_DISTR = np.array([[0.4127,0.368],[0.6513,0.453],[0.41,0.217],[0.656,0.244]], dtype=np.float32)
PULL_UP_DISTR = np.array([[0.4733, 0.4166],[0.548,0.4282],[0.483,0.5984],[0.5338,0.604]], dtype=np.float32)
DEADLIFT_DISTR = np.array([[0.4811,0.32],[0.55,0.3125],[0.4915,0.4594],[0.5384,0.458]], dtype=np.float32)
SQUAT_DISTR = np.array([[0.3867,0.229],[0.472,0.229],[0.4199,0.539],[0.513,0.539]], dtype=np.float32)

# LEFT_RIGHT_MAP = {
#     1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7, 9:10, 10:9, 11:12, 12:11
# }

BENCH_PRESS_INDICES = [7, 8, 9, 10]
PULL_UP_INDICES = [5, 6, 11, 12]
DEADLIFT_INDICES = [5, 6, 11, 12]
SQUAT_INDICES = [5, 6, 11, 12]


def homography_main(args):

    if args['keypoints'] is None:
        raise ValueError(f'No keypoints found')

    match(args['class_name']):
        case 'bench_press':
            kp_distr_norm = BENCH_PRESS_DISTR
            indices = BENCH_PRESS_INDICES
        case 'pull_up':
            kp_distr_norm = PULL_UP_DISTR
            indices = PULL_UP_INDICES
        case 'deadlift':
            kp_distr_norm = DEADLIFT_DISTR
            indices = DEADLIFT_INDICES
        case 'squat':
            kp_distr_norm = SQUAT_DISTR
            indices = SQUAT_INDICES
            
            
    reference_kp_distr = []
    for kp in kp_distr_norm:
        reference_kp_distr.append([kp[0] * args['img_shape'][1], kp[1] * args['img_shape'][0]])

    reference_kp_distr = np.array(reference_kp_distr, dtype=np.float32)

    j = 0
    
    kp_distr = np.array([args['keypoints'][j][i] for i in indices], dtype=np.float32)
    
    kp_distr = np.array([args['keypoints'][j][i] for i in indices], dtype=np.float32)
    while kp_distr.all() == np.zeros((len(indices), 2), dtype=np.float32).all():
        if j >= len(args['keypoints']):
            raise ValueError('No valid keypoints found for homography computation')
        kp_distr = np.array([args['keypoints'][j][i] for i in indices], dtype=np.float32)
        j += 1

    H, _ = cv2.findHomography(kp_distr, reference_kp_distr, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is None:
        raise ValueError('Homography could not be computed')
    
    corrected_kpts = []

    for kps in args['keypoints']:
        
        if (kps == 0).all():
            warped_kps = np.zeros_like(kps)
            corrected_kpts.append(warped_kps)
            continue
        
        kps_temp = kps[indices]

        kps_reshaped = kps_temp.astype(np.float32).reshape(-1,1,2)
        warped_kps = cv2.perspectiveTransform(kps_reshaped, H)
        warped_kps = warped_kps.reshape(-1,2)
        
        
        warped_kps_full = kps.copy()
        warped_kps_full[indices] = warped_kps
        corrected_kpts.append(warped_kps_full)
        
    return corrected_kpts, reference_kp_distr, indices