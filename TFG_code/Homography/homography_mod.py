import numpy as np
import cv2 # type: ignore
import os
import json

LABEL_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels'
OUTPUT_IMAGE_DIR = '/datatmp2/joan/tfg_joan/results/homography'
CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

BENCH_PRESS_DISTR = np.array([[0.4127,0.368],[0.6513,0.453],[0.41,0.217],[0.656,0.244]], dtype=np.float32)
PULL_UP_DISTR = np.array([[0.478,0.437],[0.542,0.437],[0.433,0.194],[0.579,0.194]], dtype=np.float32)
DEADLIFT_DISTR = np.array([[0.36,0.713],[0.56,0.736],[0.43,0.859],[0.51,0.867]], dtype=np.float32)
SQUAT_DISTR = np.array([[0.534,0.573],[0.455,0.61],[0.579,0.877],[0.49,0.942]], dtype=np.float32)

# LEFT_RIGHT_MAP = {
#     1:2, 2:1, 3:4, 4:3, 5:6, 6:5, 7:8, 8:7, 9:10, 10:9, 11:12, 12:11
# }

# Colzes, canells
BENCH_PRESS_INDICES = [7, 8, 9, 10]
PULL_UP_INDICES = [5, 6, 9, 10]

# Canells, peus
DEADLIFT_INDICES = [9, 10, 15, 16]

# Espatlles, genolls
SQUAT_INDICES = [5, 6, 13, 14]


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