import numpy as np
import cv2 # type: ignore
import os
import json

# Mapeo COCO 17: intercambiar índices left<->right
# COCO order: 0:nose,1:l_eye,2:r_eye,3:l_ear,4:r_ear,5:l_sh,6:r_sh,7:l_elb,8:r_elb,9:l_wri,10:r_wri,
# 11:l_hip,12:r_hip

BENCH_PRESS_DISTR = np.array([[0.486,0.372],[0.615,0.435],[0.388, 0.393],[0.4823,0.5023]], dtype=np.float32)
PULL_UP_DISTR = np.array([[0.4733, 0.4166],[0.548,0.4282],[0.483,0.5984],[0.5338,0.604]], dtype=np.float32)
DEADLIFT_DISTR = np.array([[0.4811,0.32],[0.55,0.3125],[0.4915,0.4594],[0.5384,0.458]], dtype=np.float32)
SQUAT_DISTR = np.array([[0.3867,0.229],[0.472,0.229],[0.4199,0.539],[0.513,0.539]], dtype=np.float32)

LEFT_RIGHT_MAP = {
    0:1, 1:0, 2:3, 3:2
}

TRUNK_INDICES = [5, 6, 11, 12]

"""

Calcula pose de referència per a cada exercici (pose més representativa)
Després calcula homografia per a cada vídeo, per ajustar els keypoints
de cada imatge a aquesta pose de referència
Calcula la transformació a tots els frames del vídeo i escolleix per a tots
la que té menys error de reprojecció
També es comprova si el vídeo està horitzontalment simètric, i es
volteja si cal

"""

EPS = 1e-9


def mirror_keypoints(kps, img_width):
    
    kps_mirrored = kps.copy()
    
    kps_mirrored[:, 0] = img_width - 1 - kps[:, 0]
    kps_mirrored_copy = kps_mirrored.copy()
    
    for left, right in LEFT_RIGHT_MAP.items():
        kps_mirrored[left] = kps_mirrored_copy[right]
        kps_mirrored[right] = kps_mirrored_copy[left]
        
    return kps_mirrored


#Com hi ha vídeos horitzontalment simètrics, determina si voltejar-los o no
def determine_video_flip(kps, img_width, kps_base):
    
    kps_mirrored = mirror_keypoints(kps, img_width)
    
    dist_original = np.linalg.norm(kps - kps_base, axis=1).sum()
    dist_mirrored = np.linalg.norm(kps_mirrored - kps_base, axis=1).sum()
    
    if (dist_mirrored + EPS) < dist_original:
        return True
    else:
        return False

def homography_main(args):
    
    match(args['class_name']):
        case 'bench_press':
            kp_distr_norm = BENCH_PRESS_DISTR
        case 'pull_up':
            kp_distr_norm = PULL_UP_DISTR
        case 'deadlift':
            kp_distr_norm = DEADLIFT_DISTR
        case 'squat':
            kp_distr_norm = SQUAT_DISTR
          
    example_trunk = np.array([args['keypoints'][0][i] for i in TRUNK_INDICES], dtype=np.float32)
    
    kp_distr = []
    for kp in kp_distr_norm:
        kp_distr.append([kp[0] * args['img_shape'][1], kp[1] * args['img_shape'][0]])

    kp_distr = np.array(kp_distr, dtype=np.float32)

    flip = determine_video_flip(example_trunk, args['img_shape'][1], kp_distr)

    H_torso, inliers = cv2.estimateAffinePartial2D(example_trunk, kp_distr, method=cv2.RANSAC, ransacReprojThreshold=10.0)

    if H_torso is None:
        raise ValueError("Homography could not be computed.")

    A = H_torso[:2, :2].astype(np.float32)
    t = H_torso[:2, 2].astype(np.float32)

    c = (args['img_shape'][1]/2, args['img_shape'][0]/2)
    
    p = (A @ c) + t
    
    U, Svals, Vt = np.linalg.svd(A)
    scale = float((Svals[0] + Svals[1]) / 2.0)
    
    A_norot = np.array([[scale, 0.0],
            [0.0, scale]], dtype=np.float32)

    # calcular nueva traslación t_norot para que c -> p siga siendo cierto:
    t_norot = p - (A_norot @ c)
                
    M_modified = np.vstack([np.hstack([A_norot, t_norot.reshape(2,1)]), [0.0, 0.0, 1.0]]).astype(np.float32)

    for kps in args['keypoints']:
        
        if flip:
            kps = mirror_keypoints(kps, args['img_shape'][1])

        kps_homogeneous = np.hstack([kps, np.ones((kps.shape[0], 1))])
        warped_kps = (M_modified @ kps_homogeneous.T).T
        corrected_kpts = warped_kps[:, :2]
        
        
    return corrected_kpts