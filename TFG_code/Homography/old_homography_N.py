import numpy as np
import cv2 # type: ignore
import os
import json

LABEL_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels'
OUTPUT_IMAGE_DIR = '/datatmp2/joan/tfg_joan/results/homography_2'
CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

# Mapeo COCO 17: intercambiar índices left<->right
# COCO order: 0:nose,1:l_eye,2:r_eye,3:l_ear,4:r_ear,5:l_sh,6:r_sh,7:l_elb,8:r_elb,9:l_wri,10:r_wri,
# 11:l_hip,12:r_hip,13:l_knee,14:r_knee,15:l_ankle,16:r_ankle

"""

Calcula pose de referència per a cada exercici (pose més representativa)
Després calcula homografia per a cada vídeo, per ajustar els keypoints
de cada imatge a aquesta pose de referència
Calcula la transformació a tots els frames del vídeo i escolleix per a tots
la que té menys error de reprojecció
També es comprova si el vídeo està horitzontalment simètric, i es
volteja si cal

"""
BENCH_PRESS_DISTR = np.array([[0.486,0.372],[0.615,0.435],[0.388, 0.393],[0.4823,0.5023]], dtype=np.float32)
PULL_UP_DISTR = np.array([[0.4733, 0.4166],[0.548,0.4282],[0.483,0.5984],[0.5338,0.604]], dtype=np.float32)
DEADLIFT_DISTR = np.array([[0.4811,0.32],[0.55,0.3125],[0.4915,0.4594],[0.5384,0.458]], dtype=np.float32)
SQUAT_DISTR = np.array([[0.3867,0.229],[0.472,0.229],[0.4199,0.539],[0.513,0.539]], dtype=np.float32)

LEFT_RIGHT_MAP = {
    1:2, 2:1, 3:4, 4:3,
    5:6, 6:5, 7:8, 8:7,
    9:10, 10:9, 11:12, 12:11
}

TRUNK_INDICES = [5, 6, 11, 12]

EPS = 1e-9

def find_centroid(kps):
    """
    Troba centre amb ubicació de espatlles per normalitzar els keypoints entre imatges
    """
    return ((kps[5] + kps[6]) / 2.0).astype(np.float32)  # Centre entre espatlles

def find_length(kps):
    # print(kps.shape)
    return float(np.linalg.norm(kps[5] - kps[6]))  # Distància entre espatlles

def find_norm_matrix(kps, center_base, length_base, min_scaling = 0.6, max_scale = 20):
    """
    Troba matriu de normalització per escalar i centrar els keypoints
    Es fica un màxim i un mínim per evitar transformacions massa grans
    """
    
    center_kps = find_centroid(kps)
    length_kps = find_length(kps)

    s = length_base / max(length_kps, EPS)
    s = float(np.clip(s, min_scaling, max_scale))

    N = np.array([[s, 0.0, center_base[0] - s * center_kps[0]],
                  [0.0, s, center_base[1] - s * center_kps[1]],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    return N

def mirror_keypoints(kps, img_width):
    
    kps_mirrored = kps.copy()
    
    kps_mirrored[:, 0] = img_width - 1 - kps[:, 0]
    kps_mirrored_copy = kps_mirrored.copy()
    
    for left, right in LEFT_RIGHT_MAP.items():
        kps_mirrored[left] = kps_mirrored_copy[right]
        kps_mirrored[right] = kps_mirrored_copy[left]
        
    return kps_mirrored

def build_base_pose(reference_kps_list, method='reference'):
    """
        Es construeix pose de referència per a realitzar la homografia
        de les següents imatges
    """
    
    num_frames = reference_kps_list.shape[0]
    
    if method == 'reference':
        distances = np.zeros(num_frames)
        for i in range(num_frames):
            dists_i = np.linalg.norm(reference_kps_list - reference_kps_list[i], axis=(1,2))
            distances[i] = np.mean(dists_i)
        best_frame = int(np.argmin(distances))
        kps_base = reference_kps_list[best_frame]
    elif method == 'random':
        frame_idx = np.random.randint(0, num_frames)
        kps_base = reference_kps_list[frame_idx]
    else:
        raise ValueError("Mètode desconegut per a la construcció de la pose base: {}".format(method))

    return kps_base

#Transforma imatge i keypoints cap a la base de referència
def warp_keypoints_video(kp_frames, kps_base_norm, center_base, length_base, img_width, img_height, method='homography'):

    frame_number = kp_frames.shape[0]
    errors = []
    possible_M = {}
    
    for j in range(frame_number):
        kps = kp_frames[j]
        
        N = find_norm_matrix(kps, center_base, length_base)

        kps_homogeneous = np.hstack([kps, np.ones((kps.shape[0], 1))])
        kps_norm_h = (N @ kps_homogeneous.T).T
        kps_norm = kps_norm_h[:, :2] / kps_norm_h[:, 2:]
    
        if method == 'homography':
            H_kps, mask = cv2.findHomography(kps_norm, kps_base, cv2.RANSAC, 10.0)

            if H_kps is None:
                return None
        elif method == 'affine':
            M, inliers = cv2.estimateAffinePartial2D(kps_norm, kps_base, method=cv2.RANSAC, ransacReprojThreshold=10.0)
            H_kps = np.vstack([M, [0.0,0.0,1.0]])
        
        else: 
            raise ValueError("Mètode desconegut per a la transformació: {}".format(method))

        M_total = H_kps @ N
        
        errors_per_frame = []
        valid = True
        
        for i in range(frame_number):
            kps_i = kp_frames[i]

            #Afegir coordenada homogènia
            kps_homogeneous = np.hstack([kps_i, np.ones((kps_i.shape[0], 1))])
            
            warped_kps = (M_total @ kps_homogeneous.T).T
            warped_kps = warped_kps[:, :2] / warped_kps[:, 2:]

            d = np.linalg.norm(warped_kps - kps_base_norm, axis=1)  # (13,)
            if np.any(np.isnan(d)) or np.any(np.isinf(d)):
                valid = False
                break
            errors_per_frame.append(np.mean(d))
            
            avg_error = float(np.median(errors_per_frame))
            
            errors.append((int(j), avg_error))
            possible_M[j] = M_total

        if not valid:
            continue
        
    errors.sort(key=lambda x: x[1])
    best_frame_idx = errors[0][0]
    best_M = possible_M[best_frame_idx]

    return best_M.astype(np.float32), int(best_frame_idx), errors

#Com hi ha vídeos horitzontalment simètrics, determina si voltejar-los o no
def determine_video_flip(kps, img, img_width, kps_base):
    
    kps_mirrored = mirror_keypoints(kps, img_width)
    
    dist_original = np.linalg.norm(kps - kps_base, axis=1).sum()
    dist_mirrored = np.linalg.norm(kps_mirrored - kps_base, axis=1).sum()
    
    if (dist_mirrored + EPS) < dist_original:
        return True
    else:
        return False

if __name__ == "__main__":
    
    for class_name in CLASSES:
        print('CLASS:', class_name)
        
        image_list = []
        kp_list = []
        output_path_list = []
        
        for sub_dir, _, labels in sorted(os.walk(os.path.join(LABEL_DIR, class_name))):
            
            if labels == []:
                continue
            
            image_list_video = []
            kp_list_video = []
            output_path_list_video = []
            
            for label in sorted(labels):
                label_path = os.path.join(sub_dir, label)
                image_path = label_path.replace('labels', 'images').replace('.json', '.jpg')
                output_image_path = os.path.join(OUTPUT_IMAGE_DIR, class_name, os.path.basename(sub_dir), label.replace('.json', '_warped.jpg'))
                # print(output_image_path)
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                save_more = True
                # print(label_path)
                # print(image_path)
                
                with open(label_path, 'r') as f:
                            
                    data = json.load(f)
                    
                    instance_info = data.get('instance_info', {})
                    
                    if not instance_info or 'keypoints' not in instance_info[0]:
                        continue
                    
                    keypoints = instance_info[0]['keypoints']
                    
                    if keypoints is None or keypoints == np.zeros((13,2)).tolist():
                        continue
                    
                    #Eliminar cames dels keypoints
                    if len(keypoints) > 13:
                        keypoints = [keypoints[i] for i in range(13)]
                    
                    keypoints = np.array(keypoints)
                    
                img = cv2.imread(image_path)

                image_list_video.append(img)
                kp_list_video.append(keypoints)
                output_path_list_video.append(output_image_path)
            
            image_list.append(image_list_video)
            kp_list.append(kp_list_video)
            output_path_list.append(output_path_list_video)

        kps_base = build_base_pose(np.array([video_kp_list[0] for video_kp_list in kp_list]), method='reference')

        # print(kps_base.shape)

        length_base = find_length(np.array(kps_base))
        centers_all = np.array([find_centroid(k) for video in kp_list for k in video])
        center_base = np.median(centers_all, axis=0)

        
        N_base = find_norm_matrix(kps_base, center_base, length_base)
        kps_base_homogeneous = np.hstack([kps_base, np.ones((kps_base.shape[0], 1))])   
        kps_base_norm_h = (N_base @ kps_base_homogeneous.T).T
        kps_base_norm = kps_base_norm_h[:, :2] / kps_base_norm_h[:, 2:]

        for video_kp_list, video_img_list, video_output_path_list in zip(kp_list, image_list, output_path_list):
            
            M_best, best_frame, errors = warp_keypoints_video(
                np.array(video_kp_list), kps_base_norm, center_base, length_base,
                video_img_list[0].shape[1], video_img_list[0].shape[0], method='homography')
            
            if M_best is None:
                    continue
            
            A = M_best[:2, :2].astype(np.float32)
            t = M_best[:2, 2].astype(np.float32)

            c = (video_img_list[0].shape[0]/2, video_img_list[0].shape[1]/2)
            
            p = (A @ c) + t
            
            U, Svals, Vt = np.linalg.svd(A)
            scale = float((Svals[0] + Svals[1]) / 2.0)
            
            A_norot = np.array([[scale, 0.0],
                    [0.0, scale]], dtype=np.float32)

            # calcular nueva traslación t_norot para que c -> p siga siendo cierto:
            t_norot = p - (A_norot @ c)
                        
            M_modified = np.vstack([np.hstack([A_norot, t_norot.reshape(2,1)]), [0.0, 0.0, 1.0]]).astype(np.float32)
            
            flip_check = False

            for img, kps, output_path in zip(video_img_list, video_kp_list, video_output_path_list):
                
                current_trunk = kps[TRUNK_INDICES]

                if flip_check == False:
                    flip = determine_video_flip(kps, img, video_img_list[0].shape[1], kps_base)
                    flip_check = True

                if flip:
                    kps = mirror_keypoints(kps, img.shape[1])
                    img = cv2.flip(img, 1)

                kps_homogeneous = np.hstack([kps, np.ones((kps.shape[0], 1))])
                warped_kps = (M_modified @ kps_homogeneous.T).T
                warped_kps = warped_kps[:, :2] / warped_kps[:, 2:]
                
                img_warped = img.copy()

                img_warped = cv2.warpAffine(img, M_modified[:2, :], (img.shape[1], img.shape[0]))

                for kp in warped_kps:
                    img_warped = cv2.circle(img_warped, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

                cv2.imwrite(output_path, img_warped)
                
        print('COMPLETED', class_name)