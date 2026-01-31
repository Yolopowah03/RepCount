from ultralytics import YOLO #type: ignore
import os
import cv2 as cv #type: ignore
import numpy as np
import json
import time

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def draw_keypoints(image, kpts, kpt_colors, skeleton_info, kpt_thr=0.3, radius=3, thickness=2):
    
    for kid, kpt in sorted(enumerate(kpts)):

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

def save_json(keypoints, scores, output_path, kpt_colors, kpt_thr,):

    pred_save_path = output_path.replace(".jpg", ".json").replace(".png", ".json")

    with open(pred_save_path, "w") as f:
        json.dump(
            dict(
                instance_info=[
                    {
                        "keypoints": keypoints.tolist(),
                        "keypoint_scores": scores,
                    }
                ]
            ),
            f,
            indent="\t",
        )

def yolo_mod(args):
    
    begin_time = time.perf_counter()
    
    model = YOLO(args['model_path'])
    
    cap = cv.VideoCapture(args['input_video_path'])
    
    success, frame = cap.read()
    i = 0
    total_resulting_kpts = []
    while success:

        if i % args['skip_frames'] != 1:
            file = f"{os.path.splitext(os.path.basename(args['input_video_path']))[0]}_frame_{i:06d}.jpg"
            
            if args['output_dir'] is not None:
                save_path = os.path.join(args['output_dir'], file)
                save_path_json = save_path.replace(".jpg", ".json").replace(".png", ".json")
            
            results = model(frame, verbose=False)  # predict on an image
            
            max_area = 0.0
            max_score = 0.0
            
            for result in results:
                
                if result.boxes and len(result.boxes) > 0:
                    xywh = result.boxes.xywh.cpu().numpy().squeeze()
                    if len(xywh.shape) == 1:
                        area = xywh[2] * xywh[3]
                    elif len(xywh.shape) > 1:
                        area = max(xywh[:, 2] * xywh[:, 3])
                        result = result[np.argmax(xywh[:, 2] * xywh[:, 3])]
                    
                    if area > max_area:
                        kpts = result.keypoints.xy
                        scores = result.keypoints.conf
                        
                        kpts = kpts.cpu()
                        scores = scores.cpu()

                        resulting_kpts = np.asarray(kpts, dtype=np.float32)
                        resulting_scores = scores.tolist()
                        max_area = area
                        
                        score = np.mean([s for s in resulting_scores if s is not None])
                        max_score = score
                    
                if max_score < args['min_conf']:
                    print('No keypoints detected in frame, skipping', i)
                    resulting_kpts = np.zeros((1, args['n_keypoints'], 2), dtype=np.float32)
                    resulting_scores = [0.0 for _ in range(args['n_keypoints'])]
        
            save_img = draw_keypoints(frame, resulting_kpts[0], args['kpts_colors'], args['skeleton_info'])
            
            if args['save_json'] or args['save_frames']:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))

            if args['save_json']:
                save_json(resulting_kpts[0], resulting_scores, save_path_json, args['kpts_colors'], args['kpt_thr'])

            if args['save_frames']:
                cv.imwrite(save_path, save_img)
            
            total_resulting_kpts.append(resulting_kpts[0][0:args['n_keypoints']])
        
        i += 1
        success, frame = cap.read()
    
    cap.release()

    return {'total_time': time.perf_counter()-begin_time, 'resulting_kpts': total_resulting_kpts}