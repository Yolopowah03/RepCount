from ultralytics import YOLO #type: ignore
import os
import cv2 as cv #type: ignore
import numpy as np
import json
import time

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

def yolo_main(args):
    
    begin_time = time.time()
    
    model = YOLO(args['model_path'])
    
    cap = cv.VideoCapture(args['input_video_path'])
    
    success, frame = cap.read()
    i = 0
    total_resulting_kpts = []
    while success:

        file = f"{os.path.splitext(os.path.basename(args['input_video_path']))[0]}_frame_{i:06d}.jpg"
        
        save_path = os.path.join(args['output_dir'], file)
        save_path_json = save_path.replace(".jpg", ".json").replace(".png", ".json")
        
        print(save_path)
        
        results = model(frame)  # predict on an image
        
        max_score = 0
        for result in results:
            
            kpts = result.keypoints.xy  # x and y coordinates
            scores = result.keypoints.conf  # confidence scores
            kpts = kpts.cpu()
            scores = scores.cpu()
            mean_score = np.mean(scores.tolist())
            
            if mean_score >= max_score:
                resulting_kpts = np.asarray(kpts, dtype=np.float32)
                resulting_scores = scores.tolist()
                max_score = mean_score
        
        print(resulting_kpts.shape)
        save_img = draw_keypoints(frame, resulting_kpts[0], args['kpts_colors'], args['skeleton_info'])
        save_json(resulting_kpts[0], resulting_scores, save_path_json, args['kpts_colors'], args['kpt_thr'])

        cv.imwrite(save_path, save_img)
        i += 1
        
        total_resulting_kpts.append(resulting_kpts[0])
        success, frame = cap.read()
    
    cap.release()

    return {'total_time': time.time()-begin_time, 'resulting_kpts': total_resulting_kpts}