from Generation.utils import video_utils
from Generation.utils.head_pose import get_frontal_frame
from Generation.utils.scene_detection import detect_scenes_and_assign_ids
from AnonHead.segment_multiple import Segment
import os
import sys
import random
import logging
import cv2
from typing import List, Dict, Any
from deepface import DeepFace
from PIL import Image
import argparse


def clip_creation(args):
    """
    Detects scenes in the driving video and for each scene:
      - Extracts a frame at the scene start (as a BGR np.ndarray).
      - Passes that BGR array directly to DeepFace.extract_faces.
      - If >1 face is detected, marks the scene as '_noface'.
      - If 0 faces are detected, proceeds without face recognition.
      - If exactly 1 face is detected, extracts the subvideo and compares
        the extracted face against known faces using DeepFace.verify.
      - For the one-face case, assigns (or reuses) a random seed and renames
        the first-frame and subvideo files to include that seed.
      
    Returns:
        scene_vid_paths: A list of file paths for the saved first frame images.
    """
    os.makedirs(args.save_folder, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(args.driving_path))[0]
    
    segment = Segment(load_seg=False, load_face=False) # no need for segmentation

    print("Detecting scenes in driving video...")
    scenes = detect_scenes_and_assign_ids(
        args.driving_path,
        threshold=args.scene_threshold,
        similarity_threshold=args.scene_similarity_threshold
    )
    unique_ids = set(s["scene_id"] for s in scenes)
    print(f"Detected {len(scenes)} scene segments in total.")
    print(f"[DEBUG] UNIQUE SCENE IDs = {sorted(unique_ids)} (TOTAL: {len(unique_ids)})")
    for i, scene_info in enumerate(scenes):
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        print(f"[DEBUG] scene_index={i} => scene_id={sid}, start={start}, end={end}")
    if args.max_len != None:
        print(f"Only first {args.max_len} seconds will be used!")
        
    if len(scenes)==0: return None
    
    temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
    os.makedirs(temp_driving_dir, exist_ok=True)
    temp_source_dir = os.path.join(args.save_folder, "temp_source")
    os.makedirs(temp_source_dir, exist_ok=True)

    scene_video_paths = []  # list of tuples: (scene_id, scene_video_path, ff_filename)
    unique_faces = {}       # dictionary mapping seed (int) -> path to known face image

    for i, scene_info in enumerate(scenes):
        if args.max_len == "Limit":
            break
        sid = scene_info["scene_id"]
        start, end = scene_info["start"], scene_info["end"]
        if args.max_len != None and scene_info["end"] > args.max_len:
            scene_info["end"] = args.max_len
            args.max_len = "Limit"
            
        frame = get_frontal_frame(args.driving_path, start, end, best=True)
        # --- Run face extraction on the frame directly (BGR np.ndarray) ---
        extracted_faces = segment.retinaface_detect_and_annotate(img = frame, method="face", only_boxes=True)
        faces = []
        for face in extracted_faces:
            x1, x2, y1, y2 = face
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            crop = frame[y1:y2, x1:x2].copy() 
            faces.append(crop)
            
        
        # --- Process based on the number of detected faces ---
        if len(faces) > 1:
            print(f"More than one face detected in scene {i} (scene id {sid}). ")
            continue
        
        elif len(faces) == 0:
            print(f"No face detected in scene {i} (scene id {sid}). Proceeding without face recognition.")
            scene_vid_path = os.path.join(temp_driving_dir, f"{file_name}_scene_{i}_id_{sid}.mp4")
            video_utils.extract_subvideo(args.driving_path, scene_vid_path, start, end)
            ff_filename = os.path.join(temp_source_dir, f"{file_name}_ff_{sid}_noface.png" )
            cv2.imwrite(ff_filename, frame)
            scene_video_paths.append((i, sid, scene_vid_path, ff_filename, extracted_faces))
        else:
            print(f"Scene {i}: {len(faces)} face detected (scene id {sid}).")
            scene_vid_path = os.path.join(temp_driving_dir, f"{file_name}_scene_{i}_id_{sid}.mp4")
            video_utils.extract_subvideo(args.driving_path, scene_vid_path, start, end)
            face_np = faces[0]

            # Compare against known unique faces
            used_seed = None
            if len(unique_faces) == 0:
                attributes = DeepFace.analyze(
                    img_path = face_np, 
                    actions = ['age', 'gender', 'race', 'emotion'],
                    detector_backend = 'skip',
                    enforce_detection = False
                    )[0]
                new_seed = random.randint(1000, 9999)
                used_seed = new_seed
                unique_faces[used_seed] = (face_np,attributes)
                print(f"Scene {i}: New unique face added with seed {used_seed}.")
                frame_path = os.path.join(temp_source_dir, f"{file_name}_ff_{sid}_{used_seed}.png")
                cv2.imwrite(frame_path, frame)   
                scene_video_paths.append((i, sid, scene_vid_path, frame_path, extracted_faces))             
                continue
                
            for seed, (face_emb, attributes) in unique_faces.items():
                # result = DeepFace.verify(
                #     img1_path=face_np,
                #     img2_path=face_emb,
                #     model_name="Facenet512",
                #     detector_backend="skip",
                #     distance_metric="cosine",
                #     enforce_detection=False,
                #     threshold=5e-5
                # )
                result = DeepFace.verify(
                    img1_path=face_np,
                    img2_path=face_emb,
                    model_name="VGG-Face",
                    detector_backend="skip",
                    distance_metric="cosine",
                    enforce_detection=False
                )
                
                
                if result["verified"]:
                    used_seed = seed
                    print(f"Scene {i}: Face matches unique face with seed {seed}.")
                    frame_path = os.path.join(temp_source_dir, f"{file_name}_ff_{sid}_{used_seed}.png")
                    cv2.imwrite(frame_path, frame)
                    scene_video_paths.append((i, sid, scene_vid_path, frame_path, extracted_faces))  
                    break
            if used_seed is None:
                attributes = DeepFace.analyze(
                    img_path = face_np, 
                    actions = ['age', 'gender', 'race', 'emotion'],
                    detector_backend = 'skip',
                    enforce_detection = False
                    )[0]
                new_seed = random.randint(1000, 9999)
                used_seed = new_seed
                unique_faces[used_seed] = (face_np,attributes)
                print(f"Scene {i}: New unique face added with seed {used_seed}.")
                frame_path = os.path.join(temp_source_dir, f"{file_name}_ff_{sid}_{used_seed}.png")
                cv2.imwrite(frame_path, frame)
                scene_video_paths.append((i, sid, scene_vid_path, frame_path, extracted_faces))  
                
            print(f'Number of Unique Faces: {len(unique_faces)}')


    return scene_video_paths, unique_faces
