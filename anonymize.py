import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2
import shutil
import traceback
import gc 
import sys


from scene_clips import clip_creation
from Generation.gen_lia import main as motion_lia
from Generation.gen_lp import main as motion_lp
from Generation.src.config.argument_config import ArgumentConfig
from AnonHead.predict_multiple import Predictor
from AnonHead.segment_multiple import Segment



def main(args):
    
    file_name = os.path.splitext(os.path.basename(args.driving_path))[0]
    try:
        scene_paths, attributes = clip_creation(args)
        
    except Exception as e:
        print(f"[ERROR] Failed to load model weights from {model_path}")
        traceback.print_exc()
        
    if len(scene_paths) == 0:
        print("No suitable scene found!")
        sys.exit()
    
    
    predictor = Predictor()  
    segment = Segment(load_det=False, load_seg=False)      
    prompts = {}
    
    scenes_info = []
    
    for i, (scene_number, sid, vid_path, im_path, bboxes) in enumerate(scene_paths):
        
        basename = os.path.splitext(os.path.basename(im_path))[0]
        
        # Check if the filename indicates no face was detected
        if "noface" in basename:
            print(f"Scene {i}: no face detected. Copying the frame without anonymization.")
            image = Image.open(im_path)
            os.makedirs(os.path.join(args.save_folder, "source_images"), exist_ok=True)
            image.save(os.path.join(args.save_folder, "source_images", f'{file_name}_source_{i}_noface.png'))
            continue
            
        #Segment the face
        
        outputs = segment.retinaface_detect_and_annotate(
            img=im_path,
            margin=0.8,
            method=args.segment,
            bboxes = bboxes,
        )
        
        if len(outputs) == 0:
            print(f"Scene {i}: no face detected. Copying the frame without anonymization.")
            image = Image.open(im_path)
            os.makedirs(os.path.join(args.save_folder, "source_images"), exist_ok=True)
            image.save(os.path.join(args.save_folder, "source_images", f"{file_name}_source_{i}_noface.png"))
            continue
        
        if len(outputs) > 1:
            print(f"Scene {i}: more than 1 face detected. Skipping!")
            continue
        
        
        
        # Assuming filename is like: ff_{sid}_{seed}.png
        seed = basename.split('_')[-1]
        try:
            seed_val = int(seed)
        except Exception:
            seed_val = 6000  # fallback if conversion fails
            
        print(f"Scene {i}: using seed {seed_val}.")
        
        if not seed_val in prompts:
            prompts[seed_val], _ = predictor.create_prompt(im_path, deepface_result=attributes[seed_val][1])
        
        conf = {
            "image": outputs[0][1],
            "prompt": prompts[seed_val],
            "mask": outputs[0][0],
            "negative_prompt": args.negative_prompt,
            "strength": [0.9, 0.4, 0.3],
            "max_height": args.max_height,
            "max_width": args.max_width,
            "steps": 35,
            "seed": seed_val,
            "guidance_scale": 8.0,
            "im_path": im_path
        }
        
        img = Image.open(im_path)
        
        crop, dist, attrs = predictor.anonymize(**conf)
        
        crop = crop.resize(outputs[0][1].size, Image.LANCZOS)
        
        out_image = segment.merge_crops(img, [(outputs[0][0], crop, outputs[0][2])])
        
        os.makedirs(os.path.join(args.save_folder, "source_images"), exist_ok=True)
        out_path= os.path.join(args.save_folder, "source_images", f"{file_name}_source_{i}.png")
    
        out_image.save(out_path)
        
        scenes_info.append((scene_number, sid, vid_path, out_path))
        
        
    del segment
    del predictor
    torch.cuda.empty_cache()
    gc.collect()
            
            
    if args.motion == 'lia':
        motion_lia(scenes = scenes_info,
                   file_name = file_name,
                   save_folder = args.save_folder)
        
    else:
        
        motion_lp(
            scenes = scenes_info,
            file_name = file_name,
            stitch = args.no_stitch,
            save_folder = args.save_folder
        )
        
    
    def remove_contents(folder):
        """Remove all files and folders inside the given folder."""
        for entry in os.listdir(folder):
            path = os.path.join(folder, entry)
            try:
                if os.path.isfile(path) or os.path.islink(path):
                    os.unlink(path)  # Remove the file or link
                elif os.path.isdir(path):
                    shutil.rmtree(path)  # Remove the directory and its contents
            except Exception as e:
                print(f"Failed to delete {path}. Reason: {e}")
                

    if args.cache:
        temp_driving_dir = os.path.join(args.save_folder, "temp_driving")
        temp_source_dir = os.path.join(args.save_folder, "temp_source")
        remove_contents(temp_driving_dir)
        remove_contents(temp_source_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #anonymize
    parser.add_argument("--driving_path", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--segment", type=str, choices=['head', 'face'], default='face')
    parser.add_argument("--cache", action='store_false', default=True, help='no save of temp files')
    #generate
    parser.add_argument("--save_folder", type=str, default="./results")
    parser.add_argument("--no_stitch", action='store_false', default=True, help='no stitching, use if head movement is a lot')
    parser.add_argument("--max_len", type=int, default=None, help="Max Duration of the video")
    parser.add_argument("--scene_threshold", type=float, default=0.2) #old one 0.2
    parser.add_argument("--scene_similarity_threshold", type=float, default=2.0)
    #motion model 
    parser.add_argument("--motion", type=str, choices=['lia', 'lp'], default='lp')
    args = parser.parse_args()
    
    main(args)


