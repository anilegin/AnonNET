import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2
import shutil
import gc 

from AnonHead.predict_multiple import Predictor
from AnonHead.segment_multiple import Segment



def main(args):
    
    file_name = os.path.splitext(os.path.basename(args.image))[0]
    
    if args.segment == "face":  
        segment = Segment(load_det=False, load_seg=False)    
        
    else:
        segment = Segment(load_det=False, load_face=False) 
    
    
            
    #Segment the face
    outputs = segment.retinaface_detect_and_annotate(
        img=args.image,
        margin=args.margin,
        method=args.segment,
    )
    
    
    if len(outputs) == 0:
        raise ValueError("No Person detected! Anonymization aborted!")
    
    image_pil = Image.open(args.image)
    
    del segment
    torch.cuda.empty_cache()
    gc.collect()
    
    predictor = Predictor()  

    conf = {
        "image": image_pil,
        "prompt": args.prompt,
        "mask": None,
        "negative_prompt": args.negative_prompt,
        "strength": args.strength,
        "max_height": args.max_height,
        "max_width": args.max_width,
        "steps": args.steps,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "im_path": args.image
    }
    
    
    generated = []
    for i, (mask_img, seg_img, bbox) in enumerate(outputs):
        
        conf["mask"] = mask_img
        conf["image"] = seg_img

        crop, dist, attrs = predictor.anonymize(**conf)
        if crop is None:
            continue
        
        # Merge the new face back in
        resized_mask = mask_img.resize(seg_img.size, Image.LANCZOS)
        resized_crop = crop.resize(seg_img.size, Image.LANCZOS)
        generated.append((resized_mask, resized_crop, bbox))
        
    segment = Segment(load_det=False, load_face=False, load_seg=False) 
    out_image = segment.merge_crops(image_pil, generated)

    out_image = out_image.resize(image_pil.size, Image.LANCZOS)
    
    if len(outputs) == 1:
        dist_str = f"{dist:.3f}"
        attrs_str = "_".join(a.replace(" ", "-") for a in attrs)
        filename_info = f"{dist_str}_{attrs_str}" if attrs_str else dist_str
        save_p = os.path.join(args.save_folder, f"{file_name}_anonymized_{filename_info}.png")
        
    else:
        save_p = os.path.join(args.save_folder, f"{file_name}_anonymized_{len(outputs)}_person.png")
        
    os.makedirs(args.save_folder, exist_ok=True)
    out_image.save(save_p)
    print(f"Image has been saved to {save_p}")

        
        

        
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #anonymize
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--strength", type=float, nargs="+", default=[0.9, 0.4, 0.3],
                    help="Exactly one or three values corresponds [face mask, lineart, openpose] (e.g. --strength 0.9 0.4 0.3)")
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--segment", type=str, choices=['head', 'face'], default='face')
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--save_folder", type=str, default="./results")

    args = parser.parse_args()
    
    if len(args.strength) not in [1,3]:
        raise ValueError("You must provide either 1 or 3 strength values.")
    
    main(args)


