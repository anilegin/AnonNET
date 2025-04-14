import os
import torch
import numpy as np
from PIL import Image
import argparse
import cv2

from predict_multiple import Predictor
from segment_multiple import Segment

def main(image_path, prompt, negative_prompt, strength, max_height, max_width, steps, seed, no_force, guidance_scale, out_path, out_mask):
    """
    Example CLI usage (very barebones):

    python predictor.py \
      --image /path/to/input.jpg \
      --mask /path/to/mask.png \
      --prompt "A tabby cat on a bench" \
      --negative_prompt "deformed, ugly" \
      --strength 0.8 \
      --max_height 612 \
      --max_width 612 \
      --steps 20 \
      --seed 42 \
      --guidance_scale 10
    """

    im_name = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_path, f"{im_name}_out.png")
    out_mask = os.path.join(out_mask, f"{im_name}_mask.png")
    
    image = Image.open(image_path)
    
    print('Segmentation Started!')
    segment = Segment()
    outputs = segment.yolo_detect_and_annotate(img=image_path, force=no_force)
    print('Segmentation Finished!')
    for i in range(len(outputs)):
        outputs[i][0].save(f"./.cache/{i}_mmask.png")
        outputs[i][1].save(f"./.cache/{i}_ccrop.png")
    
    conf = {
        "image": image,
        "prompt": prompt,
        "mask": None,
        "negative_prompt": negative_prompt,
        "strength": strength,
        "max_height": max_height,
        "max_width": max_width,
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "out_path": out_path,
        "im_path": image_path
    }
    
    generated = []
    print('Segments Anonymization started!')
    predictor = Predictor()
    for i in range(len(outputs)):
        conf['mask'] = outputs[i][0]
        conf['image'] = outputs[i][1]
        
        crop = predictor.anonymize(**conf)
        # resize generated crop
        x1, x2, y1, y2 = outputs[i][2]
        # Retrieve the original width and height from the bounding box
        width = x2 - x1
        height = y2 - y1
        # Resize the cropped image to its original bounding box size
        resized_crop = crop.resize((width, height), Image.LANCZOS)
        
        generated.append((outputs[i][0], resized_crop, outputs[i][2]))
    
    final_image = segment.merge_crops(image, generated)
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f'Image is saved to ./res/final_{file_name}.png')
    #final_image.save(f'./res/final_{file_name}.png')
    return final_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--max_height", type=int, default=612)
    parser.add_argument("--max_width", type=int, default=612)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_force", action='store_false', default=True, help='Try RetinaFace in case YOLO fails')
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--out_path", type=str, default="./res_old")
    parser.add_argument("--out_mask", type=str, default='./masks')
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with individual arguments instead of 'args'
    final_image = main(
                    image_path=args.image,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    strength=args.strength,
                    max_height=args.max_height,
                    max_width=args.max_width,
                    steps=args.steps,
                    seed=args.seed,
                    no_force=args.no_force,
                    guidance_scale=args.guidance_scale,
                    out_path=args.out_path,
                    out_mask=args.out_mask
                )
