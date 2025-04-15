import torch
import torch.nn as nn
from .networks.generator import Generator
from .utils.video_utils import create_concat_video
import argparse
import numpy as np
import torchvision
import os
import tyro
import subprocess
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import glob
import math
import cv2
import gc
import traceback
import re

    

def load_image(filename, size):
    """Load and preprocess single image."""
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))
    return img / 255.0

def img_preprocessing(img_path, size):
    """Convert image to normalized tensor."""
    img = load_image(img_path, size)
    img = torch.from_numpy(img).unsqueeze(0).float()
    return (img - 0.5) * 2.0

def vid_preprocessing(vid_path, size=256):
    """OpenCV-based video preprocessing with robust error handling."""
    try:
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {vid_path}")
        
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            frame_norm = (frame_resized.astype(np.float32) / 127.5) - 1.0
            frames.append(np.transpose(frame_norm, (2, 0, 1)))
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames read from video: {vid_path}")
        
        vid_tensor = torch.from_numpy(np.stack(frames)).unsqueeze(0).float()
        return vid_tensor.cuda(), fps
        
    except Exception as e:
        print(f"Error in cv2 video processing: {str(e)}")
        raise

def save_video(vid_target_recon, save_path, fps):
    # Convert the PyTorch tensor to a NumPy array and normalize it to [0, 255]
    vid = vid_target_recon.permute(0, 2, 3, 4, 1).clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).numpy().astype(np.uint8)  # [T, H, W, C]


    height, width = vid.shape[2:4]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height))
    if not out.isOpened():
        print(f"[ERROR] VideoWriter failed to open: {save_path}")


    for frame in vid[0]: 
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        out.write(frame_bgr)

    out.release()
        
def init_lia():
    
    print('==> loading model')
    gen = Generator(
        256, 
        512, 
        20, 
        1
    ).cuda()
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pretrained_weights/vox.pt")
    try:
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        gen.load_state_dict(weight)
        gen.eval()
        
    except Exception as e:
        print(f"[ERROR] Failed to load model weights from {model_path}")
        traceback.print_exc()
    
    return gen

def main(
    scenes,
    original,
    file_name = 'final',
    save_folder = './results',
    ):
    
    gen = init_lia()
    vid_target_recon = []

    for (scene_number, scene_id, scene_video_path, source_image) in tqdm(scenes, desc="Processing Scenes"):
        print(f"[INFO] Processing scene_number={scene_number}, scene_id={scene_id}")
        print(f"       => {scene_video_path}")

        print(f"[INFO] -> Running LIA on (source={source_image}, driving={scene_video_path})")
        
         # Process video
        print('==> loading source image')
        img_source = img_preprocessing(source_image, 256).cuda()
        vid_target, fps = vid_preprocessing(scene_video_path)
        vid_target = vid_target.cuda()
        
        if "no_face" in source_image:
            print(f"[INFO] -> Skipping LIA for no_face image: {source_image}")
            vid_target_recon.append(vid_target)
            continue
        
        # Generate anonymized video
        with torch.no_grad():
            h_start = gen.enc.enc_motion(vid_target[:, 0, :, :, :])
            
            for i in range(vid_target.size(1)):
                img_target = vid_target[:, i, :, :, :]
                img_recon = gen(img_source, img_target, h_start)
                vid_target_recon.append(img_recon.unsqueeze(2))
    
    final_merged = os.path.join(save_folder, f"{file_name}_output_lia.mp4")
    final_merged_concat = os.path.join(save_folder, f"{file_name}_output_lia_concat.mp4")     
    vid_target_recon = torch.cat(vid_target_recon, dim=2)
    save_video(vid_target_recon, final_merged, fps)
    create_concat_video(original, final_merged, source_image, final_merged_concat)
    print(f"Saving video to: {final_merged}")
        