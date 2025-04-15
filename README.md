# AnonNET: A Unified Framework for Expression-Consistent Anonymization in Talking Head Videos



**AnonNET** is a multi-stage face anonymization pipeline designed to preserve non-identifying facial attributes (e.g., age, gender, race, and expression) while ensuring robust identity obfuscation in both images and talking-head videos. It combines attribute-conditioned diffusion-based inpainting with landmark-free motion synthesis, making it suitable for real-world and privacy-critical video anonymization applications.

This repository provides the implementation of AnonNET as described in our paper:

📄 **Now You See Me, Now You Don’t: A Unified Framework for Expression-Consistent Anonymization in Talking Head Videos**  
 
---

## Key Features

- **Attribute-aware anonymization**  
  Guided by facial attribute recognition (age, gender, race, expression)

- **Diffusion-based inpainting**  
  Utilizes Stable Diffusion v1.5 with ControlNet conditioning (e.g., face parsing masks)

- **Motion synthesis**  
  Landmark-free reenactment using LIA or LivePortrait

---

## 🛠 Installation

> Requires: **Python 3.9** and **CUDA >= 12.1** (GPU required for inpainting and motion synthesis)


```bash
git clone https://github.com/anilegin/AnonNET.git
cd AnonNET
python3.9 -m venv AnonNET
source AnonNET/bin/activate  # On Windows use: AnonNET\Scripts\activate
pip install -r requirements.txt
```

---

## Extra Dependencies

Manually download or cache models for:

`vox.pt` – Required for LIA motion synthesis  
  Download from the [Releases](https://github.com/anilegin/AnonNET/releases/download/v1.0.0/vox.pt) page and place under `Generation/pretrained_weights`.

Other model weights (Stable Diffusion, RetinaFace, etc.) are expected to be downloaded and stored automatically when the script is initalized.

---

## 📸 Image Anonymization

```bash
python image_anonymize.py \
    --image path/to/input.jpg \
    --segment face \
    --prompt "A photorealistic portrait of a middle-aged Asian woman with a neutral expression" \
    --strength 0.9 0.4 0.3 \
    --save_folder results/
```

### Parameters

| Argument        | Description                                               |
|----------------|-----------------------------------------------------------|
| `--image`       | Input image path                                          |
| `--segment`     | Type of mask: `face` or `head`                            |
| `--prompt`      | (Optional) Otherwise attribute-aware one will be generated|
| `--strength`    | Mask, lineart, and openpose guidance strengths            |
| `--steps`       | Denosing steps                                            |
| `--seed`        | Random seed (optional)                                    |
| `--save_folder` | Output folder for anonymized image                        |

---

## 🎬 Video Anonymization

```bash
python anonymize.py \
    --driving_path path/to/video.mp4 \
    --motion lp \
    --segment face \
    --save_folder results/
```

### Parameters

| Argument           | Description                                           |
|-------------------|-------------------------------------------------------|
| `--driving_path`   | Path to input video                                   |
| `--motion`         | `lp` for LivePortrait, `lia` for Latent Image Animator|
| `--segment`        | `face` or `head` segmentation                         |
| `--max_len`        | Max clip length (optional)                            |
| `--save_folder`    | Output folder                                         |
| `--no_stitch`      | No eye-lip retargeting for LivePortrait               |

---
##  Contact

For questions or feedback, please contact: [anilegin@gmail.com](mailto:anilegin@gmail.com)

## 📃 License

This repository is released under the MIT License.
