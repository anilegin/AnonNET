import os
import argparse
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from .models.bisenet import BiSeNet
from .utils.common import ATTRIBUTES, COLOR_LIST, letterbox


def prepare_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_tensor = transform(image)
    image_batch = image_tensor.unsqueeze(0)

    return image_batch


# @torch.no_grad()
# def inference(
#     input_p: str,
#     weight = "./AnonHead/face_parsing/weights/resnet18.pt",
#     model = "resnet18",
#     face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13]
    
# ):
#     image = Image.open(input_p).convert("RGB")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_classes = 19

#     model = BiSeNet(num_classes, backbone_name=model)
#     model.to(device)

#     if os.path.exists(weight):
#         model.load_state_dict(torch.load(weight))
#     else:
#         raise ValueError(f"Weights not found from given path ({weight})")

#     model.eval()
#     resized_image = image.resize((512, 512), resample=Image.BILINEAR)
#     transformed_image = prepare_image(resized_image)
#     image_batch = transformed_image.to(device)

#     # Forward pass
#     output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only

#     # Debug: Print output tensor shape and values
#     print(f"Output shape: {output.shape}")  # Expected: (num_classes, H, W)
#     print(f"Sample output tensor (min, max): {output.min().item()}, {output.max().item()}")

#     # Convert to class map (segmentation mask)
#     predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
    
#     print(f"Predicted mask shape: {predicted_mask.shape}")  # Expected: (H, W)
#     print(f"Unique values in mask: {np.unique(predicted_mask)}")  # Shows segmentation class indices
    
#     # Define face-related indices
#     # face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13]  # Keep only face parts

#     # Create a binary mask where face parts are white (255) and everything else is black (0)
#     face_mask = np.isin(predicted_mask, face_indices).astype(np.uint8) * 255

#     # Make sure background is BLACK (0) and face is WHITE (255)
#     face_mask = (face_mask == 255).astype(np.uint8) * 255  # Ensures only face is white

#     original_size = image.size
#     face_mask_image = Image.fromarray(face_mask).resize(original_size, Image.LANCZOS)

#     face_mask_image.save(os.path.join('/home/aegin/projects/anonymization/AnonNET/.cache', 'final.png'))
#     return face_mask_image



@torch.no_grad()
def inference(
    image: Image.Image,
    weight = "./AnonHead/face_parsing/weights/resnet18.pt",
    model = "resnet18",
    face_indices = [1] # old one [1, 2, 3, 4, 5, 9, 10, 11, 12, 13] full face 
    
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 19

    model = BiSeNet(num_classes, backbone_name=model)
    model.to(device)

    if os.path.exists(weight):
        model.load_state_dict(torch.load(weight))
    else:
        raise ValueError(f"Weights not found from given path ({weight})")

    model.eval()
    resized_image = image.resize((512, 512), resample=Image.BILINEAR)
    transformed_image = prepare_image(resized_image)
    image_batch = transformed_image.to(device)

    # Forward pass
    output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only

    # Debug: Print output tensor shape and values
    print(f"Output shape: {output.shape}")  # Expected: (num_classes, H, W)
    print(f"Sample output tensor (min, max): {output.min().item()}, {output.max().item()}")

    # Convert to class map (segmentation mask)
    predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
    
    print(f"Predicted mask shape: {predicted_mask.shape}")  # Expected: (H, W)
    print(f"Unique values in mask: {np.unique(predicted_mask)}")  # Shows segmentation class indices
    
    # Define face-related indices
    # face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13]  # Keep only face parts

    # Create a binary mask where face parts are white (255) and everything else is black (0)
    face_mask = np.isin(predicted_mask, face_indices).astype(np.uint8) * 255

    # Make sure background is BLACK (0) and face is WHITE (255)
    face_mask = (face_mask == 255).astype(np.uint8) * 255  # Ensures only face is white

    original_size = image.size
    face_mask_image = Image.fromarray(face_mask).resize(original_size, Image.LANCZOS)

    return face_mask_image
        #face_mask_image.save(os.path.join(output_path, filename.replace('.png', '_binary_mask.png')))

        # Skip visualization (commented out)
        # vis_parsing_maps(
        #     resized_image,
        #     predicted_mask,
        #     save_image=True,
        #     save_path=os.path.join(output_path, filename),
        # )



def parse_args():
    parser = argparse.ArgumentParser(description="Face parsing inference")
    parser.add_argument("--model", type=str, default="resnet18", help="model name, i.e resnet18, resnet34")
    parser.add_argument(
        "--weight",
        type=str,
        default="./weights/resnet18.pt",
        help="path to trained model, i.e resnet18/34"
    )
    parser.add_argument("--input", type=str, default="./assets/images/", help="path to an image or a folder of images")
    parser.add_argument("--output", type=str, default="./assets/", help="path to save model outputs")

    return parser.parse_args()


# @torch.no_grad()
# def inference(config):
#     output_path = config.output
#     input_path = config.input
#     weight = config.weight
#     model = config.model

#     output_path = os.path.join(output_path, model)
#     os.makedirs(output_path, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_classes = 19

#     model = BiSeNet(num_classes, backbone_name=model)
#     model.to(device)

#     if os.path.exists(weight):
#         model.load_state_dict(torch.load(weight))
#     else:
#         raise ValueError(f"Weights not found from given path ({weight})")

#     if os.path.isfile(input_path):
#         input_path = [input_path]
#     else:
#         input_path = [os.path.join(input_path, f) for f in os.listdir(input_path)]

#     model.eval()
#     for file_path in input_path:
#         image = Image.open(file_path).convert("RGB")
#         print(f"Processing image: {file_path}")

#         resized_image = image.resize((512, 512), resample=Image.BILINEAR)
#         transformed_image = prepare_image(resized_image)
#         image_batch = transformed_image.to(device)

#         # Forward pass
#         output = model(image_batch)[0]  # feat_out, feat_out16, feat_out32 -> use feat_out for inference only

#         # Debug: Print output tensor shape and values
#         print(f"Output shape: {output.shape}")  # Expected: (num_classes, H, W)
#         print(f"Sample output tensor (min, max): {output.min().item()}, {output.max().item()}")

#         # Convert to class map (segmentation mask)
#         predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        
#         print(f"Predicted mask shape: {predicted_mask.shape}")  # Expected: (H, W)
#         print(f"Unique values in mask: {np.unique(predicted_mask)}")  # Shows segmentation class indices
        
#         # Define face-related indices
#         face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13]  # Keep only face parts

#         # Create a binary mask where face parts are white (255) and everything else is black (0)
#         face_mask = np.isin(predicted_mask, face_indices).astype(np.uint8) * 255

#         # Make sure background is BLACK (0) and face is WHITE (255)
#         face_mask = (face_mask == 255).astype(np.uint8) * 255  # Ensures only face is white

#         # Save the binary mask
#         filename = os.path.basename(file_path)
#         original_size = image.size
#         face_mask_image = Image.fromarray(face_mask).resize(original_size, Image.LANCZOS)
#         face_mask_image.save(os.path.join(output_path, filename.replace('.png', '_binary_mask.png')))

#         # Skip visualization (commented out)
#         # vis_parsing_maps(
#         #     resized_image,
#         #     predicted_mask,
#         #     save_image=True,
#         #     save_path=os.path.join(output_path, filename),
#         # )

if __name__ == "__main__":
    args = parse_args()
    inference(config=args)




