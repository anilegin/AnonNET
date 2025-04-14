from .head_segmentation import segmentation_pipeline as seg_pipeline
import numpy as np 
import cv2
import torch
# from ultralytics import YOLO
from PIL import Image
import argparse
import os
import sys
import torch
import torchvision.transforms as transforms

import gc

# YOLO_MODEL = "models/yolov8x.pt"

from .custom_yolo import CustomYOLO
from deepface import DeepFace

from .face_parsing.models.bisenet import BiSeNet
from .face_parsing.utils.common import ATTRIBUTES, COLOR_LIST, letterbox


class Segment:
    # def load_YOLO(self, device: str, model: str):
    #     # There are many different sizes for the pretrained YOLOv8-model. Yolov8x is the largest one. 

    #     yolo = YOLO(model)

    #     yolo.to(device)

    #     return yolo
    
    def load_face_seg(
        self,
        weight = "./AnonHead/face_parsing/weights/resnet18.pt",
        model = "resnet18"
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
        
        return model
    
    def __init__(self, load_det = True, load_seg = True, load_face = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "models/yolov8x.pt"
        # self.yolo = self.load_YOLO(self.device, self.model)
        if load_det:
            self.yolo = CustomYOLO()
        if load_seg:
            self.segmentation_pipeline = seg_pipeline.HumanHeadSegmentationPipeline(device=self.device)
        if load_face:
            self.face_seg = self.load_face_seg()
    
    def fill_black_holes(self, mask: np.ndarray):

        mask_bin = np.where(mask > 127, 255, 0).astype(np.uint8)


        inverted = 255 - mask_bin

        h, w = inverted.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        cv2.floodFill(inverted, flood_mask, (0, 0), 255)
        cv2.floodFill(inverted, flood_mask, (w - 1, 0), 255)
        cv2.floodFill(inverted, flood_mask, (0, h - 1), 255)
        cv2.floodFill(inverted, flood_mask, (w - 1, h - 1), 255)

        inverted_back = 255 - inverted

        filled_mask = cv2.bitwise_or(mask_bin, inverted_back)

        return filled_mask
    
    def segment(self, image: np.ndarray):
        """
        Segments the input image, turning all non-black pixels in the segmented output to white.

        Args:
            image (numpy.ndarray): Input image as a NumPy array.

        Returns:
            numpy.ndarray: Output image with non-black pixels turned white.
        """
        segmented_image = image 

        # Create a mask for non-black pixels
        non_black_mask = (segmented_image[:, :, 0] != 0) | \
                        (segmented_image[:, :, 1] != 0) | \
                        (segmented_image[:, :, 2] != 0)

        # Turn all non-black pixels to white
        segmented_image[non_black_mask] = [255, 255, 255]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

        return cleaned_image


    def annotate(self, img: Image.Image):
        """ Generates annotation image (segmentation) for the given image. 
        
        Arguments:
            image: image to segment. 
        Returns:
            mask_image: annotated segment(s) as a binary image or None if no people in img.
        """
        # Convert image to RGB
        img = img.convert("RGB")
        
        img_array = np.array(img)
        
        segmentation_map = self.segmentation_pipeline(img_array)
        final = img_array * cv2.cvtColor(segmentation_map, cv2.COLOR_GRAY2RGB)
        
        masked_image = self.segment(final)
        mask_image = Image.fromarray(masked_image)
        
        mask_image = mask_image.convert("L")  # Convert mask to grayscale
        mask_data = np.array(mask_image)
        binary_mask = np.where(mask_data > 127, 255, 0).astype(np.uint8)
        final_mask_image = Image.fromarray(binary_mask)

        return final_mask_image
    
    @torch.no_grad()
    def annotate_face(self, img: Image.Image, fill=True):
        
        def prepare_image(image):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            image_tensor = transform(image)
            image_batch = image_tensor.unsqueeze(0)

            return image_batch
        
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        face_indices = [1, 2, 3, 4, 5, 9, 10, 11, 12, 13] #full face 
        
        resized_image = img.resize((512, 512), resample=Image.BILINEAR)
        transformed_image = prepare_image(resized_image)
        image_batch = transformed_image.to(device)
        
        output = self.face_seg(image_batch)[0]
        predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)
        face_mask = np.isin(predicted_mask, face_indices).astype(np.uint8) * 255
        face_mask = (face_mask == 255).astype(np.uint8) * 255  # Ensures only face is white
        original_size = img.size
        face_mask_image = Image.fromarray(face_mask).resize(original_size, Image.LANCZOS)
        
        if fill:
            mask_array = np.array(face_mask_image.convert("L"))
            mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
            mask_filled = self.fill_black_holes(mask_array)
            
            face_mask_image = Image.fromarray(mask_filled)
        
        torch.cuda.empty_cache()
        gc.collect()

        return face_mask_image


    @torch.no_grad()
    def retinaface_detect_and_annotate(self, img, margin: float = 0.8, method: str = 'head', fill = True ,only_boxes = False, bboxes=None):
    
        output = []
        
        if isinstance(img,str):
            path_img = img
            img = Image.open(img)
            img_array = np.array(img.convert("RGB"))
        else:
            path_img = img
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_array)
            img_array = np.array(img.convert("RGB"))
        
        if bboxes == None:
            try:
                extracted_faces = DeepFace.extract_faces(
                    img_path = path_img,        # can be a NumPy array
                    detector_backend = 'retinaface',   
                    enforce_detection = False,     
                    expand_percentage = margin*100,
                    align = True                 
                )
            except Exception as e:
                print(f"DeepFace extract failed: {e}")
                extracted_faces = []

            extracted_faces = [f for f in extracted_faces if f.get("confidence", 0) > 0.85]
            num_faces = len(extracted_faces)
            print(f"Detected {num_faces} face(s).")
            if num_faces == 0: return output #no face
            img_array = np.array(img.convert("RGB"))  # For cropping
            h, w = img_array.shape[:2]
            
            if only_boxes:
                coordinates = []
                for face_data in extracted_faces:
                    # 'face' is the extracted face as a numpy array
                    face_np = face_data["face"]
                    facial_area = face_data.get("facial_area", None)
                    x, y, w_face, h_face = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    x1 = x
                    y1 = y
                    x2 = x + w_face
                    y2 = y + h_face
                    coordinates.append((x1, x2, y1, y2))
                return coordinates
            
            for face_data in extracted_faces:
                # 'face' is the extracted face as a numpy array
                face_np = face_data["face"]
                facial_area = face_data.get("facial_area", None)
                x, y, w_face, h_face = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                x1, x2, y1, y2 = x, x+w_face, y, y+h_face
                
                # Make the box a square
                size = max(w_face, h_face)  
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                x1 = center_x - size // 2
                x2 = center_x + size // 2
                y1 = center_y - size // 2
                y2 = center_y + size // 2

                margin_size = int(margin * size) 
                x1 -= margin_size
                x2 += margin_size
                y1 -= margin_size
                y2 += margin_size

                x1 = max(0, x1)
                y1 = max(0, y1)
                # x2 = min(w - 1, x2)  
                # y2 = min(h - 1, y2)  
                x2 = min(w, x2)  
                y2 = min(h, y2) 
                
                coordinates = (x1, x2, y1, y2)
        
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")
            if method not in ['face','head']:
                method = 'face'
                self.face_seg = self.load_face_seg()
            if method == 'face':
                mask_pil = self.annotate_face(cropped_img_pil, fill)
            if method == 'head':
                mask_pil = self.annotate(cropped_img_pil)
            output.append((mask_pil, cropped_img_pil, coordinates))
        else:
            output = []
            for face in bboxes:
                x1, x2, y1, y2 = face
                cropped_np = img_array[y1:y2, x1:x2]
                cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")
                if method not in ['face','head']:
                    method = 'face'
                    self.face_seg = self.load_face_seg()
                if method == 'face':
                    mask_pil = self.annotate_face(cropped_img_pil, fill)
                if method == 'head':
                    mask_pil = self.annotate(cropped_img_pil)
                output.append((mask_pil, cropped_img_pil, face))
            
        torch.cuda.empty_cache()
        gc.collect()
        # print(output)
        return output
    
    @torch.no_grad()
    def yolo_detect_and_annotate(self, img: str, margin: float = 0.8, force = True, bboxes = [], only_boxes = False):
        """
        Args:
            img: Path to image.
        
        Returns:
            A list of tuples: [(mask_image, cropped_image), ...].
            - mask_image: a PIL image (grayscale binary mask).
            - cropped_image: the original cropped region as a PIL image (RGB).
            Returns an empty list if no detections are found.
        """

        # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
        #results = self.yolo.predict(img, save=False, classes=0)
        # results = self.yolo.detect(img)
        if bboxes == []:
            bboxes = self.yolo.run(img)
            path_img = img
            img = Image.open(img)
            
            
            # If YOLO returns no bboxes or no boxes, try retina face
            if not bboxes and force:
                
                print("No faces found by YOLO. Trying RetinaFace to find face(s).")
                output = []
                output = self.retinaface_detect_and_annotate(path_img, method = 'head')
                #print(output)
                return output

            # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
            # boxes_xyxy = bboxes[0].boxes.xyxy # for ultralytics
        else:
            path_img = img
            img = Image.open(img)
            
        boxes_xyxy = bboxes
        if only_boxes:
            return boxes_xyxy
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]

        output = []
        for box in boxes_xyxy:
            x1, x2, y1, y2 = box
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")
            
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box:", box)
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            # Make the box a square
            size = max(box_width, box_height)  
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2

            margin_size = int(margin * size) 
            x1 -= margin_size
            x2 += margin_size
            y1 -= margin_size
            y2 += margin_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            # x2 = min(w - 1, x2)  
            # y2 = min(h - 1, y2)  
            x2 = min(w, x2)  
            y2 = min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box after margin adjustment:", (x1, x2, y1, y2))
                continue
            
            coordinates = (x1,x2,y1,y2)
            
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

            mask_pil = self.annotate(cropped_img_pil)  # grayscale mask

            output.append((mask_pil, cropped_img_pil, coordinates))
        
        torch.cuda.empty_cache()
        gc.collect()    
        
        return output
    
    
    @torch.no_grad()
    def bisenet_detect_and_annotate(self, img: str, margin: float = 0.8, force = True, bboxes = [], only_boxes = False, fill=True):
        """
        Args:
            img: Path to image.
        
        Returns:
            A list of tuples: [(mask_image, cropped_image), ...].
            - mask_image: a PIL image (grayscale binary mask).
            - cropped_image: the original cropped region as a PIL image (RGB).
            Returns an empty list if no detections are found.
        """

        # Run YOLO detection. Adjust 'classes=0' if you're detecting persons on class 0.
        #bboxes = self.yolo.predict(img, save=False, classes=0)
        # bboxes = self.yolo.detect(img)
        if bboxes == []:
            bboxes = self.yolo.run(img)
            path_img = img
            img = Image.open(img)
            
            
            # If YOLO returns no bboxes or no boxes, try retina face
            if not bboxes and force:
                
                print("No faces found by YOLO. Trying RetinaFace to find face(s).")
                output = []
                output = self.retinaface_detect_and_annotate(path_img, method = 'face')
                #print(output)
                return output

        # 'boxes.xyxy' is a Nx4 tensor: [x1, y1, x2, y2]
        # boxes_xyxy = bboxes[0].boxes.xyxy # for ultralytics
        else:
            path_img = img
            img = Image.open(img)
            
            
        boxes_xyxy = bboxes
        if only_boxes:
            return boxes_xyxy
        img_array = np.array(img.convert("RGB"))  # For cropping
        h, w = img_array.shape[:2]

        output = []
        for box in boxes_xyxy:
            x1, x2, y1, y2 = box
            x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)

            print(f"Original Box: (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

            # Ensure box is valid
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid bounding box:", box)
                continue

            box_width = x2 - x1
            box_height = y2 - y1

            # Make the box a square
            size = max(box_width, box_height)  
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Adjust the coordinates to keep the square centered
            x1 = center_x - size // 2
            x2 = center_x + size // 2
            y1 = center_y - size // 2
            y2 = center_y + size // 2

            margin_size = int(margin * size) 
            x1 -= margin_size
            x2 += margin_size
            y1 -= margin_size
            y2 += margin_size

            x1 = max(0, x1)
            y1 = max(0, y1)
            # x2 = min(w - 1, x2)  
            # y2 = min(h - 1, y2)
            x2 = min(w, x2)  
            y2 = min(h, y2)  

            # Validate adjusted square box
            if x2 <= x1 or y2 <= y1:
                print("Skipping invalid square bounding box:", (x1, x2, y1, y2))
                continue

            coordinates = (x1, x2, y1, y2)

            # Crop the original image region
            cropped_np = img_array[y1:y2, x1:x2]
            cropped_img_pil = Image.fromarray(cropped_np, mode="RGB")

            mask_pil = self.annotate_face(cropped_img_pil,fill=fill)  # grayscale mask

            output.append((mask_pil, cropped_img_pil, coordinates))
            
        torch.cuda.empty_cache()
        gc.collect()
            
        return output
        
    
    def merge_crops(
        self,
        original_img: Image.Image,
        results: list[tuple[Image.Image, Image.Image, tuple[int, int, int, int]]]
    ) -> Image.Image:
        """
        Merges anonymized crops back into the original image with robust dimension handling.
        """
        final_array = np.array(original_img.convert("RGB"))
        H, W = final_array.shape[:2]

        for (mask_pil, cropped_pil, (orig_x1, orig_x2, orig_y1, orig_y2)) in results:
            x1 = max(0, min(orig_x1, W))
            x2 = max(0, min(orig_x2, W))
            y1 = max(0, min(orig_y1, H))
            y2 = max(0, min(orig_y2, H))

            region_width = x2 - x1
            region_height = y2 - y1
            if region_width <= 0 or region_height <= 0:
                continue

            try:
                resized_mask = mask_pil.resize((region_width, region_height), Image.LANCZOS)
                resized_crop = cropped_pil.resize((region_width, region_height), Image.LANCZOS)
            except ValueError as e:
                print(f"Resize error: {e}")
                continue


            mask_array = np.array(resized_mask)
            cropped_array = np.array(resized_crop)
            white_pixels = mask_array > 127


            if white_pixels.shape != (region_height, region_width):
                print(f"Dimension mismatch: Mask {white_pixels.shape} vs Region ({region_height}, {region_width})")
                continue


            try:
                final_array[y1:y2, x1:x2][white_pixels] = cropped_array[white_pixels]
            except Exception as e:
                print(f"Merge error: {e}")
                print(f"Region dimensions: {final_array[y1:y2, x1:x2].shape}")
                print(f"Mask dimensions: {white_pixels.shape}")
                print(f"Crop dimensions: {cropped_array.shape}")

        return Image.fromarray(final_array, mode="RGB")
        
