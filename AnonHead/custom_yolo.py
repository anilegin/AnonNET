import onnxruntime
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
onnxruntime.set_default_logger_severity(4)

class CustomYOLO:
    def __init__(self, model_path="AnonHead/models/yolov8.onnx", conf_threshold=0.8, iou_threshold=0.3):
        """
        Initialize the CustomYOLO class with the model path and thresholds.
        
        Args:
            model_path (str): Path to the YOLO ONNX model.
            conf_threshold (float): Confidence threshold for predictions.
            iou_threshold (float): IoU threshold for NMS.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.ort_session, self.input_shape = self.load_model()

    def load_model(self):
        """
        Load the ONNX model using onnxruntime.
        """
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        
        ort_session = onnxruntime.InferenceSession(self.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        input_shape = ort_session.get_inputs()[0].shape
        return ort_session, input_shape

    def load_image(self, image_path):
        """
        Load and preprocess the input image.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            Tuple: Original image, preprocessed image tensor, and RGB image.
        """
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_height, input_width = self.input_shape[2:]
        resized = cv2.resize(rgb_image, (input_width, input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return image, input_tensor, rgb_image

    def predict(self, image, input_tensor):
        """
        Run inference and filter predictions using NMS.
        
        Args:
            image (np.array): Original image.
            input_tensor (np.array): Preprocessed image tensor.

        Returns:
            List: List of bounding boxes [(x1, x2, y1, y2), ...].
        """
        image_height, image_width = image.shape[:2]
        
        # Run inference
        outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: input_tensor})[0]
        predictions = np.squeeze(outputs).T

        # Filter out low-confidence predictions
        scores = np.max(predictions[:, 4:], axis=1)
        valid_mask = scores > self.conf_threshold
        predictions = predictions[valid_mask]
        scores = scores[valid_mask]

        # Extract bounding boxes (x_center, y_center, w, h)
        boxes = predictions[:, :4]

        # Convert to (x1, y1, x2, y2) format
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2

        # Rescale boxes to original image size
        input_height, input_width = self.input_shape[2:]
        boxes /= np.array([input_width, input_height, input_width, input_height])
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)

        # Apply NMS
        keep_indices = self.nms(boxes, scores)
        boxes = boxes[keep_indices]

        # Filter valid boxes
        final_boxes = []
        for (x1, y1, x2, y2) in boxes:
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image_width - 1, x2)
            y2 = min(image_height - 1, y2)
            if x2 > x1 and y2 > y1:
                final_boxes.append((x1, x2, y1, y2))
        
        return final_boxes

    def nms(self, boxes, scores):
        """
        Apply Non-Maximum Suppression to filter overlapping boxes.
        """
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            remaining_indices = np.where(ious < self.iou_threshold)[0]
            sorted_indices = sorted_indices[remaining_indices + 1]
        return keep_boxes

    def compute_iou(self, box, boxes):
        """
        Compute IoU between a single box and a set of boxes.
        """
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])
        inter_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area
        return inter_area / union_area

    def run(self, image_path):
        """
        Run the detection pipeline on an image.
        
        Args:
            image_path (str): Path to the input image.
        
        Returns:
            List: List of bounding boxes [(x1, x2, y1, y2), ...].
        """
        image, input_tensor, _ = self.load_image(image_path)
        predictions = self.predict(image, input_tensor)
        return predictions


if __name__ == '__main__':
    yolo = CustomYOLO(model_path="models/yolov8.onnx")
    image_path = "/home/aegin/projects/anonymization/RV_inpaint/rv5_controlnet/images/2menlooking.jpg"
    predictions = yolo.run(image_path)
    print("Filtered Bounding Boxes:", predictions)
