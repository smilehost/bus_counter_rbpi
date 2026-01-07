"""
Hailo Accelerator Inference Module for YOLO-BoTSORT Tracker

This module provides a HailoDetector class that wraps Hailo accelerator inference
for YOLO object detection, providing a YOLO-compatible API for seamless integration
with the existing YOLO-BoTSORT tracker.

Requirements:
    - hailort library for Hailo accelerator communication
    - HEF (Hailo Executable Format) model file
    - Hailo 26 accelerator hardware

Author: Auto-generated
Date: 2026-01-07
"""

import cv2
import numpy as np
from typing import List, Optional, Union, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import hailort
    HAILORT_AVAILABLE = True
except ImportError:
    HAILORT_AVAILABLE = False
    logger.warning("hailort library not available. HailoDetector will not function.")


class HailoResult:
    """
    YOLO-compatible result wrapper for Hailo inference outputs.
    
    This class mimics the structure of ultralytics YOLO results to ensure
    compatibility with existing code that expects YOLO result objects.
    
    Attributes:
        boxes: HailoBoxes object containing detection information
        orig_shape: Original frame shape (height, width)
    """
    
    def __init__(self, boxes: 'HailoBoxes', orig_shape: Tuple[int, int]):
        """
        Initialize HailoResult.
        
        Args:
            boxes: HailoBoxes object containing detection data
            orig_shape: Original frame shape as (height, width)
        """
        self.boxes = boxes
        self.orig_shape = orig_shape


class HailoBoxes:
    """
    YOLO-compatible boxes wrapper for Hailo detection outputs.
    
    This class provides the same interface as ultralytics YOLO boxes
    to ensure seamless integration with existing code.
    
    Attributes:
        xyxy: Bounding boxes in [x1, y1, x2, y2] format
        conf: Confidence scores for each detection
        cls: Class IDs for each detection
        data: Combined detection data [x1, y1, x2, y2, conf, cls]
    """
    
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        """
        Initialize HailoBoxes.
        
        Args:
            xyxy: Bounding boxes array of shape (N, 4) in [x1, y1, x2, y2] format
            conf: Confidence scores array of shape (N,)
            cls: Class IDs array of shape (N,)
        """
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls
        
        # Combine all data into a single array for compatibility
        if len(xyxy) > 0:
            self.data = np.column_stack([xyxy, conf, cls])
        else:
            self.data = np.empty((0, 6))
    
    def cpu(self):
        """Return boxes on CPU (for compatibility with YOLO API)."""
        return self
    
    def numpy(self):
        """Return boxes as numpy arrays (for compatibility with YOLO API)."""
        return self


class HailoDetector:
    """
    Hailo accelerator detector with YOLO-compatible API.
    
    This class wraps Hailo accelerator inference for YOLO object detection,
    providing a predict() method with the same signature as ultralytics YOLO.
    
    The detector handles:
    - Hailo device and model initialization
    - Frame preprocessing (resize, normalize, batch dimension)
    - Inference execution on Hailo accelerator
    - Output postprocessing to YOLO-compatible format
    
    Example:
        >>> detector = HailoDetector("yolov8n.hef", input_shape=(640, 640))
        >>> results = detector.predict(frame, conf=0.5, iou=0.45, classes=[0], verbose=False)
        >>> for result in results:
        ...     if result.boxes is not None:
        ...         boxes = result.boxes.xyxy
        ...         scores = result.boxes.conf
        ...         classes = result.boxes.cls
    """
    
    def __init__(
        self,
        hef_path: str,
        input_shape: Tuple[int, int] = (640, 640),
        batch_size: int = 1,
        device_id: Optional[int] = None
    ):
        """
        Initialize HailoDetector.
        
        Args:
            hef_path: Path to HEF (Hailo Executable Format) model file
            input_shape: Input shape expected by the model (height, width)
            batch_size: Batch size for inference (default: 1)
            device_id: Optional Hailo device ID to use (default: None for auto-select)
            
        Raises:
            ImportError: If hailort library is not available
            FileNotFoundError: If HEF file does not exist
            RuntimeError: If Hailo device or model initialization fails
        """
        if not HAILORT_AVAILABLE:
            raise ImportError(
                "hailort library is not available. "
                "Please install it using: pip install hailort"
            )
        
        self.hef_path = hef_path
        self.input_shape = input_shape  # (height, width)
        self.batch_size = batch_size
        self.device_id = device_id
        
        # Initialize Hailo components
        self.vdevice = None
        self.infer_model = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        
        # Model metadata (will be loaded from HEF)
        self.num_classes = None
        self.anchors = None
        self.strides = None
        
        # Initialize the detector
        self._initialize_hailo()
        
        logger.info(f"HailoDetector initialized with model: {hef_path}")
        logger.info(f"Input shape: {input_shape}, Batch size: {batch_size}")
    
    def _initialize_hailo(self) -> None:
        """
        Initialize Hailo device and inference model.
        
        This method:
        1. Creates a Hailo VDevice
        2. Loads the HEF model
        3. Configures the inference model
        4. Gets input/output stream information
        
        Raises:
            FileNotFoundError: If HEF file does not exist
            RuntimeError: If any Hailo initialization step fails
        """
        try:
            # Check if HEF file exists
            import os
            if not os.path.exists(self.hef_path):
                raise FileNotFoundError(
                    f"HEF model file not found: {self.hef_path}"
                )
            
            # Create Hailo VDevice
            logger.info("Creating Hailo VDevice...")
            if self.device_id is not None:
                self.vdevice = hailort.VDevice(device_params=hailort.VDeviceSpecificParams(device_id=self.device_id))
            else:
                self.vdevice = hailort.VDevice()
            
            # Load HEF model
            logger.info(f"Loading HEF model from: {self.hef_path}")
            hef = hailort.HEF(self.hef_path)
            
            # Configure inference model
            logger.info("Configuring inference model...")
            self.infer_model = self.vdevice.create_infer_model(hef)
            
            # Set batch size
            self.infer_model.set_batch_size(self.batch_size)
            
            # Get input/output vstream info
            self.input_vstream_info = self.infer_model.get_input_vstream_infos()[0]
            self.output_vstream_info = self.infer_model.get_output_vstream_infos()[0]
            
            # Extract model metadata
            self._extract_model_metadata(hef)
            
            logger.info("Hailo device and model initialized successfully")
            
        except hailort.HailoRTException as e:
            raise RuntimeError(f"Hailo runtime error during initialization: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Hailo detector: {e}")
    
    def _extract_model_metadata(self, hef: hailort.HEF) -> None:
        """
        Extract metadata from the HEF model.
        
        Args:
            hef: Hailo HEF object
        """
        try:
            # Get input shape from HEF
            input_info = hef.get_input_vstream_infos()[0]
            self.input_shape = (input_info.shape.height, input_info.shape.width)
            
            # Try to get number of classes from output shape
            # Typical YOLO output: (batch, num_anchors * (5 + num_classes), grid_h, grid_w)
            output_info = hef.get_output_vstream_infos()[0]
            output_shape = output_info.shape
            
            # Estimate number of classes from output shape
            # This is a heuristic and may need adjustment based on model architecture
            if len(output_shape) == 4:
                # Format: (batch, channels, height, width)
                # channels = num_anchors * (5 + num_classes)
                # num_anchors is typically 3 for YOLO
                channels = output_shape.channels
                if (channels - 15) % 3 == 0:
                    self.num_classes = (channels - 15) // 3
                else:
                    # Default to COCO classes if we can't determine
                    self.num_classes = 80
                    logger.warning(
                        f"Could not determine number of classes from output shape. "
                        f"Defaulting to {self.num_classes} (COCO)."
                    )
            else:
                self.num_classes = 80
                logger.warning(
                    f"Unexpected output shape {output_shape}. "
                    f"Defaulting to {self.num_classes} classes."
                )
            
            logger.info(f"Model metadata - Input shape: {self.input_shape}, "
                       f"Num classes: {self.num_classes}")
            
        except Exception as e:
            logger.warning(f"Could not extract model metadata: {e}")
            self.num_classes = 80  # Default to COCO
    
    def preprocess(
        self,
        frame: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame for Hailo inference.
        
        This method:
        1. Resizes the frame to the model's input size
        2. Normalizes pixel values to [0, 1]
        3. Converts BGR to RGB if needed
        4. Adds batch dimension
        
        Args:
            frame: Input frame in BGR format (height, width, channels)
            target_size: Optional target size (height, width). If None, uses model input size.
            
        Returns:
            Tuple of (preprocessed_frame, scale_factor, original_size)
            - preprocessed_frame: Preprocessed frame ready for inference (1, height, width, 3)
            - scale_factor: Scale factor used for resizing (new_size / original_size)
            - original_size: Original frame size (height, width)
        """
        if target_size is None:
            target_size = self.input_shape
        
        original_size = (frame.shape[0], frame.shape[1])
        target_height, target_width = target_size
        
        # Calculate scale factor (use the same scale for both dimensions)
        scale_factor = min(target_width / original_size[1], target_height / original_size[0])
        
        # Resize frame while maintaining aspect ratio
        new_width = int(original_size[1] * scale_factor)
        new_height = int(original_size[0] * scale_factor)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR to RGB (YOLO models typically expect RGB)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched, scale_factor, original_size
    
    def postprocess(
        self,
        raw_output: np.ndarray,
        scale_factor: float,
        original_size: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
        classes: Optional[List[int]] = None
    ) -> HailoBoxes:
        """
        Postprocess raw Hailo output to YOLO-compatible format.
        
        This method:
        1. Decodes raw output tensors to bounding boxes
        2. Applies confidence threshold
        3. Applies NMS (Non-Maximum Suppression) with IOU threshold
        4. Scales coordinates back to original image size
        5. Filters by class if specified
        
        Args:
            raw_output: Raw output tensor from Hailo inference
            scale_factor: Scale factor used during preprocessing
            original_size: Original frame size (height, width)
            conf_threshold: Confidence threshold for filtering detections
            iou_threshold: IOU threshold for NMS
            classes: Optional list of class IDs to filter (None = all classes)
            
        Returns:
            HailoBoxes object containing filtered detections
        """
        # Decode raw output to boxes, scores, and classes
        boxes, scores, class_ids = self._decode_output(raw_output)
        
        if len(boxes) == 0:
            return HailoBoxes(
                np.empty((0, 4)),
                np.empty((0,)),
                np.empty((0,))
            )
        
        # Apply confidence threshold
        conf_mask = scores >= conf_threshold
        boxes = boxes[conf_mask]
        scores = scores[conf_mask]
        class_ids = class_ids[conf_mask]
        
        if len(boxes) == 0:
            return HailoBoxes(
                np.empty((0, 4)),
                np.empty((0,)),
                np.empty((0,))
            )
        
        # Apply NMS
        nms_indices = self._apply_nms(boxes, scores, iou_threshold)
        boxes = boxes[nms_indices]
        scores = scores[nms_indices]
        class_ids = class_ids[nms_indices]
        
        if len(boxes) == 0:
            return HailoBoxes(
                np.empty((0, 4)),
                np.empty((0,)),
                np.empty((0,))
            )
        
        # Scale coordinates back to original size
        # Remove padding and scale back
        target_height, target_width = self.input_shape
        orig_height, orig_width = original_size
        
        # Calculate padding
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)
        pad_top = (target_height - new_height) // 2
        pad_left = (target_width - new_width) // 2
        
        # Remove padding
        boxes[:, 0] -= pad_left  # x1
        boxes[:, 1] -= pad_top   # y1
        boxes[:, 2] -= pad_left  # x2
        boxes[:, 3] -= pad_top   # y2
        
        # Scale back to original size
        boxes[:, [0, 2]] /= scale_factor  # x coordinates
        boxes[:, [1, 3]] /= scale_factor  # y coordinates
        
        # Clip to image boundaries
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_height)
        
        # Filter by class if specified
        if classes is not None:
            class_mask = np.isin(class_ids, classes)
            boxes = boxes[class_mask]
            scores = scores[class_mask]
            class_ids = class_ids[class_mask]
        
        return HailoBoxes(boxes, scores, class_ids)
    
    def _decode_output(self, raw_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decode raw Hailo output to boxes, scores, and class IDs.
        
        This is a generic decoder that works with standard YOLO output format.
        The exact implementation may need to be adjusted based on the specific
        YOLO model architecture and Hailo post-processing configuration.
        
        Args:
            raw_output: Raw output tensor from Hailo inference
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        # Reshape output if needed
        # Typical YOLO output shape: (batch, num_predictions, 5 + num_classes)
        # where 5 = [x, y, w, h, confidence]
        
        output = raw_output.squeeze()  # Remove batch dimension
        
        # Handle different output formats
        if output.ndim == 3:
            # Format: (height, width, channels)
            # Transpose to (channels, height, width)
            output = output.transpose(2, 0, 1)
        
        # Flatten spatial dimensions
        output = output.reshape(-1, output.shape[-1])
        
        # Extract boxes (assuming center format: [x, y, w, h, conf, class_probs...])
        if output.shape[1] >= 5:
            # Extract center coordinates, dimensions, and confidence
            x_center = output[:, 0]
            y_center = output[:, 1]
            width = output[:, 2]
            height = output[:, 3]
            conf = output[:, 4]
            
            # Convert center format to xyxy format
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            boxes = np.column_stack([x1, y1, x2, y2])
            
            # Extract class probabilities
            if output.shape[1] > 5:
                class_probs = output[:, 5:]
                class_ids = np.argmax(class_probs, axis=1)
                # Multiply confidence by class probability
                scores = conf * np.max(class_probs, axis=1)
            else:
                class_ids = np.zeros(len(conf), dtype=int)
                scores = conf
        else:
            # Alternative format: [x1, y1, x2, y2, conf, class_id]
            boxes = output[:, :4]
            scores = output[:, 4]
            class_ids = output[:, 5].astype(int) if output.shape[1] > 5 else np.zeros(len(scores), dtype=int)
        
        return boxes, scores, class_ids
    
    def _apply_nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to remove overlapping boxes.
        
        Args:
            boxes: Bounding boxes array (N, 4) in [x1, y1, x2, y2] format
            scores: Confidence scores array (N,)
            iou_threshold: IOU threshold for NMS
            
        Returns:
            Indices of boxes to keep after NMS
        """
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        # Sort by score (descending)
        indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Keep the box with highest score
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Calculate IOU with remaining boxes
            ious = self._calculate_iou(boxes[current], boxes[indices[1:]])
            
            # Remove boxes with IOU above threshold
            mask = ious <= iou_threshold
            indices = indices[1:][mask]
        
        return np.array(keep, dtype=int)
    
    def _calculate_iou(
        self,
        box1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Intersection over Union (IOU) between a box and multiple boxes.
        
        Args:
            box1: Single box [x1, y1, x2, y2]
            boxes2: Multiple boxes (N, 4) in [x1, y1, x2, y2] format
            
        Returns:
            IOU values array (N,)
        """
        # Calculate intersection
        x1 = np.maximum(box1[0], boxes2[:, 0])
        y1 = np.maximum(box1[1], boxes2[:, 1])
        x2 = np.minimum(box1[2], boxes2[:, 2])
        y2 = np.minimum(box1[3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = box1_area + boxes2_area - intersection
        
        # Calculate IOU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        
        return iou
    
    def predict(
        self,
        frame: np.ndarray,
        conf: float = 0.5,
        iou: float = 0.45,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List[HailoResult]:
        """
        Run inference on a frame with YOLO-compatible API.
        
        This method provides the same signature as ultralytics YOLO.predict()
        for seamless integration with existing code.
        
        Args:
            frame: Input frame in BGR format (height, width, channels)
            conf: Confidence threshold for filtering detections (default: 0.5)
            iou: IOU threshold for NMS (default: 0.45)
            classes: Optional list of class IDs to filter (None = all classes)
            verbose: Whether to print verbose output (default: False)
            
        Returns:
            List of HailoResult objects (one per input frame)
            
        Raises:
            RuntimeError: If inference fails
        """
        try:
            # Preprocess frame
            preprocessed, scale_factor, original_size = self.preprocess(frame)
            
            if verbose:
                logger.info(f"Preprocessed frame shape: {preprocessed.shape}")
                logger.info(f"Scale factor: {scale_factor}")
                logger.info(f"Original size: {original_size}")
            
            # Run inference
            bindings = {
                self.input_vstream_info.name: preprocessed,
            }
            
            with self.infer_model.configure() as configured_model:
                raw_outputs = configured_model.infer(bindings)
            
            # Get output
            raw_output = raw_outputs[self.output_vstream_info.name]
            
            if verbose:
                logger.info(f"Raw output shape: {raw_output.shape}")
            
            # Postprocess output
            boxes = self.postprocess(
                raw_output,
                scale_factor,
                original_size,
                conf_threshold=conf,
                iou_threshold=iou,
                classes=classes
            )
            
            # Create result object
            result = HailoResult(boxes, original_size)
            
            if verbose:
                logger.info(f"Number of detections: {len(boxes.xyxy)}")
            
            return [result]
            
        except hailort.HailoRTException as e:
            raise RuntimeError(f"Hailo runtime error during inference: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to run inference: {e}")
    
    def __del__(self):
        """Cleanup Hailo resources."""
        try:
            if self.vdevice is not None:
                self.vdevice.release()
                logger.info("Hailo device released")
        except Exception as e:
            logger.warning(f"Error releasing Hailo device: {e}")


def create_hailo_detector(
    hef_path: str,
    input_shape: Tuple[int, int] = (640, 640),
    batch_size: int = 1,
    device_id: Optional[int] = None
) -> HailoDetector:
    """
    Factory function to create a HailoDetector instance.
    
    This is a convenience function that provides a simple interface
    for creating a HailoDetector with default parameters.
    
    Args:
        hef_path: Path to HEF (Hailo Executable Format) model file
        input_shape: Input shape expected by the model (height, width)
        batch_size: Batch size for inference (default: 1)
        device_id: Optional Hailo device ID to use (default: None for auto-select)
        
    Returns:
        HailoDetector instance
        
    Raises:
        ImportError: If hailort library is not available
        FileNotFoundError: If HEF file does not exist
        RuntimeError: If Hailo device or model initialization fails
        
    Example:
        >>> detector = create_hailo_detector("yolov8n.hef")
        >>> results = detector.predict(frame, conf=0.5)
    """
    return HailoDetector(
        hef_path=hef_path,
        input_shape=input_shape,
        batch_size=batch_size,
        device_id=device_id
    )


if __name__ == "__main__":
    """
    Example usage and testing of HailoDetector.
    
    This section demonstrates how to use the HailoDetector class
    and can be used for basic testing.
    """
    import sys
    
    # Example usage
    print("Hailo Inference Module")
    print("=" * 50)
    
    if not HAILORT_AVAILABLE:
        print("ERROR: hailort library is not available.")
        print("Please install it using: pip install hailort")
        sys.exit(1)
    
    # Example: Create a detector (requires actual HEF file and Hailo hardware)
    # detector = create_hailo_detector("yolov8n.hef")
    # 
    # # Load a test frame
    # frame = cv2.imread("test_image.jpg")
    # if frame is not None:
    #     # Run inference
    #     results = detector.predict(frame, conf=0.5, iou=0.45, classes=[0], verbose=True)
    #     
    #     # Process results
    #     for result in results:
    #         if result.boxes is not None and len(result.boxes.xyxy) > 0:
    #             print(f"Detected {len(result.boxes.xyxy)} objects")
    #             for i, (box, score, cls) in enumerate(zip(
    #                 result.boxes.xyxy, result.boxes.conf, result.boxes.cls
    #             )):
    #                 print(f"  Object {i}: {box}, score={score:.3f}, class={int(cls)}")
    #         else:
    #             print("No detections")
    
    print("\nModule loaded successfully!")
    print("To use HailoDetector, you need:")
    print("1. A Hailo 26 accelerator device")
    print("2. A HEF (Hailo Executable Format) model file")
    print("3. The hailort library installed")
