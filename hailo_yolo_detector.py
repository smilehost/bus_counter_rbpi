"""
HAILO-optimized YOLO detector for Raspberry Pi 5 with HAILO 8L AI Accelerator
This module provides a wrapper for YOLO inference using HAILO hardware acceleration
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import os

try:
    from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, FormatType
    from hailo_platform import InputVStreamParams, OutputVStreamParams
    HAILO_AVAILABLE = True
except ImportError:
    print("Warning: HAILO SDK not available. Falling back to CPU inference.")
    HAILO_AVAILABLE = False

from class_names import get_class_name


class HailoYOLODetector:
    """
    HAILO-optimized YOLO detector class
    Provides high-performance inference using HAILO 8L AI Accelerator
    """
    
    def __init__(self, model_path: str, config=None):
        """
        Initialize HAILO YOLO detector
        
        Args:
            model_path: Path to HAILO HEF model file
            config: Configuration object with HAILO settings
        """
        self.model_path = model_path
        self.config = config
        self.target = None
        self.network_group = None
        self.input_vstream_info = None
        self.output_vstream_info = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.infer_pipeline = None
        
        # Performance tracking
        self.inference_times = []
        self.last_inference_time = 0
        
        # Initialize HAILO if available
        if HAILO_AVAILABLE and self._check_hailo_model():
            self._initialize_hailo()
        else:
            raise RuntimeError("HAILO SDK not available or model file not found")
    
    def _check_hailo_model(self) -> bool:
        """Check if HAILO model file exists and is valid"""
        if not os.path.exists(self.model_path):
            print(f"Error: HAILO model file not found: {self.model_path}")
            return False
        
        if not self.model_path.endswith('.hef'):
            print(f"Error: Model file must be a HAILO HEF file: {self.model_path}")
            return False
        
        return True
    
    def _initialize_hailo(self):
        """Initialize HAILO device and load model"""
        try:
            # Load HEF model
            self.hef = HEF(self.model_path)
            
            # Create VDevice
            params = VDevice.create_params()
            self.target = VDevice(params)
            
            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                hef=self.hef, 
                interface=HailoStreamInterface.PCIe
            )
            
            # Get network group name
            network_group_names = self.hef.get_network_group_names()
            if len(network_group_names) == 0:
                raise RuntimeError("No network groups found in HEF file")
            
            network_group_name = network_group_names[0]
            
            # Set batch size from config
            if self.config and hasattr(self.config, 'HAILO_BATCH_SIZE'):
                configure_params[network_group_name].batch_size = self.config.HAILO_BATCH_SIZE
            
            # Configure network groups
            network_groups = self.target.configure(self.hef, configure_params)
            self.network_group = network_groups[0]
            
            # Get input and output stream info
            self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
            self.output_vstream_info = self.hef.get_output_vstream_infos()[0]
            
            # Create input and output virtual streams params
            # Use FLOAT32 for both input and output as required by HAILO
            input_format = FormatType.FLOAT32
            if self.config and hasattr(self.config, 'HAILO_INPUT_FORMAT'):
                if self.config.HAILO_INPUT_FORMAT.upper() == "UINT8":
                    input_format = FormatType.UINT8
            
            self.input_vstreams_params = InputVStreamParams.make(
                self.network_group, format_type=input_format
            )
            # Use FLOAT32 for output as required by HAILO
            self.output_vstreams_params = OutputVStreamParams.make(
                self.network_group, format_type=FormatType.FLOAT32
            )
            
            # Create inference pipeline
            print(f"[DEBUG] Creating InferVStreams...")
            print(f"[DEBUG] Network group before InferVStreams: {self.network_group}")
            print(f"[DEBUG] Input params: {self.input_vstreams_params}")
            print(f"[DEBUG] Output params: {self.output_vstreams_params}")
            try:
                self.infer_pipeline = InferVStreams(
                    self.network_group,
                    self.input_vstreams_params,
                    self.output_vstreams_params
                )
                print(f"[DEBUG] InferVStreams created: {self.infer_pipeline}")
                print(f"[DEBUG] InferVStreams type: {type(self.infer_pipeline)}")
                
                # "Wake up" the pipeline by entering the context manager
                # This creates the missing '_infer_pipeline' attribute
                print(f"[DEBUG] Entering InferVStreams context manager...")
                self.infer_pipeline.__enter__()
                print(f"[DEBUG] InferVStreams context manager entered successfully")
                print(f"[DEBUG] InferVStreams attributes after __enter__: {dir(self.infer_pipeline)}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to create InferVStreams: {e}")
                print(f"[ERROR] Exception type: {type(e)}")
                import traceback
                traceback.print_exc()
                raise
            
            print(f"HAILO model loaded successfully: {self.model_path}")
            print(f"Input shape: {self.input_vstream_info.shape}")
            print(f"Output shape: {self.output_vstream_info.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HAILO: {str(e)}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for HAILO inference
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame ready for HAILO inference
        """
        # Get model input size
        height, width = self.input_vstream_info.shape[:2]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        frame_resized = cv2.resize(frame_rgb, (width, height))
        
        # For FLOAT32 format, normalize to [0, 1] range
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def postprocess_output(self, output_data: np.ndarray, frame_shape: Tuple[int, int],
                         confidence_threshold: float = 0.5,
                         iou_threshold: float = 0.6) -> np.ndarray:
        """
        Postprocess HAILO output to extract detections
        
        Args:
            output_data: Raw output from HAILO inference
            frame_shape: Original frame shape (height, width)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Array of detections [x1, y1, x2, y2, confidence, class_id]
        """
        # HAILO YOLO output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class_id]
        if len(output_data.shape) == 3:
            output_data = output_data[0]  # Remove batch dimension
        
        # Output is already FLOAT32, no need to normalize
        # Filter by confidence
        valid_detections = output_data[output_data[:, 4] >= confidence_threshold]
        
        if len(valid_detections) == 0:
            return np.array([])
        
        # Scale bounding boxes to original frame size
        orig_height, orig_width = frame_shape[:2]
        model_height, model_width = self.input_vstream_info.shape[:2]
        
        scale_x = orig_width / model_width
        scale_y = orig_height / model_height
        
        valid_detections[:, 0] *= scale_x  # x1
        valid_detections[:, 1] *= scale_y  # y1
        valid_detections[:, 2] *= scale_x  # x2
        valid_detections[:, 3] *= scale_y  # y2
        
        # Apply Non-Maximum Suppression
        if len(valid_detections) > 1:
            valid_detections = self._apply_nms(valid_detections, iou_threshold)
        
        return valid_detections
    
    def _apply_nms(self, detections: np.ndarray, iou_threshold: float) -> np.ndarray:
        """Apply Non-Maximum Suppression to detections"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence (descending)
        sorted_indices = np.argsort(detections[:, 4])[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            # Keep the detection with highest confidence
            current = sorted_indices[0]
            keep.append(current)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate IoU with remaining detections
            current_box = detections[current, :4]
            remaining_boxes = detections[sorted_indices[1:], :4]
            
            ious = np.array([self._calculate_iou(current_box, box) for box in remaining_boxes])
            
            # Keep detections with IoU below threshold
            sorted_indices = sorted_indices[1:][ious < iou_threshold]
        
        return detections[keep]
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, frame: np.ndarray, 
              confidence_threshold: float = 0.5,
              iou_threshold: float = 0.6,
              classes: Optional[List[int]] = None) -> Tuple[np.ndarray, float]:
        """
        Perform object detection on frame
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to filter (None for all classes)
            
        Returns:
            Tuple of (detections, inference_time)
            detections: Array of [x1, y1, x2, y2, confidence, class_id]
            inference_time: Inference time in seconds
        """
        start_time = time.time()
        
        # Preprocess frame
        input_data = self.preprocess_frame(frame)
        
        # Prepare input dictionary
        input_dict = {self.input_vstream_info.name: input_data}
        
        # Run inference
        print(f"[DEBUG] Starting HAILO inference...")
        print(f"[DEBUG] Input dict keys: {list(input_dict.keys())}")
        print(f"[DEBUG] Input shape: {input_dict[self.input_vstream_info.name].shape}")
        print(f"[DEBUG] Output stream name: {self.output_vstream_info.name}")
        print(f"[DEBUG] Network group object: {self.network_group}")
        print(f"[DEBUG] Network group type: {type(self.network_group)}")
        print(f"[DEBUG] Infer pipeline object: {self.infer_pipeline}")
        print(f"[DEBUG] Infer pipeline type: {type(self.infer_pipeline)}")
        
        # Check if network group needs to be re-activated
        print(f"[DEBUG] Checking network group activation state...")
        try:
            # Try to get network group status if available
            if hasattr(self.network_group, 'is_active'):
                print(f"[DEBUG] Network group is_active: {self.network_group.is_active()}")
            if hasattr(self.network_group, 'get_state'):
                print(f"[DEBUG] Network group state: {self.network_group.get_state()}")
        except Exception as state_e:
            print(f"[DEBUG] Could not check network group state: {state_e}")
        
        try:
            print(f"[DEBUG] Trying network group activation...")
            activation_context = self.network_group.activate()
            print(f"[DEBUG] Activation context created: {activation_context}")
            with activation_context:
                print(f"[DEBUG] Network group activated, calling infer_pipeline.infer...")
                # Try different inference approaches
                try:
                    print(f"[DEBUG] About to call infer_pipeline.infer with input_dict: {input_dict}")
                    infer_results = self.infer_pipeline.infer(input_dict)
                    print(f"[DEBUG] infer_pipeline.infer returned: {infer_results}")
                except AttributeError as ae:
                    print(f"[DEBUG] Attribute error with infer(): {ae}")
                    # Try alternative method names
                    if hasattr(self.infer_pipeline, 'run'):
                        print(f"[DEBUG] Trying infer_pipeline.run()...")
                        infer_results = self.infer_pipeline.run(input_dict)
                    elif hasattr(self.infer_pipeline, 'predict'):
                        print(f"[DEBUG] Trying infer_pipeline.predict()...")
                        infer_results = self.infer_pipeline.predict(input_dict)
                    else:
                        print(f"[DEBUG] No alternative inference method found")
                        raise ae
                
                print(f"[DEBUG] Inference completed, results keys: {list(infer_results.keys())}")
                output_data = infer_results[self.output_vstream_info.name]
                print(f"[DEBUG] Output data type: {type(output_data)}")
                print(f"[DEBUG] Output data value: {output_data}")
                if hasattr(output_data, 'shape'):
                    print(f"[DEBUG] Output data shape: {output_data.shape}")
                else:
                    print(f"[DEBUG] Output data has no shape attribute, converting to numpy array...")
                    output_data = np.array(output_data)
                    print(f"[DEBUG] Converted output data shape: {output_data.shape}")
        except AttributeError as e:
            # Fallback for different HAILO SDK versions
            print(f"[DEBUG] HAILO API error: {e}")
            print(f"[DEBUG] Exception details: {type(e).__name__}: {str(e)}")
            print("[DEBUG] Trying alternative inference method...")
            try:
                # Try direct inference without network group activation
                print(f"[DEBUG] Trying direct inference...")
                print(f"[DEBUG] Network group state before direct inference: {self.network_group}")
                try:
                    print(f"[DEBUG] About to call direct infer_pipeline.infer...")
                    infer_results = self.infer_pipeline.infer(input_dict)
                    print(f"[DEBUG] Direct infer_pipeline.infer returned: {infer_results}")
                except AttributeError as ae:
                    print(f"[DEBUG] Direct inference attribute error: {ae}")
                    # Try alternative method names
                    if hasattr(self.infer_pipeline, 'run'):
                        print(f"[DEBUG] Trying infer_pipeline.run()...")
                        infer_results = self.infer_pipeline.run(input_dict)
                    elif hasattr(self.infer_pipeline, 'predict'):
                        print(f"[DEBUG] Trying infer_pipeline.predict()...")
                        infer_results = self.infer_pipeline.predict(input_dict)
                    else:
                        print(f"[DEBUG] No alternative direct inference method found")
                        raise ae
                
                print(f"[DEBUG] Direct inference completed, results keys: {list(infer_results.keys())}")
                output_data = infer_results[self.output_vstream_info.name]
                print(f"[DEBUG] Direct output data type: {type(output_data)}")
                if hasattr(output_data, 'shape'):
                    print(f"[DEBUG] Direct output data shape: {output_data.shape}")
                else:
                    print(f"[DEBUG] Direct output data has no shape attribute, converting to numpy array...")
                    output_data = np.array(output_data)
                    print(f"[DEBUG] Converted direct output data shape: {output_data.shape}")
            except Exception as e2:
                print(f"[ERROR] Both inference methods failed: {e2}")
                print(f"[ERROR] Exception type: {type(e2)}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"HAILO inference failed: {e2}")
        except Exception as e:
            print(f"[ERROR] Unexpected error during inference: {e}")
            print(f"[ERROR] Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"HAILO inference failed: {e}")
        
        # Postprocess output
        detections = self.postprocess_output(
            output_data, frame.shape, confidence_threshold, iou_threshold
        )
        
        # Filter by classes if specified
        if classes is not None and len(detections) > 0:
            class_mask = np.isin(detections[:, 5], classes)
            detections = detections[class_mask]
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.last_inference_time = inference_time
        self.inference_times.append(inference_time)
        
        # Keep only last 100 inference times for performance tracking
        if len(self.inference_times) > 100:
            self.inference_times = self.inference_times[-100:]
        
        return detections, inference_time
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times) * 1000,  # ms
            'min_inference_time': np.min(self.inference_times) * 1000,   # ms
            'max_inference_time': np.max(self.inference_times) * 1000,   # ms
            'last_inference_time': self.last_inference_time * 1000,       # ms
            'fps': 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            print(f"[DEBUG] Cleaning up HAILO resources...")
            # Exit the context manager if it was entered
            if hasattr(self, 'infer_pipeline') and self.infer_pipeline:
                print(f"[DEBUG] Exiting InferVStreams context manager...")
                self.infer_pipeline.__exit__(None, None, None)
                del self.infer_pipeline
                print(f"[DEBUG] InferVStreams cleaned up")
            if self.network_group:
                print(f"[DEBUG] Cleaning up network group...")
                del self.network_group
                print(f"[DEBUG] Network group cleaned up")
            if self.target:
                print(f"[DEBUG] Releasing target device...")
                self.target.release()
                print(f"[DEBUG] Target device released")
        except Exception as e:
            print(f"[DEBUG] Error during cleanup: {e}")
            import traceback
            traceback.print_exc()


def create_hailo_detector(model_path: str, config=None) -> HailoYOLODetector:
    """
    Factory function to create HAILO YOLO detector
    
    Args:
        model_path: Path to HAILO HEF model file
        config: Configuration object
        
    Returns:
        HailoYOLODetector instance
    """
    return HailoYOLODetector(model_path, config)


def download_hailo_yolo_model(model_name: str = "yolov8n", save_dir: str = "./") -> str:
    """
    Download pre-compiled HAILO YOLO model
    
    Args:
        model_name: Model name (yolov8n, yolov8s, etc.)
        save_dir: Directory to save the model
        
    Returns:
        Path to downloaded model file
    """
    # Base URLs for HAILO model zoo
    base_url = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.15.0/hailo8l/"
    
    model_urls = {
        "yolov8n": base_url + "yolov8n.hef",
        "yolov8s": base_url + "yolov8s.hef", 
        "yolov5m": base_url + "yolov5m.hef",
        "yolov5s": base_url + "yolov5s.hef"
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(model_urls.keys())}")
    
    model_url = model_urls[model_name]
    model_path = os.path.join(save_dir, f"{model_name}.hef")
    
    # Download model if not exists
    if not os.path.exists(model_path):
        print(f"Downloading {model_name} model from HAILO model zoo...")
        try:
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print(f"Model downloaded successfully: {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")
    else:
        print(f"Model already exists: {model_path}")
    
    return model_path