# YOLO-BoTSORT Multi-Object Tracking System

A comprehensive multi-object tracking system that combines YOLO detection with BoTSORT tracking for bus counting and general object tracking applications. Optimized for both Windows development and Raspberry Pi 5 deployment with **HAILO 8L AI Accelerator** support.

## Features

- **YOLO Detection**: Uses Ultralytics YOLOv8 for fast and accurate object detection
- **HAILO 8L Acceleration**: Hardware-accelerated inference on Raspberry Pi 5
- **BoTSORT Tracking**: Advanced tracking algorithm with ReID capabilities
- **Bus Counting**: Specialized counting functionality with directional tracking
- **Real-time Processing**: Optimized for real-time video processing
- **Cross-platform**: Works on Windows for development and Raspberry Pi 5 for deployment
- **Pi5 cam0 Focus**: Optimized for cam0 on Raspberry Pi 5
- **Visualization**: Rich visualization with trails, IDs, and statistics
- **Performance Monitoring**: Built-in FPS and performance tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for development)
- Raspberry Pi 5 with HAILO 8L AI Accelerator (for deployment)
- HAILO SDK installed on Pi5

### Install Dependencies

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### HAILO 8L Setup (Pi5 Deployment)

1. **Install HAILO SDK** on Raspberry Pi 5:
```bash
# Follow HAILO official installation guide
# Typically involves:
sudo apt update
sudo apt install hailort-all
```

2. **Verify HAILO Installation**:
```bash
hailortcli query
```

3. **Download HAILO Models**:
```bash
python hailo_example_usage.py  # Select option 2 to download models
```

### Manual Installation (if requirements.txt fails)

```bash
# Core dependencies
pip install opencv-python==4.10.0.84
pip install pandas==2.2.2
pip install ultralytics==8.3.0
pip install cvzone==1.6.1
pip install "numpy<2.3.0"

# PyTorch (CUDA version for GPU)
pip install torch==2.3.1+cu121
pip install torchvision==0.18.1+cu121

# Tracking and ML dependencies
pip install cython==3.0.11
pip install filterpy==1.4.5
pip install scikit-learn==1.5.0
pip install motmetrics==1.2.0
pip install "lap>=0.5.0"
pip install scipy==1.14.0

# Face recognition (optional)
pip install deepface==0.0.96
pip install "tensorflow>=2.20.0"
pip install "tf-keras>=2.20.1"
pip install "mtcnn>=1.0.0"
pip install "retina-face>=0.0.17"

# System monitoring
pip install "psutil>=5.8.0"
```

### Raspberry Pi 5 with HAILO 8L Installation

For Raspberry Pi 5 deployment with HAILO 8L AI Accelerator:

```bash
# Install HAILO-specific dependencies
pip install hailort>=4.17.0
pip install hailo-platform>=4.17.0

# Install system packages for GStreamer camera support
sudo apt-get update
sudo apt-get install python3-gi python3-gst-1.0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools gstreamer1.0-libav
sudo apt-get install libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev

# For Raspberry Pi camera specifically
sudo apt-get install gstreamer1.0-rpicamsrc

# Install CPU version of PyTorch (HAILO handles inference)
pip install torch==2.3.1+cpu
pip install torchvision==0.18.1+cpu

# Install other dependencies
pip install -r requirements.txt
```

### GStreamer Camera Support

The project now uses GStreamer instead of picamera2 for Raspberry Pi camera support, which is compatible with Python 3.10:

**Benefits:**
- Compatible with Python 3.10 (no need for Python 3.11+)
- Better performance and stability
- Hardware-accelerated video processing
- More flexible pipeline configuration

**GStreamer Pipeline Example:**
```
rpicamsrc ! video/x-raw,width=1920,height=1080,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1
```

## Quick Start

### Basic Usage

#### Development (Windows/Linux with GPU)
1. **Webcam Tracking**:
```bash
python yolo_botsort_tracker.py
```

2. **Video File Processing**:
```bash
python yolo_botsort_tracker.py --source path/to/video.mp4 --output output_video.mp4
```

#### HAILO Deployment (Raspberry Pi 5)
1. **HAILO + cam0 Tracking**:
```bash
python yolo_botsort_tracker.py  # Automatically uses HAILO if available
```

2. **HAILO Examples**:
```bash
python hailo_example_usage.py
```

This will show a menu with HAILO-specific examples:
- HAILO + cam0 tracking
- Download HAILO models
- HAILO performance test
- Camera and system info

### Example Usage

#### Regular Examples
```bash
python example_usage.py
```

#### HAILO Examples
```bash
python hailo_example_usage.py
```

## Configuration

### Main Configuration (`config.py`)

The configuration automatically detects the platform and optimizes settings:

#### HAILO Settings (Pi5)
```python
# Automatically set for Pi5 with HAILO
USE_HAILO = True
YOLO_MODEL = "yolov8n.hef"  # HAILO compiled model
HAILO_DEVICE = "hailo0"
VIDEO_SOURCE = "/dev/video0"  # cam0
CAMERA_TYPE = "rpi"  # or "usb" for USB camera (uses GStreamer for RPi camera)
ENABLE_REID = False  # Disabled for performance

# Pi5 Performance Optimizations
DISPLAY_SIZE = (1024, 768)
PROCESS_EVERY_N_FRAMES = 2
MAX_DETECTIONS = 50
```

#### Regular YOLO Settings (Development)
```python
# Automatically set for development
USE_HAILO = False
YOLO_MODEL = "yolo11n.pt"  # Regular PyTorch model
VIDEO_SOURCE = 0  # Webcam
ENABLE_REID = True

# Development Settings
DISPLAY_SIZE = (1280, 720)
PROCESS_EVERY_N_FRAMES = 1
MAX_DETECTIONS = 100
```

### Command Line Arguments

```bash
python yolo_botsort_tracker.py [OPTIONS]

Options:
  --source SOURCE      Video source (camera index or file path)
  --output OUTPUT      Output video file path
  --model MODEL        YOLO model path
  --conf CONF          Detection confidence threshold
  --no-save           Do not save output video
  --no-display        Do not display video
```

## File Structure

```
bus_counter_rbpi/
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
├── yolo_botsort_tracker.py   # Main tracking script
├── botsort_tracker.py        # BoTSORT implementation
├── utils.py                  # Utility functions
├── example_usage.py          # Example usage scripts
└── readme.md                 # This file
```

## Performance Optimization

### For Development (Windows with GPU)

- Use CUDA-enabled PyTorch
- Enable ReID for better tracking accuracy
- Use larger YOLO models (yolov8s.pt, yolov8m.pt)
- Process every frame for maximum accuracy

### For Deployment (Raspberry Pi 5 with HAILO 8L)

- Use HAILO-compiled HEF models
- Disable ReID for better performance
- Use HAILO 8L hardware acceleration
- Focus on cam0 for optimal performance
- Process every 1-2 frames (HAILO is fast enough)
- Use optimized display resolution

Example HAILO Pi5 optimized settings:
```python
USE_HAILO = True
YOLO_MODEL = "yolov8n.hef"  # HAILO compiled model
VIDEO_SOURCE = "/dev/video0"  # cam0
PROCESS_EVERY_N_FRAMES = 1  # HAILO can handle every frame
DISPLAY_SIZE = (1024, 768)
BOTSORT_TRACKER['with_reid'] = False
```

## Bus Counting Features

### Directional Counting

The system can count objects moving in specific directions:

```python
COUNTING_DIRECTION = "down"   # Count objects moving down
COUNTING_DIRECTION = "up"     # Count objects moving up
COUNTING_DIRECTION = "both"    # Count both directions
```

### Counting Line

A horizontal line divides the frame for counting. Objects crossing this line are counted based on their movement direction.

### Track Age Filtering

Minimum track age prevents false counts from short-lived detections:

```python
MIN_TRACK_AGE = 10  # Require 10 frames before counting
```

## Visualization Features

- **Bounding Boxes**: Color-coded boxes for each tracked object
- **Track IDs**: Unique ID and confidence score for each track
- **Movement Trails**: Visual trails showing object movement history
- **Counting Line**: Visual indication of counting boundary
- **Statistics**: Real-time FPS, counts, and active tracks
- **Center Points**: Center points of tracked objects

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce input resolution
   - Use smaller YOLO model
   - Process every N frames

2. **Low FPS on Raspberry Pi**:
   - Use CPU PyTorch version
   - Disable ReID
   - Increase frame skipping
   - Reduce display resolution

3. **Poor Tracking Accuracy**:
   - Increase confidence threshold
   - Enable ReID
   - Adjust tracking thresholds
   - Use larger YOLO model

4. **Camera Not Found**:
     - Check camera index (try 0, 1, 2...)
     - Verify camera drivers
     - Test with other applications
     - For Pi5, check `/dev/video0` device permissions
     - For RPi camera with GStreamer, ensure gstreamer1.0-rpicamsrc is installed
     - Test GStreamer pipeline: `gst-launch-1.0 rpicamsrc ! videoconvert ! autovideosink`

5. **HAILO Issues**:
    - Verify HAILO SDK installation: `hailortcli query`
    - Check HAILO device permissions
    - Ensure HEF model files are compatible
    - Monitor HAILO temperature and power

6. **Pi5 Performance Issues**:
    - Use HAILO models instead of regular YOLO
    - Reduce display resolution
    - Disable ReID feature
    - Check camera format compatibility

### Performance Tips

1. **GPU Usage**:
   - Install CUDA toolkit
   - Use CUDA PyTorch versions
   - Monitor GPU memory usage

2. **CPU Optimization**:
   - Limit number of detections
   - Increase frame skipping
   - Use smaller models

3. **Memory Management**:
   - Clear track history periodically
   - Limit trail length
   - Use appropriate buffer sizes

## API Reference

### Main Classes

- **`YOLOBoTSORTTracker`**: Main tracking class with HAILO support
- **`HailoYOLODetector`**: HAILO-optimized YOLO detector
- **`Pi5Camera`**: Raspberry Pi 5 camera handler
- **`BoTSORT`**: Tracking algorithm implementation
- **`Track`**: Individual track management
- **`Visualizer`**: Visualization utilities
- **`BusCounter`**: Counting functionality
- **`PerformanceMonitor`**: Performance tracking

### Key Methods

```python
# Initialize tracker (auto-detects HAILO)
tracker = YOLOBoTSORTTracker()

# Process video (auto-detects camera type)
tracker.process_video(source, output_path)

# Process single frame
processed_frame = tracker.process_frame(frame, frame_count)

# Initialize HAILO detector
from hailo_yolo_detector import HailoYOLODetector
detector = HailoYOLODetector("yolov8n.hef", config)

# Initialize Pi5 camera (auto-detects GStreamer for RPi camera)
from pi5_camera import create_pi5_camera
camera = create_pi5_camera(camera_type="auto", camera_index="/dev/video0")

# Test camera functionality
python test_gstreamer_camera.py
```

## HAILO 8L Specific Features

### Model Management
- Automatic HAILO model download from HAILO Model Zoo
- Support for YOLOv8, YOLOv5 variants
- Hardware-accelerated inference pipeline

### Performance Optimization
- Batch processing support
- Memory-efficient inference
- Real-time performance monitoring
- Automatic fallback to CPU if HAILO unavailable

### Camera Integration
- GStreamer-based Pi5 camera module support (Python 3.10 compatible)
- USB camera optimization for cam0
- Automatic camera type detection
- Hardware-specific optimizations
- No dependency on picamera2 (requires Python 3.11+)
