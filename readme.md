# YOLO-BoTSORT Multi-Object Tracking System

A comprehensive multi-object tracking system that combines YOLO detection with BoTSORT tracking for bus counting and general object tracking applications. Optimized for both Windows development and Raspberry Pi 5 deployment.

## Features

- **YOLO Detection**: Uses Ultralytics YOLOv8 for fast and accurate object detection
- **BoTSORT Tracking**: Advanced tracking algorithm with ReID capabilities
- **Bus Counting**: Specialized counting functionality with directional tracking
- **Real-time Processing**: Optimized for real-time video processing
- **Cross-platform**: Works on Windows for development and Raspberry Pi 5 for deployment
- **Visualization**: Rich visualization with trails, IDs, and statistics
- **Performance Monitoring**: Built-in FPS and performance tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for development)
- Raspberry Pi 5 (for deployment)

### Install Dependencies

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
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

### Raspberry Pi 5 Specific Installation

For Raspberry Pi 5 deployment, use CPU versions of PyTorch:

```bash
# Install PyTorch CPU version for Pi 5
pip install torch==2.3.1+cpu
pip install torchvision==0.18.1+cpu

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

1. **Webcam Tracking**:
```bash
python yolo_botsort_tracker.py
```

2. **Video File Processing**:
```bash
python yolo_botsort_tracker.py --source path/to/video.mp4 --output output_video.mp4
```

3. **Custom Settings**:
```bash
python yolo_botsort_tracker.py --source video.mp4 --model yolov8s.pt --conf 0.7
```

### Example Usage

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

This will show a menu with different examples:
- Webcam tracking
- Video file processing
- Custom settings
- Performance optimization
- Raspberry Pi 5 optimization

## Configuration

### Main Configuration (`config.py`)

Key configuration options:

```python
# YOLO Settings
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
YOLO_CONFIDENCE = 0.5
YOLO_CLASSES = [0]  # Person class (COCO dataset)

# BoTSORT Settings
BOTSORT_TRACKER = {
    'track_high_thresh': 0.6,
    'track_low_thresh': 0.1,
    'track_buffer': 30,
    'with_reid': True,
    'frame_rate': 30
}

# Video Processing
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
OUTPUT_VIDEO = "output_tracking.mp4"
SAVE_VIDEO = True
SHOW_VIDEO = True

# Bus Counting
COUNTING_DIRECTION = "down"  # "up", "down", or "both"
MIN_TRACK_AGE = 10
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

### For Deployment (Raspberry Pi 5)

- Use CPU versions of PyTorch
- Disable ReID for better performance
- Use smaller YOLO models (yolov8n.pt)
- Process every 2-3 frames for better FPS
- Reduce display resolution

Example Pi 5 optimized settings:
```python
YOLO_MODEL = "yolov8n.pt"
PROCESS_EVERY_N_FRAMES = 3
DISPLAY_SIZE = (800, 600)
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

- **`YOLOBoTSORTTracker`**: Main tracking class
- **`BoTSORT`**: Tracking algorithm implementation
- **`Track`**: Individual track management
- **`Visualizer`**: Visualization utilities
- **`BusCounter`**: Counting functionality
- **`PerformanceMonitor`**: Performance tracking

### Key Methods

```python
# Initialize tracker
tracker = YOLOBoTSORTTracker()

# Process video
tracker.process_video(source, output_path)

# Process single frame
processed_frame = tracker.process_frame(frame, frame_count)
```
