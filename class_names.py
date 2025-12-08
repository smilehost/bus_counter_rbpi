"""
COCO dataset class names for YOLO models
This module contains the mapping between class IDs and class names for the COCO dataset
"""

# COCO dataset class names (80 classes)
COCO_CLASS_NAMES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

def get_class_name(class_id):
    """
    Get class name by class ID
    
    Args:
        class_id (int): Class ID from YOLO detection
        
    Returns:
        str: Class name or 'unknown' if not found
    """
    return COCO_CLASS_NAMES.get(class_id, 'unknown')

def get_all_class_names():
    """
    Get all COCO class names
    
    Returns:
        list: List of all class names
    """
    return list(COCO_CLASS_NAMES.values())

def print_all_classes():
    """Print all COCO classes with their IDs"""
    print("COCO Dataset Classes:")
    print("=" * 40)
    for class_id, class_name in COCO_CLASS_NAMES.items():
        print(f"{class_id:2d}: {class_name}")
    print("=" * 40)
    print(f"Total: {len(COCO_CLASS_NAMES)} classes")