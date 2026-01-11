"""
VocalHands - Sign Language Detection Configuration
===================================================
Central configuration file for all project settings.
"""

import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================
# ASL Alphabet signs to detect (matches Kaggle ASL Alphabet dataset)
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data
SIGNS = [
    # ASL Alphabet (A-Z)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # Extra signs from Kaggle dataset
    "SPACE", "DELETE",
    # Note: "NOTHING" excluded (only 1 sample)
]

# Number of samples to collect per sign
SAMPLES_PER_SIGN = 100

# =============================================================================
# MEDIAPIPE CONFIGURATION
# =============================================================================
# Hand detection settings
MAX_NUM_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7
MODEL_COMPLEXITY = 1  # 0 = lite, 1 = full

# Number of landmarks per hand (MediaPipe provides 21 landmarks)
NUM_LANDMARKS = 21
# Each landmark has x, y, z coordinates
FEATURES_PER_LANDMARK = 3
# Total features per hand
FEATURES_PER_HAND = NUM_LANDMARKS * FEATURES_PER_LANDMARK  # 63

# =============================================================================
# MODEL CONFIGURATION  
# =============================================================================
# KNN Parameters - tuned for better real-world generalization
KNN_N_NEIGHBORS = 7       # Higher K = more robust to noise (was 5)
KNN_WEIGHTS = "distance"  # 'uniform' or 'distance' - distance works better
KNN_ALGORITHM = "auto"    # 'auto', 'ball_tree', 'kd_tree', 'brute'
KNN_METRIC = "euclidean"  # Distance metric

# Model file paths
KNN_MODEL_PATH = os.path.join(MODEL_DIR, "knn_sign_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# =============================================================================
# CAMERA CONFIGURATION
# =============================================================================
CAMERA_INDEX = 0  # Default camera (0 for built-in, 1+ for external)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# =============================================================================
# UI CONFIGURATION
# =============================================================================
# Colors (BGR format for OpenCV)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_PURPLE = (255, 0, 255)

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 2

# Detection display settings
PREDICTION_SMOOTHING_WINDOW = 5  # Number of frames to smooth predictions (lower = faster response)
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence to display prediction (lower = more sensitive)
