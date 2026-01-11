"""
VocalHands - Hand Detection Utilities
======================================
MediaPipe-based hand landmark detection and feature extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HandDetector:
    """
    MediaPipe Hand Detector for extracting hand landmarks.
    
    This class wraps MediaPipe's hand detection solution and provides
    methods to extract normalized landmark coordinates suitable for
    machine learning models.
    """
    
    def __init__(
        self,
        max_num_hands: int = config.MAX_NUM_HANDS,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
        model_complexity: int = config.MODEL_COMPLEXITY
    ):
        """
        Initialize the HandDetector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            model_complexity: Model complexity (0=lite, 1=full)
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
        )
        
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, any]:
        """
        Detect hands in an image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            Tuple of (processed_image, results)
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Improve performance by marking image as not writeable
        image_rgb.flags.writeable = False
        results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        return image, results
    
    def extract_landmarks(
        self, 
        results, 
        normalize: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extract hand landmarks as a flat numpy array.
        
        Each hand has 21 landmarks with (x, y, z) coordinates.
        The landmarks are normalized relative to the hand's bounding box
        for scale and position invariance.
        
        Args:
            results: MediaPipe hand detection results
            normalize: Whether to normalize landmarks relative to bounding box
            
        Returns:
            Numpy array of shape (63,) for single hand, or None if no hand detected
        """
        if not results.multi_hand_landmarks:
            return None
        
        # Get the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract raw coordinates
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks = np.array(landmarks)
        
        if normalize:
            landmarks = self._normalize_landmarks(landmarks)
        
        return landmarks
    
    def extract_all_hands_landmarks(
        self, 
        results, 
        normalize: bool = True
    ) -> List[np.ndarray]:
        """
        Extract landmarks from all detected hands.
        
        Args:
            results: MediaPipe hand detection results
            normalize: Whether to normalize landmarks
            
        Returns:
            List of numpy arrays, one per detected hand
        """
        if not results.multi_hand_landmarks:
            return []
        
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks = np.array(landmarks)
            if normalize:
                landmarks = self._normalize_landmarks(landmarks)
            
            all_landmarks.append(landmarks)
        
        return all_landmarks
    
    def _normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks for position and scale invariance.
        
        This centers the landmarks around the wrist (landmark 0) and
        scales them based on the palm size for better generalization.
        
        Args:
            landmarks: Raw landmark array of shape (63,)
            
        Returns:
            Normalized landmark array
        """
        # Reshape to (21, 3) for easier manipulation
        coords = landmarks.reshape(-1, 3)
        
        # Use wrist (landmark 0) as the reference point
        wrist = coords[0].copy()
        
        # Center around wrist
        coords_centered = coords - wrist
        
        # Calculate scale based on distance from wrist to middle finger MCP (landmark 9)
        # This provides a consistent scale reference
        palm_size = np.linalg.norm(coords_centered[9])
        
        if palm_size > 0:
            coords_normalized = coords_centered / palm_size
        else:
            coords_normalized = coords_centered
        
        return coords_normalized.flatten()
    
    def draw_landmarks(
        self, 
        image: np.ndarray, 
        results,
        draw_connections: bool = True
    ) -> np.ndarray:
        """
        Draw hand landmarks on the image.
        
        Args:
            image: BGR image to draw on
            results: MediaPipe hand detection results
            draw_connections: Whether to draw connections between landmarks
            
        Returns:
            Image with landmarks drawn
        """
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw_connections:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                else:
                    self.mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        None,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        None
                    )
        
        return image
    
    def get_hand_bbox(
        self, 
        results, 
        image_shape: Tuple[int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Get bounding box of the detected hand.
        
        Args:
            results: MediaPipe hand detection results
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max) or None
        """
        if not results.multi_hand_landmarks:
            return None
        
        h, w, _ = image_shape
        hand_landmarks = results.multi_hand_landmarks[0]
        
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]
        
        padding = 20
        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)
        
        return (x_min, y_min, x_max, y_max)
    
    def get_handedness(self, results) -> Optional[str]:
        """
        Get which hand was detected (Left or Right).
        
        Args:
            results: MediaPipe hand detection results
            
        Returns:
            'Left', 'Right', or None
        """
        if not results.multi_handedness:
            return None
        
        return results.multi_handedness[0].classification[0].label
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Landmark indices for reference
LANDMARK_NAMES = {
    0: "WRIST",
    1: "THUMB_CMC",
    2: "THUMB_MCP",
    3: "THUMB_IP",
    4: "THUMB_TIP",
    5: "INDEX_FINGER_MCP",
    6: "INDEX_FINGER_PIP",
    7: "INDEX_FINGER_DIP",
    8: "INDEX_FINGER_TIP",
    9: "MIDDLE_FINGER_MCP",
    10: "MIDDLE_FINGER_PIP",
    11: "MIDDLE_FINGER_DIP",
    12: "MIDDLE_FINGER_TIP",
    13: "RING_FINGER_MCP",
    14: "RING_FINGER_PIP",
    15: "RING_FINGER_DIP",
    16: "RING_FINGER_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}
