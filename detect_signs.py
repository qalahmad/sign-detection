"""
VocalHands - Real-Time Sign Detection
======================================
Detect sign language gestures in real-time using webcam.

Usage:
    python detect_signs.py

Controls:
    - Press 'q' or ESC to quit
    - Press 'c' to clear the sentence buffer
    - Press 'SPACE' to add a space to the sentence
    - Press 'BACKSPACE' to remove the last character
"""

import cv2
import numpy as np
import pickle
import time
from collections import deque
from typing import Optional, Tuple

import config
from utils.hand_detector import HandDetector


class SignDetector:
    """
    Real-time sign language detector using trained KNN model.
    
    Features:
    - Real-time hand landmark detection
    - Prediction smoothing for stability
    - Sentence building from detected signs
    - Visual feedback with confidence scores
    """
    
    def __init__(self):
        """Initialize the sign detector."""
        self.detector = HandDetector()
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.cap = None
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=config.PREDICTION_SMOOTHING_WINDOW)
        
        # Sentence building
        self.sentence = ""
        self.last_prediction = None
        self.same_prediction_count = 0
        self.add_threshold = 15  # Frames of same prediction to add to sentence (hold sign ~0.5 sec)
        
        # Load the trained model
        self._load_model()
    
    def _load_model(self):
        """Load the trained KNN model from disk."""
        try:
            with open(config.KNN_MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.label_encoder = model_data["label_encoder"]
            
            print(f"[OK] Model loaded successfully")
            print(f"  Classes: {list(self.label_encoder.classes_)}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {config.KNN_MODEL_PATH}\n"
                "Please train the model first:\n"
                "  1. Run 'python collect_data.py' to collect data\n"
                "  2. Run 'python train_model.py' to train the model"
            )
    
    def predict(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict the sign from landmarks.
        
        Args:
            landmarks: Normalized landmark array
            
        Returns:
            Tuple of (predicted_sign, confidence)
        """
        # Scale the features
        features = self.scaler.transform(landmarks.reshape(1, -1))
        
        # Get prediction and probabilities
        prediction_idx = self.model.predict(features)[0]
        
        # Get distances to k nearest neighbors for confidence estimation
        distances, indices = self.model.kneighbors(features)
        
        # Calculate confidence based on neighbor distances
        # Smaller distances = higher confidence
        mean_distance = np.mean(distances[0])
        confidence = 1 / (1 + mean_distance)  # Normalize to 0-1 range
        
        # Get the class label
        predicted_sign = self.label_encoder.inverse_transform([prediction_idx])[0]
        
        return predicted_sign, confidence
    
    def get_smoothed_prediction(
        self, 
        prediction: str, 
        confidence: float
    ) -> Tuple[Optional[str], float]:
        """
        Get smoothed prediction using temporal voting.
        
        Args:
            prediction: Current frame prediction
            confidence: Current confidence score
            
        Returns:
            Tuple of (smoothed_prediction, average_confidence)
        """
        self.prediction_buffer.append((prediction, confidence))
        
        if len(self.prediction_buffer) < config.PREDICTION_SMOOTHING_WINDOW // 2:
            return None, 0.0
        
        # Count predictions
        prediction_counts = {}
        confidence_sums = {}
        
        for pred, conf in self.prediction_buffer:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            confidence_sums[pred] = confidence_sums.get(pred, 0) + conf
        
        # Get most common prediction
        most_common = max(prediction_counts, key=prediction_counts.get)
        count = prediction_counts[most_common]
        avg_confidence = confidence_sums[most_common] / count
        
        # Only return if it's the majority
        if count >= len(self.prediction_buffer) * 0.5:
            return most_common, avg_confidence
        
        return None, 0.0
    
    def update_sentence(self, prediction: str) -> bool:
        """
        Update the sentence buffer based on prediction.
        
        Args:
            prediction: The detected sign
            
        Returns:
            True if a new character was added
        """
        if prediction == self.last_prediction:
            self.same_prediction_count += 1
        else:
            self.same_prediction_count = 0
            self.last_prediction = prediction
        
        # Add to sentence if held for enough frames
        if self.same_prediction_count == self.add_threshold:
            self.sentence += prediction
            return True
        
        return False
    
    def _draw_ui(
        self, 
        frame: np.ndarray, 
        prediction: Optional[str],
        confidence: float,
        hand_detected: bool,
        raw_prediction: Optional[str] = None,
        raw_confidence: float = 0.0
    ) -> np.ndarray:
        """Draw the user interface overlay."""
        h, w, _ = frame.shape
        
        # Create semi-transparent overlays
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 100), (30, 30, 30), -1)
        
        # Bottom sentence bar
        cv2.rectangle(overlay, (0, h - 100), (w, h), (30, 30, 30), -1)
        
        # Prediction box (right side) - made larger
        cv2.rectangle(overlay, (w - 280, 110), (w - 10, 320), (40, 40, 40), -1)
        
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame, "VocalHands - Sign Detection",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_WHITE, 2
        )
        
        # Status indicator
        if hand_detected:
            status = "Hand Detected"
            status_color = config.COLOR_GREEN
        else:
            status = "Show your hand sign"
            status_color = config.COLOR_YELLOW
        
        cv2.putText(
            frame, status,
            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2
        )
        
        # Raw detection display (what the model sees right now)
        cv2.putText(
            frame, "Live Detection:",
            (w - 270, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1
        )
        
        if raw_prediction and raw_confidence > 0.1:
            cv2.putText(
                frame, raw_prediction,
                (w - 270, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_YELLOW, 2
            )
            cv2.putText(
                frame, f"({raw_confidence * 100:.0f}%)",
                (w - 120, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1
            )
        else:
            cv2.putText(
                frame, "---",
                (w - 250, 175), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 2
            )
        
        # Confirmed/Smoothed prediction display
        cv2.putText(
            frame, "Confirmed Sign:",
            (w - 270, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1
        )
        
        if prediction and confidence >= config.CONFIDENCE_THRESHOLD:
            # Large prediction text
            cv2.putText(
                frame, prediction,
                (w - 220, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, config.COLOR_CYAN, 3
            )
            
            # Confidence bar
            bar_width = int(200 * confidence)
            cv2.rectangle(frame, (w - 270, 280), (w - 70, 300), config.COLOR_WHITE, 2)
            
            # Color based on confidence
            if confidence >= 0.7:
                bar_color = config.COLOR_GREEN
            elif confidence >= 0.4:
                bar_color = config.COLOR_YELLOW
            else:
                bar_color = config.COLOR_RED
            
            cv2.rectangle(frame, (w - 268, 282), (w - 268 + bar_width, 298), bar_color, -1)
            
            # Confidence percentage
            cv2.putText(
                frame, f"{confidence * 100:.0f}%",
                (w - 65, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1
            )
        else:
            cv2.putText(
                frame, "-",
                (w - 180, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (100, 100, 100), 3
            )
        
        # Sentence display
        cv2.putText(
            frame, "Sentence:",
            (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_WHITE, 1
        )
        
        # Show sentence with cursor
        sentence_display = self.sentence + "_" if len(self.sentence) < 50 else "..." + self.sentence[-47:] + "_"
        cv2.putText(
            frame, sentence_display,
            (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, config.COLOR_CYAN, 2
        )
        
        # Controls hint
        controls = "Q: Quit | C: Clear | SPACE: Add space | BACKSPACE: Delete"
        cv2.putText(
            frame, controls,
            (w - 500, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1
        )
        
        # Progress indicator for sentence building (shows how long you need to hold)
        if prediction and self.same_prediction_count > 0:
            progress = min(self.same_prediction_count / self.add_threshold, 1.0)
            indicator_width = int(150 * progress)
            
            # Label
            cv2.putText(
                frame, "Hold to add:",
                (20, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1
            )
            # Background
            cv2.rectangle(frame, (130, h - 107), (280, h - 90), (50, 50, 50), -1)
            # Progress
            if progress >= 1.0:
                cv2.rectangle(frame, (130, h - 107), (130 + indicator_width, h - 90), config.COLOR_GREEN, -1)
            else:
                cv2.rectangle(frame, (130, h - 107), (130 + indicator_width, h - 90), config.COLOR_YELLOW, -1)
        
        return frame
    
    def run(self):
        """Run the real-time sign detection."""
        print("\n" + "=" * 50)
        print("VocalHands - Real-Time Detection")
        print("=" * 50)
        print("\nControls:")
        print("  Q/ESC     - Quit")
        print("  C         - Clear sentence")
        print("  SPACE     - Add space to sentence")
        print("  BACKSPACE - Remove last character")
        print("\n" + "=" * 50)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            return
        
        print("\nCamera initialized. Starting detection...")
        print("Show your hand signs to the camera!\n")
        
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    continue
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                
                # Track FPS
                current_time = time.time()
                fps_counter.append(1 / (current_time - last_time + 1e-6))
                last_time = current_time
                fps = np.mean(fps_counter)
                
                # Detect hands
                frame, results = self.detector.detect_hands(frame)
                
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame, results)
                
                # Make prediction if hand detected
                prediction = None
                confidence = 0.0
                raw_prediction = None
                raw_confidence = 0.0
                hand_detected = results.multi_hand_landmarks is not None
                
                if hand_detected:
                    landmarks = self.detector.extract_landmarks(results, normalize=True)
                    
                    if landmarks is not None:
                        raw_prediction, raw_confidence = self.predict(landmarks)
                        prediction, confidence = self.get_smoothed_prediction(
                            raw_prediction, raw_confidence
                        )
                        
                        if prediction and confidence >= config.CONFIDENCE_THRESHOLD:
                            added = self.update_sentence(prediction)
                            if added:
                                print(f"Added: {prediction} | Sentence: {self.sentence}")
                else:
                    # Clear prediction buffer when no hand
                    self.prediction_buffer.clear()
                    self.same_prediction_count = 0
                
                # Draw UI with both raw and smoothed predictions
                frame = self._draw_ui(frame, prediction, confidence, hand_detected, raw_prediction, raw_confidence)
                
                # FPS display
                cv2.putText(
                    frame, f"FPS: {fps:.0f}",
                    (frame.shape[1] - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1
                )
                
                # Display
                cv2.imshow("VocalHands - Sign Detection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Quit
                    break
                    
                elif key == ord('c'):  # Clear
                    self.sentence = ""
                    print("Sentence cleared")
                    
                elif key == ord(' '):  # Space
                    self.sentence += " "
                    print(f"Space added | Sentence: {self.sentence}")
                    
                elif key == 8:  # Backspace
                    if self.sentence:
                        self.sentence = self.sentence[:-1]
                        print(f"Deleted | Sentence: {self.sentence}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            
            if self.sentence:
                print(f"\nFinal sentence: {self.sentence}")


def main():
    """Main entry point."""
    try:
        detector = SignDetector()
        detector.run()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()
