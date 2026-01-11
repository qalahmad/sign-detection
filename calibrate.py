"""
VocalHands - Quick Calibration Tool
====================================
Add your own hand samples to improve model accuracy.

This tool lets you quickly record a few samples of each sign
using YOUR hands and YOUR lighting conditions, which dramatically
improves real-world detection accuracy.

Usage:
    python calibrate.py

Controls:
    - Press SPACE to capture a sample (captures 5 samples quickly)
    - Press 'n' to move to next sign
    - Press 'p' to move to previous sign  
    - Press 's' to save and retrain model
    - Press 'q' to quit without saving
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from collections import defaultdict

import config
from utils.hand_detector import HandDetector


class Calibrator:
    """Quick calibration tool to supplement training data with user samples."""
    
    def __init__(self, samples_per_capture: int = 5):
        """
        Initialize calibrator.
        
        Args:
            samples_per_capture: Number of samples to capture per SPACE press
        """
        self.detector = HandDetector()
        self.cap = None
        self.samples_per_capture = samples_per_capture
        
        # Calibrate the main alphabet + SPACE
        self.signs = [s for s in config.SIGNS if len(s) == 1]  # A-Z
        self.signs.append("SPACE")  # Add SPACE for calibration
        self.current_sign_idx = 0
        
        # New calibration data
        self.calibration_data = defaultdict(list)
        
        # Load existing calibration counts
        self.existing_counts = self._get_existing_counts()
    
    def _get_existing_counts(self) -> dict:
        """Get count of existing samples per sign."""
        counts = {}
        for sign in self.signs:
            sign_dir = os.path.join(config.DATA_DIR, sign)
            data_file = os.path.join(sign_dir, "landmarks.npy")
            if os.path.exists(data_file):
                data = np.load(data_file)
                counts[sign] = len(data)
            else:
                counts[sign] = 0
        return counts
    
    def capture_samples(self, results) -> int:
        """
        Capture multiple samples quickly.
        
        Args:
            results: MediaPipe detection results
            
        Returns:
            Number of samples captured
        """
        captured = 0
        current_sign = self.signs[self.current_sign_idx]
        
        for _ in range(self.samples_per_capture):
            landmarks = self.detector.extract_landmarks(results, normalize=True)
            if landmarks is not None:
                self.calibration_data[current_sign].append(landmarks.tolist())
                captured += 1
        
        return captured
    
    def save_calibration(self):
        """Merge calibration data with existing dataset."""
        print("\nSaving calibration data...")
        
        for sign, new_samples in self.calibration_data.items():
            if not new_samples:
                continue
                
            sign_dir = os.path.join(config.DATA_DIR, sign)
            os.makedirs(sign_dir, exist_ok=True)
            
            data_file = os.path.join(sign_dir, "landmarks.npy")
            
            # Load existing data
            if os.path.exists(data_file):
                existing = np.load(data_file).tolist()
            else:
                existing = []
            
            # Merge
            combined = existing + new_samples
            
            # Save
            np.save(data_file, np.array(combined))
            
            # Update metadata
            metadata = {
                "sign": sign,
                "num_samples": len(combined),
                "calibration_samples": len(new_samples),
                "features_per_sample": config.FEATURES_PER_HAND,
                "last_updated": datetime.now().isoformat()
            }
            metadata_file = os.path.join(sign_dir, "metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  {sign}: Added {len(new_samples)} samples (total: {len(combined)})")
        
        print("\n[OK] Calibration data saved!")
        return True
    
    def _draw_ui(self, frame: np.ndarray, hand_detected: bool) -> np.ndarray:
        """Draw the calibration UI."""
        h, w, _ = frame.shape
        current_sign = self.signs[self.current_sign_idx]
        new_samples = len(self.calibration_data[current_sign])
        existing = self.existing_counts.get(current_sign, 0)
        
        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 140), (30, 30, 30), -1)
        cv2.rectangle(overlay, (0, h - 60), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame, "VocalHands - Calibration",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_WHITE, 2
        )
        
        # Current sign - LARGE
        cv2.putText(
            frame, f"Show sign: {current_sign}",
            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.5, config.COLOR_CYAN, 3
        )
        
        # Progress
        cv2.putText(
            frame, f"[{self.current_sign_idx + 1}/{len(self.signs)}]",
            (350, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WHITE, 2
        )
        
        # Sample counts
        cv2.putText(
            frame, f"Existing: {existing} | New: {new_samples}",
            (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_GREEN, 2
        )
        
        # Hand detection status
        if hand_detected:
            cv2.putText(
                frame, "READY - Press SPACE to capture",
                (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_GREEN, 2
            )
        else:
            cv2.putText(
                frame, "Show your hand...",
                (w // 2 - 120, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_YELLOW, 2
            )
        
        # Controls
        controls = "SPACE: Capture | N/P: Next/Prev | S: Save & Train | Q: Quit"
        cv2.putText(
            frame, controls,
            (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_WHITE, 1
        )
        
        # Total new samples indicator
        total_new = sum(len(v) for v in self.calibration_data.values())
        cv2.putText(
            frame, f"Total new samples: {total_new}",
            (w - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_YELLOW, 1
        )
        
        return frame
    
    def run(self):
        """Run the calibration interface."""
        print("\n" + "=" * 50)
        print("VocalHands - Quick Calibration")
        print("=" * 50)
        print("\nThis tool lets you add YOUR hand samples to improve accuracy.")
        print(f"Each SPACE press captures {self.samples_per_capture} samples.\n")
        print("Tip: Capture 10-20 samples per sign you want to improve.")
        print("\n" + "=" * 50)
        
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            return
        
        last_results = None
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Detect hands
                frame, results = self.detector.detect_hands(frame)
                last_results = results
                
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame, results)
                
                hand_detected = results.multi_hand_landmarks is not None
                
                # Draw UI
                frame = self._draw_ui(frame, hand_detected)
                
                cv2.imshow("VocalHands - Calibration", frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' ') and hand_detected and last_results:
                    captured = self.capture_samples(last_results)
                    sign = self.signs[self.current_sign_idx]
                    print(f"Captured {captured} samples for '{sign}'")
                    
                elif key == ord('n'):
                    self.current_sign_idx = (self.current_sign_idx + 1) % len(self.signs)
                    print(f"Switched to: {self.signs[self.current_sign_idx]}")
                    
                elif key == ord('p'):
                    self.current_sign_idx = (self.current_sign_idx - 1) % len(self.signs)
                    print(f"Switched to: {self.signs[self.current_sign_idx]}")
                    
                elif key == ord('s'):
                    if sum(len(v) for v in self.calibration_data.values()) > 0:
                        self.save_calibration()
                        print("\nRetraining model with calibration data...")
                        self.cap.release()
                        cv2.destroyAllWindows()
                        
                        # Retrain
                        import subprocess
                        subprocess.run(["python", "train_model.py"], cwd=config.BASE_DIR)
                        return
                    else:
                        print("No calibration samples to save!")
                    
                elif key == ord('q') or key == 27:
                    print("\nQuitting without saving...")
                    break
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()


def main():
    calibrator = Calibrator(samples_per_capture=5)
    calibrator.run()


if __name__ == "__main__":
    main()
