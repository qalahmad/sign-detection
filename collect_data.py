"""
VocalHands - Data Collection Script
====================================
Collect hand landmark data for training the sign language model.

Usage:
    python collect_data.py

Controls:
    - Press SPACE to start/stop recording samples for current sign
    - Press 'n' to move to next sign
    - Press 'p' to move to previous sign
    - Press 's' to save current progress
    - Press 'q' or ESC to quit and save
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


class DataCollector:
    """
    Interactive data collection tool for sign language recognition.
    
    Collects hand landmark data samples for each sign in the vocabulary,
    storing them in an organized directory structure.
    """
    
    def __init__(self):
        """Initialize the data collector."""
        self.detector = HandDetector()
        self.cap = None
        
        # Data storage
        self.data = defaultdict(list)  # {sign: [landmarks_array, ...]}
        self.current_sign_idx = 0
        self.is_recording = False
        
        # Load existing data if available
        self._load_existing_data()
        
    def _load_existing_data(self):
        """Load any existing collected data."""
        for sign in config.SIGNS:
            sign_dir = os.path.join(config.DATA_DIR, sign)
            data_file = os.path.join(sign_dir, "landmarks.npy")
            
            if os.path.exists(data_file):
                existing_data = np.load(data_file)
                self.data[sign] = existing_data.tolist()
                print(f"Loaded {len(self.data[sign])} existing samples for '{sign}'")
    
    def _save_data(self):
        """Save all collected data to disk."""
        for sign, landmarks_list in self.data.items():
            if len(landmarks_list) > 0:
                sign_dir = os.path.join(config.DATA_DIR, sign)
                os.makedirs(sign_dir, exist_ok=True)
                
                # Save as numpy array
                data_file = os.path.join(sign_dir, "landmarks.npy")
                np.save(data_file, np.array(landmarks_list))
                
                # Save metadata
                metadata = {
                    "sign": sign,
                    "num_samples": len(landmarks_list),
                    "features_per_sample": config.FEATURES_PER_HAND,
                    "last_updated": datetime.now().isoformat()
                }
                metadata_file = os.path.join(sign_dir, "metadata.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Save global dataset info
        dataset_info = {
            "signs": config.SIGNS,
            "total_samples": sum(len(v) for v in self.data.values()),
            "samples_per_sign": {sign: len(self.data[sign]) for sign in config.SIGNS},
            "features_per_sample": config.FEATURES_PER_HAND,
            "last_updated": datetime.now().isoformat()
        }
        info_file = os.path.join(config.DATA_DIR, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print("\n[OK] Data saved successfully!")
        print(f"  Total samples: {dataset_info['total_samples']}")
    
    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw the user interface overlay."""
        h, w, _ = frame.shape
        current_sign = config.SIGNS[self.current_sign_idx]
        num_samples = len(self.data[current_sign])
        
        # Create semi-transparent overlay for status bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 30), -1)
        cv2.rectangle(overlay, (0, h - 80), (w, h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(
            frame, "VocalHands - Data Collection",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, config.COLOR_WHITE, 2
        )
        
        # Current sign display
        sign_text = f"Sign: {current_sign}"
        cv2.putText(
            frame, sign_text,
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.2, config.COLOR_CYAN, 2
        )
        
        # Progress indicator
        progress_text = f"[{self.current_sign_idx + 1}/{len(config.SIGNS)}]"
        cv2.putText(
            frame, progress_text,
            (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_WHITE, 2
        )
        
        # Sample counter
        samples_text = f"Samples: {num_samples}/{config.SAMPLES_PER_SIGN}"
        color = config.COLOR_GREEN if num_samples >= config.SAMPLES_PER_SIGN else config.COLOR_YELLOW
        cv2.putText(
            frame, samples_text,
            (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
        )
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(frame, (w - 50, 50), 15, config.COLOR_RED, -1)
            cv2.putText(
                frame, "REC",
                (w - 110, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_RED, 2
            )
        
        # Progress bar
        progress_ratio = min(num_samples / config.SAMPLES_PER_SIGN, 1.0)
        bar_width = int((w - 40) * progress_ratio)
        cv2.rectangle(frame, (20, h - 65), (w - 20, h - 45), config.COLOR_WHITE, 2)
        cv2.rectangle(frame, (22, h - 63), (22 + bar_width, h - 47), config.COLOR_GREEN, -1)
        
        # Controls hint
        controls = "SPACE: Record | N/P: Next/Prev | S: Save | Q: Quit"
        cv2.putText(
            frame, controls,
            (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1
        )
        
        return frame
    
    def _draw_hand_guide(self, frame: np.ndarray) -> np.ndarray:
        """Draw a guide box showing where to position the hand."""
        h, w, _ = frame.shape
        
        # Define guide box (center of frame)
        box_size = min(h, w) // 2
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2 + 40  # Offset for UI
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        color = config.COLOR_GREEN if self.is_recording else config.COLOR_WHITE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        if not self.is_recording:
            cv2.putText(
                frame, "Position hand here",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1
            )
        
        return frame
    
    def run(self):
        """Run the data collection interface."""
        print("\n" + "=" * 50)
        print("VocalHands - Data Collection")
        print("=" * 50)
        print(f"\nCollecting data for {len(config.SIGNS)} signs")
        print(f"Target samples per sign: {config.SAMPLES_PER_SIGN}")
        print("\nControls:")
        print("  SPACE - Start/Stop recording")
        print("  N     - Next sign")
        print("  P     - Previous sign")
        print("  S     - Save progress")
        print("  Q/ESC - Quit and save")
        print("\n" + "=" * 50)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            return
        
        print("\nCamera initialized. Starting collection...")
        
        frame_count = 0
        last_record_time = 0
        record_delay = 0.1  # Minimum delay between samples (seconds)
        
        try:
            while True:
                success, frame = self.cap.read()
                if not success:
                    print("Failed to read frame")
                    continue
                
                # Flip for mirror view
                frame = cv2.flip(frame, 1)
                
                # Detect hands
                frame, results = self.detector.detect_hands(frame)
                
                # Draw landmarks
                frame = self.detector.draw_landmarks(frame, results)
                
                # Record sample if recording and hand detected
                current_time = time.time()
                if self.is_recording and results.multi_hand_landmarks:
                    if current_time - last_record_time >= record_delay:
                        landmarks = self.detector.extract_landmarks(results, normalize=True)
                        if landmarks is not None:
                            current_sign = config.SIGNS[self.current_sign_idx]
                            self.data[current_sign].append(landmarks.tolist())
                            last_record_time = current_time
                            
                            # Auto-stop when target reached
                            if len(self.data[current_sign]) >= config.SAMPLES_PER_SIGN:
                                self.is_recording = False
                                print(f"\n[OK] Completed {config.SAMPLES_PER_SIGN} samples for '{current_sign}'!")
                
                # Draw UI elements
                frame = self._draw_hand_guide(frame)
                frame = self._draw_ui(frame)
                
                # Show warning if no hand detected during recording
                if self.is_recording and not results.multi_hand_landmarks:
                    cv2.putText(
                        frame, "No hand detected!",
                        (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_RED, 2
                    )
                
                # Display frame
                cv2.imshow("VocalHands - Data Collection", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space - toggle recording
                    self.is_recording = not self.is_recording
                    state = "Recording..." if self.is_recording else "Paused"
                    print(f"\n{state} ({config.SIGNS[self.current_sign_idx]})")
                    
                elif key == ord('n'):  # Next sign
                    self.is_recording = False
                    self.current_sign_idx = (self.current_sign_idx + 1) % len(config.SIGNS)
                    current_sign = config.SIGNS[self.current_sign_idx]
                    print(f"\nSwitched to: {current_sign} ({len(self.data[current_sign])} samples)")
                    
                elif key == ord('p'):  # Previous sign
                    self.is_recording = False
                    self.current_sign_idx = (self.current_sign_idx - 1) % len(config.SIGNS)
                    current_sign = config.SIGNS[self.current_sign_idx]
                    print(f"\nSwitched to: {current_sign} ({len(self.data[current_sign])} samples)")
                    
                elif key == ord('s'):  # Save
                    print("\nSaving progress...")
                    self._save_data()
                    
                elif key == ord('q') or key == 27:  # Quit
                    print("\nSaving and exiting...")
                    self._save_data()
                    break
                
                frame_count += 1
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("\nData collection complete!")


def main():
    """Main entry point."""
    collector = DataCollector()
    collector.run()


if __name__ == "__main__":
    main()
