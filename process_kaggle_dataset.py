"""
VocalHands - Kaggle ASL Dataset Processor
==========================================
Process the Kaggle ASL Alphabet dataset to extract hand landmarks.

Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

Instructions:
1. Download the dataset from Kaggle (you'll need a Kaggle account)
2. Extract the zip file
3. Run this script with the path to the extracted folder

Usage:
    python process_kaggle_dataset.py --dataset_path "path/to/asl_alphabet_train"
    
    Or place the dataset in the default location:
    python process_kaggle_dataset.py
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from datetime import datetime
from tqdm import tqdm
from typing import Optional, List, Tuple

import config
from utils.hand_detector import HandDetector


# Default paths to look for the dataset
DEFAULT_DATASET_PATHS = [
    os.path.join(config.BASE_DIR, "data", "asl_alphabet_train", "asl_alphabet_train"),  # Your location (nested)!
    os.path.join(config.BASE_DIR, "data", "asl_alphabet_train"),
    os.path.join(config.BASE_DIR, "asl_alphabet_train"),
    os.path.join(config.BASE_DIR, "asl-alphabet", "asl_alphabet_train"),
    os.path.join(config.BASE_DIR, "archive", "asl_alphabet_train"),
    os.path.expanduser("~/Downloads/asl_alphabet_train"),
    os.path.expanduser("~/Downloads/archive/asl_alphabet_train"),
]

# Mapping from Kaggle dataset folder names to our sign names
# The Kaggle dataset has A-Z plus 'del', 'nothing', 'space'
KAGGLE_TO_SIGN_MAP = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E',
    'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J',
    'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O',
    'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T',
    'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y',
    'Z': 'Z',
    # Optional mappings for extra classes
    'space': 'SPACE',
    'del': 'DELETE',
    'nothing': 'NOTHING',
}


class KaggleDatasetProcessor:
    """
    Processes the Kaggle ASL Alphabet dataset to extract hand landmarks.
    """
    
    def __init__(self, dataset_path: str, max_samples_per_sign: int = 500):
        """
        Initialize the processor.
        
        Args:
            dataset_path: Path to the asl_alphabet_train folder
            max_samples_per_sign: Maximum samples to process per sign
        """
        self.dataset_path = dataset_path
        self.max_samples = max_samples_per_sign
        self.detector = HandDetector(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'signs_processed': []
        }
    
    def find_dataset(self) -> Optional[str]:
        """
        Find the dataset path.
        
        Returns:
            Path to dataset or None if not found
        """
        # Check provided path first
        if os.path.exists(self.dataset_path):
            return self.dataset_path
        
        # Check default locations
        for path in DEFAULT_DATASET_PATHS:
            if os.path.exists(path):
                print(f"Found dataset at: {path}")
                return path
        
        return None
    
    def get_sign_folders(self) -> List[Tuple[str, str]]:
        """
        Get list of sign folders in the dataset.
        
        Returns:
            List of (folder_name, sign_name) tuples
        """
        folders = []
        
        for item in os.listdir(self.dataset_path):
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path):
                # Map to our sign name
                sign_name = KAGGLE_TO_SIGN_MAP.get(item, item.upper())
                folders.append((item, sign_name))
        
        return sorted(folders)
    
    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process a single image and extract landmarks.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Normalized landmarks array or None if extraction failed
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Detect hands
        _, results = self.detector.detect_hands(image)
        
        # Extract landmarks
        landmarks = self.detector.extract_landmarks(results, normalize=True)
        
        return landmarks
    
    def process_sign_folder(self, folder_name: str, sign_name: str) -> Tuple[np.ndarray, int]:
        """
        Process all images in a sign folder.
        
        Args:
            folder_name: Name of the folder in the dataset
            sign_name: Name of the sign for saving
            
        Returns:
            Tuple of (landmarks_array, num_successful)
        """
        folder_path = os.path.join(self.dataset_path, folder_name)
        
        # Get all image files
        image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        # Limit samples
        if len(image_files) > self.max_samples:
            # Take evenly distributed samples
            step = len(image_files) // self.max_samples
            image_files = image_files[::step][:self.max_samples]
        
        landmarks_list = []
        
        for image_file in tqdm(image_files, desc=f"  {sign_name}", leave=False):
            image_path = os.path.join(folder_path, image_file)
            
            landmarks = self.process_image(image_path)
            
            if landmarks is not None:
                landmarks_list.append(landmarks)
                self.stats['successful_extractions'] += 1
            else:
                self.stats['failed_extractions'] += 1
            
            self.stats['total_images'] += 1
        
        if landmarks_list:
            return np.array(landmarks_list), len(landmarks_list)
        return np.array([]), 0
    
    def save_sign_data(self, sign_name: str, landmarks: np.ndarray) -> None:
        """
        Save processed landmarks for a sign.
        
        Args:
            sign_name: Name of the sign
            landmarks: Landmarks array
        """
        sign_dir = os.path.join(config.DATA_DIR, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Save landmarks
        data_file = os.path.join(sign_dir, "landmarks.npy")
        np.save(data_file, landmarks)
        
        # Save metadata
        metadata = {
            "sign": sign_name,
            "num_samples": len(landmarks),
            "features_per_sample": config.FEATURES_PER_HAND,
            "source": "kaggle_asl_alphabet",
            "last_updated": datetime.now().isoformat()
        }
        metadata_file = os.path.join(sign_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_dataset(self, signs_to_process: Optional[List[str]] = None) -> None:
        """
        Process the entire dataset.
        
        Args:
            signs_to_process: Optional list of specific signs to process
        """
        print("\n" + "=" * 60)
        print("   Kaggle ASL Alphabet Dataset Processor")
        print("=" * 60)
        
        # Find dataset
        dataset_path = self.find_dataset()
        if dataset_path is None:
            print("\n[ERROR] Dataset not found!")
            print("\nPlease download the dataset from:")
            print("   https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data")
            print("\nThen either:")
            print("   1. Extract to:", config.BASE_DIR)
            print("   2. Or run with: python process_kaggle_dataset.py --dataset_path \"your/path\"")
            return
        
        self.dataset_path = dataset_path
        print(f"\n[OK] Dataset found at: {dataset_path}")
        
        # Get sign folders
        sign_folders = self.get_sign_folders()
        
        # Filter if specific signs requested
        if signs_to_process:
            sign_folders = [
                (f, s) for f, s in sign_folders 
                if s in signs_to_process or f in signs_to_process
            ]
        
        print(f"\n[OK] Found {len(sign_folders)} sign categories")
        print(f"[OK] Max samples per sign: {self.max_samples}")
        
        print("\n" + "-" * 60)
        print("Processing signs...")
        print("-" * 60)
        
        processed_signs = {}
        
        for folder_name, sign_name in tqdm(sign_folders, desc="Overall progress"):
            landmarks, num_samples = self.process_sign_folder(folder_name, sign_name)
            
            if num_samples > 0:
                self.save_sign_data(sign_name, landmarks)
                processed_signs[sign_name] = num_samples
                self.stats['signs_processed'].append(sign_name)
                print(f"  [OK] {sign_name}: {num_samples} samples extracted")
            else:
                print(f"  [WARN] {sign_name}: No landmarks could be extracted")
        
        # Save dataset info
        self._save_dataset_info(processed_signs)
        
        # Print summary
        self._print_summary(processed_signs)
    
    def _save_dataset_info(self, processed_signs: dict) -> None:
        """Save global dataset information."""
        dataset_info = {
            "signs": list(processed_signs.keys()),
            "total_samples": sum(processed_signs.values()),
            "samples_per_sign": processed_signs,
            "features_per_sample": config.FEATURES_PER_HAND,
            "source": "kaggle_asl_alphabet",
            "source_url": "https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data",
            "processing_stats": {
                "total_images_processed": self.stats['total_images'],
                "successful_extractions": self.stats['successful_extractions'],
                "failed_extractions": self.stats['failed_extractions'],
                "success_rate": f"{(self.stats['successful_extractions'] / max(self.stats['total_images'], 1)) * 100:.1f}%"
            },
            "last_updated": datetime.now().isoformat()
        }
        
        info_file = os.path.join(config.DATA_DIR, "dataset_info.json")
        with open(info_file, 'w') as f:
            json.dump(dataset_info, f, indent=2)
    
    def _print_summary(self, processed_signs: dict) -> None:
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("   Processing Complete!")
        print("=" * 60)
        
        total_samples = sum(processed_signs.values())
        success_rate = (self.stats['successful_extractions'] / max(self.stats['total_images'], 1)) * 100
        
        print(f"\n[OK] Signs processed: {len(processed_signs)}")
        print(f"[OK] Total samples: {total_samples}")
        print(f"[OK] Success rate: {success_rate:.1f}%")
        print(f"[OK] Data saved to: {config.DATA_DIR}")
        
        print("\nSamples per sign:")
        for sign, count in sorted(processed_signs.items()):
            bar = "█" * min(count // 20, 25)
            print(f"  {sign:>10}: {count:>4} {bar}")
        
        print("\n" + "-" * 60)
        print("Next steps:")
        print("  1. Train the model: python train_model.py")
        print("  2. Run detection:   python detect_signs.py")
        print("-" * 60)
    
    def close(self):
        """Clean up resources."""
        self.detector.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process Kaggle ASL Alphabet dataset for VocalHands"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to the asl_alphabet_train folder"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=500,
        help="Maximum samples per sign (default: 500)"
    )
    parser.add_argument(
        "--signs",
        type=str,
        nargs="+",
        default=None,
        help="Specific signs to process (e.g., A B C)"
    )
    
    args = parser.parse_args()
    
    # Use provided path or empty string to trigger auto-detection
    dataset_path = args.dataset_path if args.dataset_path else ""
    
    processor = KaggleDatasetProcessor(
        dataset_path=dataset_path,
        max_samples_per_sign=args.max_samples
    )
    
    try:
        processor.process_dataset(signs_to_process=args.signs)
    finally:
        processor.close()


if __name__ == "__main__":
    main()
