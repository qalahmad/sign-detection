"""
VocalHands - Sample Dataset Generator
======================================
Creates a synthetic sample dataset structure for testing and development.

This script generates placeholder data to test the training pipeline.
For production use, collect real data using collect_data.py

Usage:
    python create_sample_dataset.py
"""

import os
import json
import numpy as np
from datetime import datetime

import config


def generate_sample_data(num_samples: int = 50) -> np.ndarray:
    """
    Generate synthetic hand landmark data for testing.
    
    This creates random normalized landmarks that simulate
    the output of MediaPipe hand detection. The data is NOT
    suitable for real training but allows testing the pipeline.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Array of shape (num_samples, 63)
    """
    # Generate random normalized coordinates
    # In real data, these would be actual hand landmarks
    data = np.random.randn(num_samples, config.FEATURES_PER_HAND) * 0.3
    
    # Add some structure to make it slightly more realistic
    # Center around origin (like normalized landmarks)
    data = data - data.mean(axis=1, keepdims=True)
    
    return data


def create_sample_dataset(signs_subset: list = None, samples_per_sign: int = 50):
    """
    Create a sample dataset structure with synthetic data.
    
    Args:
        signs_subset: List of signs to generate (default: first 10 alphabet letters)
        samples_per_sign: Number of samples per sign
    """
    print("\n" + "=" * 50)
    print("Creating Sample Dataset")
    print("=" * 50)
    
    if signs_subset is None:
        # Default to first 10 letters for quick testing
        signs_subset = config.SIGNS[:10]
    
    print(f"\nGenerating data for {len(signs_subset)} signs:")
    print(f"  Signs: {signs_subset}")
    print(f"  Samples per sign: {samples_per_sign}")
    
    total_samples = 0
    
    for sign in signs_subset:
        sign_dir = os.path.join(config.DATA_DIR, sign)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Generate sample data
        data = generate_sample_data(samples_per_sign)
        
        # Add sign-specific variation to make classes somewhat distinguishable
        # This is just for testing - real data would have actual hand shapes
        sign_idx = signs_subset.index(sign)
        data += sign_idx * 0.1  # Offset each class slightly
        
        # Save landmarks
        data_file = os.path.join(sign_dir, "landmarks.npy")
        np.save(data_file, data)
        
        # Save metadata
        metadata = {
            "sign": sign,
            "num_samples": samples_per_sign,
            "features_per_sample": config.FEATURES_PER_HAND,
            "is_synthetic": True,  # Mark as synthetic data
            "last_updated": datetime.now().isoformat()
        }
        metadata_file = os.path.join(sign_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        total_samples += samples_per_sign
        print(f"  ✓ {sign}: {samples_per_sign} samples")
    
    # Save global dataset info
    dataset_info = {
        "signs": signs_subset,
        "total_samples": total_samples,
        "samples_per_sign": {sign: samples_per_sign for sign in signs_subset},
        "features_per_sample": config.FEATURES_PER_HAND,
        "is_synthetic": True,
        "note": "This is synthetic data for testing. Collect real data using collect_data.py",
        "last_updated": datetime.now().isoformat()
    }
    info_file = os.path.join(config.DATA_DIR, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\n✓ Sample dataset created!")
    print(f"  Location: {config.DATA_DIR}")
    print(f"  Total samples: {total_samples}")
    print(f"\n⚠️  Note: This is SYNTHETIC data for testing only.")
    print("  For real sign detection, collect data using:")
    print("    python collect_data.py")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("   VocalHands - Sample Dataset Generator")
    print("=" * 60)
    
    print("\nThis will create a sample dataset for testing the pipeline.")
    print("The data is synthetic and not suitable for real sign detection.")
    
    # Create sample dataset with first 10 letters
    create_sample_dataset(
        signs_subset=["A", "B", "C", "D", "E", "F", "G", "H", "I", "L"],
        samples_per_sign=100
    )
    
    print("\n" + "=" * 60)
    print("   Next Steps")
    print("=" * 60)
    print("\n1. Train the model with synthetic data (for testing):")
    print("   python train_model.py")
    print("\n2. Or collect real data:")
    print("   python collect_data.py")
    print("\n3. Then train with real data:")
    print("   python train_model.py")
    print("\n4. Run detection:")
    print("   python detect_signs.py")


if __name__ == "__main__":
    main()
