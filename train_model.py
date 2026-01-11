"""
VocalHands - Model Training Script
===================================
Train a KNN classifier on collected hand landmark data.

Usage:
    python train_model.py

The script will:
1. Load collected landmark data from the dataset folder
2. Preprocess and prepare features
3. Train a KNN classifier
4. Evaluate on test set
5. Save the trained model
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)

import config


class SignLanguageTrainer:
    """
    Trainer for the sign language KNN model.
    
    Handles data loading, preprocessing, training, evaluation,
    and model persistence.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the collected dataset from disk.
        
        Returns:
            Tuple of (features, labels)
        """
        print("\n" + "=" * 50)
        print("Loading Dataset")
        print("=" * 50)
        
        all_features = []
        all_labels = []
        
        for sign in config.SIGNS:
            sign_dir = os.path.join(config.DATA_DIR, sign)
            data_file = os.path.join(sign_dir, "landmarks.npy")
            
            if os.path.exists(data_file):
                data = np.load(data_file)
                num_samples = len(data)
                
                all_features.extend(data.tolist())
                all_labels.extend([sign] * num_samples)
                
                print(f"  {sign}: {num_samples} samples")
            else:
                print(f"  {sign}: No data found (skipping)")
        
        if len(all_features) == 0:
            raise ValueError(
                "No data found! Please run collect_data.py first to gather training samples."
            )
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n[OK] Loaded {len(X)} total samples")
        print(f"[OK] Number of classes: {len(np.unique(y))}")
        print(f"[OK] Features per sample: {X.shape[1]}")
        
        return X, y
    
    def prepare_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        print("\n" + "=" * 50)
        print("Preparing Data")
        print("=" * 50)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=random_state,
            stratify=y_encoded
        )
        
        # Scale features (optional but can help KNN)
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"[OK] Training samples: {len(self.X_train)}")
        print(f"[OK] Test samples: {len(self.X_test)}")
        print(f"[OK] Classes: {list(self.label_encoder.classes_)}")
    
    def find_best_k(self, k_range: range = range(1, 31)) -> int:
        """
        Find the optimal k value using cross-validation.
        
        Args:
            k_range: Range of k values to try
            
        Returns:
            Best k value
        """
        print("\n" + "=" * 50)
        print("Finding Optimal K")
        print("=" * 50)
        
        best_k = 1
        best_score = 0
        scores = []
        
        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=config.KNN_WEIGHTS,
                algorithm=config.KNN_ALGORITHM,
                metric=config.KNN_METRIC
            )
            
            # 5-fold cross-validation
            cv_scores = cross_val_score(knn, self.X_train, self.y_train, cv=5)
            mean_score = cv_scores.mean()
            scores.append(mean_score)
            
            if mean_score > best_score:
                best_score = mean_score
                best_k = k
        
        print(f"[OK] Best K: {best_k}")
        print(f"[OK] Best CV Score: {best_score:.4f}")
        
        # Print top 5 k values
        print("\nTop 5 K values:")
        sorted_k = sorted(zip(k_range, scores), key=lambda x: x[1], reverse=True)[:5]
        for k, score in sorted_k:
            print(f"  K={k}: {score:.4f}")
        
        return best_k
    
    def train(self, n_neighbors: Optional[int] = None) -> KNeighborsClassifier:
        """
        Train the KNN classifier.
        
        Args:
            n_neighbors: Number of neighbors (if None, uses config default)
            
        Returns:
            Trained KNN classifier
        """
        print("\n" + "=" * 50)
        print("Training Model")
        print("=" * 50)
        
        if n_neighbors is None:
            n_neighbors = config.KNN_N_NEIGHBORS
        
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=config.KNN_WEIGHTS,
            algorithm=config.KNN_ALGORITHM,
            metric=config.KNN_METRIC
        )
        
        print(f"Training KNN with K={n_neighbors}...")
        self.model.fit(self.X_train, self.y_train)
        
        # Training accuracy
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"[OK] Training accuracy: {train_score:.4f}")
        
        return self.model
    
    def evaluate(self) -> dict:
        """
        Evaluate the trained model on the test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "=" * 50)
        print("Model Evaluation")
        print("=" * 50)
        
        # Predictions
        y_pred = self.model.predict(self.X_test)
        
        # Accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n[OK] Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        
        # Get class names for the classes present in test set
        classes_in_test = np.unique(np.concatenate([self.y_test, y_pred]))
        target_names = [self.label_encoder.classes_[i] for i in classes_in_test]
        
        # Classification report
        print("\nClassification Report:")
        print("-" * 50)
        report = classification_report(
            self.y_test, 
            y_pred, 
            target_names=target_names,
            labels=classes_in_test
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred, labels=classes_in_test)
        print("\nConfusion Matrix:")
        print("-" * 50)
        print(cm)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
    
    def save_model(self) -> None:
        """Save the trained model and related objects."""
        print("\n" + "=" * 50)
        print("Saving Model")
        print("=" * 50)
        
        # Create models directory if needed
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        # Save KNN model
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "config": {
                "n_neighbors": self.model.n_neighbors,
                "weights": config.KNN_WEIGHTS,
                "algorithm": config.KNN_ALGORITHM,
                "metric": config.KNN_METRIC,
                "features_per_sample": config.FEATURES_PER_HAND,
            },
            "trained_at": datetime.now().isoformat(),
            "classes": list(self.label_encoder.classes_)
        }
        
        with open(config.KNN_MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[OK] Model saved to: {config.KNN_MODEL_PATH}")
        
        # Save model info as JSON for reference
        info_path = os.path.join(config.MODEL_DIR, "model_info.json")
        info = {
            "model_type": "KNeighborsClassifier",
            "n_neighbors": self.model.n_neighbors,
            "weights": config.KNN_WEIGHTS,
            "algorithm": config.KNN_ALGORITHM,
            "metric": config.KNN_METRIC,
            "num_classes": len(self.label_encoder.classes_),
            "classes": list(self.label_encoder.classes_),
            "features_per_sample": config.FEATURES_PER_HAND,
            "trained_at": datetime.now().isoformat()
        }
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[OK] Model info saved to: {info_path}")


def main():
    """Main training pipeline."""
    print("\n" + "=" * 60)
    print("   VocalHands - Sign Language Model Training")
    print("=" * 60)
    
    trainer = SignLanguageTrainer()
    
    try:
        # Load dataset
        X, y = trainer.load_dataset()
        
        # Prepare data
        trainer.prepare_data(X, y)
        
        # Find best k
        best_k = trainer.find_best_k()
        
        # Train model
        trainer.train(n_neighbors=best_k)
        
        # Evaluate
        metrics = trainer.evaluate()
        
        # Save model
        trainer.save_model()
        
        print("\n" + "=" * 60)
        print("   Training Complete!")
        print("=" * 60)
        print(f"\n[OK] Final Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"[OK] Model saved to: {config.KNN_MODEL_PATH}")
        print("\nYou can now run 'python detect_signs.py' to test the model!")
        
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease collect some data first:")
        print("  python collect_data.py")


if __name__ == "__main__":
    main()
