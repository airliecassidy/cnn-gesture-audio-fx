"""
HaGRID dataset preparation and preprocessing.

HaGRID (HAnd Gesture Recognition Image Dataset) contains 18 gesture classes.
We use 6 classes mapped to audio effects:
    - palm -> Gain
    - fist -> Lowpass filter
    - thumb_up -> Highpass filter
    - thumb_down -> Distortion
    - ok -> Delay
    - peace -> Reverb
"""
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp


# Gesture class mapping to audio effects
GESTURE_TO_EFFECT = {
    'palm': 'gain',
    'fist': 'lowpass',
    'thumb_up': 'highpass',
    'thumb_down': 'distortion',
    'ok': 'delay',
    'peace': 'reverb',
}

CLASS_NAMES = list(GESTURE_TO_EFFECT.keys())
EFFECT_NAMES = list(GESTURE_TO_EFFECT.values())

# Class indices for our 6 classes
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def download_hagrid_sample(output_dir: str = './hagrid_data'):
    """
    Create a sample dataset structure for demonstration.
    In production, download from: https://github.com/hukenovs/hagrid
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create directory structure
    for split in ['train', 'test']:
        for gesture in CLASS_NAMES:
            (output_path / split / gesture).mkdir(parents=True, exist_ok=True)
    
    print(f"Created HaGRID dataset structure at {output_dir}")
    print("\nTo use real HaGRID data:")
    print("1. Download from: https://github.com/hukenovs/hagrid")
    print("2. Extract to the created directory structure")
    print("3. Run prepare_hagrid.py")
    
    return output_dir


class HaGRIDPreprocessor:
    """
    Preprocessor for HaGRID dataset using MediaPipe Hands.
    """
    
    def __init__(self, target_size: int = 224, margin: float = 0.2):
        """
        Initialize preprocessor.
        
        Args:
            target_size: Target image size for CNN
            margin: Margin around hand bounding box
        """
        self.target_size = target_size
        self.margin = margin
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def detect_hand(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect hand in image and return landmarks and bounding box.
        
        Args:
            image: BGR image
        
        Returns:
            Tuple of (landmarks, bbox) or None if no hand detected
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Get first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        # Calculate bounding box
        h, w = image.shape[:2]
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min = int(np.min(x_coords) * w)
        x_max = int(np.max(x_coords) * w)
        y_min = int(np.min(y_coords) * h)
        y_max = int(np.max(y_coords) * h)
        
        # Add margin
        margin_x = int((x_max - x_min) * self.margin)
        margin_y = int((y_max - y_min) * self.margin)
        
        x_min = max(0, x_min - margin_x)
        x_max = min(w, x_max + margin_x)
        y_min = max(0, y_min - margin_y)
        y_max = min(h, y_max + margin_y)
        
        bbox = (x_min, y_min, x_max, y_max)
        
        return landmarks, bbox
    
    def extract_roi(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract and preprocess hand ROI.
        
        Args:
            image: BGR image
            bbox: Bounding box (x_min, y_min, x_max, y_max)
        
        Returns:
            Preprocessed ROI image (RGB, normalized)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Extract ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None
        
        # Resize to target size
        roi_resized = cv2.resize(roi, (self.target_size, self.target_size))
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        return roi_rgb
    
    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Process a single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor or None if hand not detected
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        detection = self.detect_hand(image)
        if detection is None:
            return None
        
        _, bbox = detection
        roi = self.extract_roi(image, bbox)
        
        return roi
    
    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


class HaGRIDDataset(Dataset):
    """
    PyTorch Dataset for HaGRID.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 preload: bool = False):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory of processed dataset
            split: 'train', 'val', or 'test'
            transform: Optional transforms
            preload: Whether to preload all images into memory
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.preload = preload
        
        # Load samples
        self.samples = []
        self.labels = []
        
        split_dir = self.data_dir / split
        
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            
            class_idx = CLASS_TO_IDX[class_name]
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append(str(img_path))
                self.labels.append(class_idx)
        
        print(f"Loaded {len(self.samples)} samples for {split}")
        
        # Preload images if requested
        self.preloaded_images = {}
        if preload:
            print("Preloading images...")
            for idx, img_path in enumerate(tqdm(self.samples)):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.preloaded_images[idx] = img
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample.
        
        Returns:
            Tuple of (image_tensor, label)
        """
        if self.preload and idx in self.preloaded_images:
            image = self.preloaded_images[idx]
        else:
            img_path = self.samples[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: to tensor and normalize
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std
        
        label = self.labels[idx]
        
        return image, label


def prepare_hagrid_dataset(input_dir: str, output_dir: str, 
                           min_confidence: float = 0.5):
    """
    Prepare HaGRID dataset by detecting hands and extracting ROIs.
    
    Args:
        input_dir: Input directory with raw HaGRID data
        output_dir: Output directory for processed data
        min_confidence: Minimum detection confidence
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    preprocessor = HaGRIDPreprocessor()
    
    stats = {
        'total_images': 0,
        'detected_hands': 0,
        'failed_detections': 0,
    }
    
    for split in ['train', 'test']:
        print(f"\nProcessing {split} split...")
        
        for class_name in CLASS_NAMES:
            print(f"  Processing class: {class_name}")
            
            input_class_dir = input_path / split / class_name
            output_class_dir = output_path / split / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            if not input_class_dir.exists():
                print(f"    Warning: {input_class_dir} does not exist")
                continue
            
            for img_path in tqdm(list(input_class_dir.glob('*.jpg'))):
                stats['total_images'] += 1
                
                # Process image
                roi = preprocessor.process_image(str(img_path))
                
                if roi is not None:
                    # Save processed ROI
                    output_path_img = output_class_dir / img_path.name
                    cv2.imwrite(str(output_path_img), 
                               cv2.cvtColor(roi, cv2.COLOR_RGB2BGR))
                    stats['detected_hands'] += 1
                else:
                    stats['failed_detections'] += 1
    
    preprocessor.release()
    
    # Save stats
    stats_path = output_path / 'preprocessing_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "=" * 50)
    print("Preprocessing Complete")
    print("=" * 50)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"Detection rate: {stats['detected_hands'] / stats['total_images'] * 100:.1f}%")
    
    # Save class mapping
    mapping_path = output_path / 'class_mapping.json'
    with open(mapping_path, 'w') as f:
        json.dump({
            'gesture_to_effect': GESTURE_TO_EFFECT,
            'class_to_idx': CLASS_TO_IDX,
            'idx_to_class': IDX_TO_CLASS,
        }, f, indent=2)
    
    return stats


def create_data_loaders(data_dir: str, 
                        batch_size: int = 32,
                        num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_dir: Directory with processed dataset
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation/test
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = HaGRIDDataset(data_dir, split='train', transform=train_transform)
    
    # Split train into train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Override transform for val dataset
    val_dataset.dataset.transform = eval_transform
    
    test_dataset = HaGRIDDataset(data_dir, split='test', transform=eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Create sample dataset structure
    download_hagrid_sample()
    
