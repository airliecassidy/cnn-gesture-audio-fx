"""
Evaluation script for trained gesture classification model.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from tqdm import tqdm
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gesture_cnn import GestureCNN, get_model_summary
from hagrid.dataset import HaGRIDDataset, CLASS_NAMES, IDX_TO_CLASS
from vision.hand_tracker import HandTracker


def evaluate_model(model: nn.Module,
                   data_loader: DataLoader,
                   device: torch.device) -> Dict:
    """
    Comprehensive model evaluation.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, dim=1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
    }


def plot_confusion_matrix(y_true: List, y_pred: List, output_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(CLASS_NAMES)),
           yticks=np.arange(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Normalized Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = 0.5
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def plot_confidence_distribution(confidences: List, predictions: List, 
                                  labels: List, output_path: str):
    """Plot confidence distribution for correct and incorrect predictions."""
    correct_confidences = [c for c, p, l in zip(confidences, predictions, labels) if p == l]
    incorrect_confidences = [c for c, p, l in zip(confidences, predictions, labels) if p != l]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, 1, 21)
    ax.hist(correct_confidences, bins=bins, alpha=0.7, label='Correct', color='green')
    ax.hist(incorrect_confidences, bins=bins, alpha=0.7, label='Incorrect', color='red')
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confidence distribution saved to {output_path}")


def plot_per_class_metrics(f1_per_class: np.ndarray, output_path: str):
    """Plot per-class F1 scores."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, f1_per_class, color='steelblue', edgecolor='black')
    
    ax.set_xlabel('Gesture Class')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_per_class):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{f1:.3f}',
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class metrics saved to {output_path}")


def visualize_roi_crops(data_dir: str, output_path: str, num_samples: int = 6):
    """Visualize example ROI crops from dataset."""
    dataset = HaGRIDDataset(data_dir, split='test')
    
    # Sample random images
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Convert tensor to numpy for display
        image_np = image.permute(1, 2, 0).numpy()
        
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)
        
        axes[i].imshow(image_np)
        axes[i].set_title(f"{CLASS_NAMES[label]}")
        axes[i].axis('off')
    
    plt.suptitle('Example ROI Crops from HaGRID Dataset', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROI crops visualization saved to {output_path}")


def measure_inference_latency(model: nn.Module, device: torch.device, 
                               num_runs: int = 1000):
    """Measure inference latency."""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in tqdm(range(num_runs), desc="Measuring latency"):
            start = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start.record()
                _ = model(dummy_input)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                import time
                start_time = time.perf_counter()
                _ = model(dummy_input)
                elapsed = (time.perf_counter() - start_time) * 1000
            
            times.append(elapsed)
    
    times = np.array(times)
    
    results = {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
        'estimated_fps': float(1000 / np.mean(times)),
    }
    
    print("\nInference Latency Measurements:")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    
    return results


def evaluate_on_webcam(model_path: str, device: torch.device):
    """Evaluate model on live webcam feed."""
    from vision.hand_tracker import HandTracker
    from vision.inference import GestureInference
    
    # Load model
    inference = GestureInference(model_path, device=str(device))
    
    # Initialize hand tracker
    tracker = HandTracker()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("\nWebcam evaluation started...")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Stats
    frame_count = 0
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        right_hand, left_hand = tracker.process_frame(frame)
        
        # Run inference on right hand
        if right_hand:
            # Extract ROI
            roi = tracker.extract_roi(frame, right_hand)
            
            # Run inference
            start_time = cv2.getTickCount()
            pred_class, confidence, _ = inference.predict(roi)
            end_time = cv2.getTickCount()
            
            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000
            inference_times.append(inference_time)
            
            # Draw results
            frame = tracker.draw_landmarks(frame, right_hand, (0, 255, 0))
            
            # Display prediction
            if pred_class >= 0:
                gesture_name = CLASS_NAMES[pred_class]
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Low confidence", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display inference time
            avg_time = np.mean(inference_times[-30:]) if inference_times else 0
            cv2.putText(frame, f"Inference: {avg_time:.1f}ms", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw left hand
        if left_hand:
            frame = tracker.draw_landmarks(frame, left_hand, (255, 0, 0))
            pinch_dist = left_hand.get_pinch_distance()
            cv2.putText(frame, f"Pinch: {pinch_dist:.3f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        cv2.imshow('Gesture Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('screenshot.png', frame)
            print("Screenshot saved")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    
    print(f"\nProcessed {frame_count} frames")
    if inference_times:
        print(f"Average inference time: {np.mean(inference_times):.2f}ms")
        print(f"Estimated FPS: {1000/np.mean(inference_times):.1f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate gesture classification model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='./hagrid_processed',
                       help='Path to processed dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                       help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--webcam', action='store_true',
                       help='Evaluate on webcam feed')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("\nLoading model...")
    model = GestureCNN(num_classes=6)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # Print model summary
    summary = get_model_summary(model)
    print("\nModel Summary:")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")
    
    if args.webcam:
        # Evaluate on webcam
        evaluate_on_webcam(args.model_path, device)
    else:
        # Load test dataset
        print("\nLoading test dataset...")
        test_dataset = HaGRIDDataset(args.data_dir, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=False, num_workers=4)
        
        # Evaluate
        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, test_loader, device)
        
        # Print results
        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"\nPer-class F1:")
        for i, f1 in enumerate(metrics['f1_per_class']):
            print(f"  {CLASS_NAMES[i]}: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            metrics['labels'],
            metrics['predictions'],
            target_names=CLASS_NAMES
        ))
        
        # Generate plots
        print("\nGenerating plots...")
        plot_confusion_matrix(
            metrics['labels'],
            metrics['predictions'],
            output_dir / 'confusion_matrix.png'
        )
        
        plot_confidence_distribution(
            metrics['confidences'],
            metrics['predictions'],
            metrics['labels'],
            output_dir / 'confidence_distribution.png'
        )
        
        plot_per_class_metrics(
            metrics['f1_per_class'],
            output_dir / 'per_class_f1.png'
        )
        
        visualize_roi_crops(
            args.data_dir,
            output_dir / 'example_rois.png'
        )
        
        # Measure latency
        print("\nMeasuring inference latency...")
        latency_results = measure_inference_latency(model, device)
        
        # Save results
        results = {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_per_class': {
                CLASS_NAMES[i]: float(f1)
                for i, f1 in enumerate(metrics['f1_per_class'])
            },
            'latency': latency_results,
            'model_summary': summary,
        }
        
        with open(output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
