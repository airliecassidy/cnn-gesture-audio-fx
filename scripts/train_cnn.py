"""
Training script for gesture classification CNN.
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gesture_cnn import GestureCNN, get_model_summary
from hagrid.dataset import create_data_loaders, CLASS_NAMES, IDX_TO_CLASS


def train_epoch(model: nn.Module, 
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def evaluate(model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on dataset.
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    
    # Calculate F1 scores
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_per_class = f1_score(all_labels, all_predictions, average=None)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'predictions': all_predictions,
        'labels': all_labels,
    }


def plot_training_curves(history: Dict, output_path: str):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 score
    axes[1, 0].plot(history['val_f1'], label='Validation F1 (macro)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score (Macro)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {output_path}")


def plot_confusion_matrix(y_true: List, y_pred: List, output_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Normalized Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def train_model(args):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(output_dir / 'logs')
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nCreating model...")
    model = GestureCNN(num_classes=6, pretrained=True)
    model = model.to(device)
    
    # Print model summary
    summary = get_model_summary(model)
    print("\nModel Summary:")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'learning_rate': [],
    }
    
    best_val_f1 = 0.0
    best_epoch = 0
    
    print("\nStarting training...")
    print("=" * 50)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1_macro'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1_macro'], epoch)
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.2f}%, "
              f"F1: {val_metrics['f1_macro']:.4f}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"Saved best model (F1: {best_val_f1:.4f})")
        
        # Update learning rate
        scheduler.step()
    
    writer.close()
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Load best model and evaluate on test set
    print("\n" + "=" * 50)
    print("Evaluating best model on test set...")
    print("=" * 50)
    
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"  Macro F1: {test_metrics['f1_macro']:.4f}")
    print(f"\nPer-class F1:")
    for i, f1 in enumerate(test_metrics['f1_per_class']):
        print(f"  {CLASS_NAMES[i]}: {f1:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        test_metrics['labels'],
        test_metrics['predictions'],
        target_names=CLASS_NAMES
    ))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        test_metrics['labels'],
        test_metrics['predictions'],
        output_dir / 'confusion_matrix.png'
    )
    
    # Save results
    results = {
        'best_epoch': best_epoch,
        'best_val_f1': best_val_f1,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_macro': test_metrics['f1_macro'],
        'test_f1_per_class': {
            CLASS_NAMES[i]: f1 
            for i, f1 in enumerate(test_metrics['f1_per_class'])
        },
        'history': history,
        'model_summary': summary,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train gesture classification CNN')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./hagrid_processed',
                       help='Path to processed HaGRID dataset')
    parser.add_argument('--output_dir', type=str, default='./outputs/training',
                       help='Output directory for checkpoints and logs')
    
    # Training
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
