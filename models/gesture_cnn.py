"""
Lightweight CNN for gesture classification using MobileNetV2 backbone.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict
import time


class GestureCNN(nn.Module):
    """
    Lightweight gesture classification CNN based on MobileNetV2.
    
    Designed for real-time inference with ~3.5M parameters.
    Input: 224x224 RGB images of cropped hand ROI
    Output: 6-class gesture classification
    
    Gesture classes (HaGRID dataset):
        0: palm (open hand) -> Gain
        1: fist (closed hand) -> Lowpass filter
        2: thumb_up -> Highpass filter
        3: thumb_down -> Distortion
        4: ok_sign (circle) -> Delay
        5: peace_sign (two fingers) -> Reverb
    """
    
    NUM_CLASSES = 6
    INPUT_SIZE = 224
    
    # Class mapping to audio effects
    GESTURE_TO_EFFECT = {
        0: 'gain',        # palm - open hand
        1: 'lowpass',     # fist - closed hand
        2: 'highpass',    # thumb_up
        3: 'distortion',  # thumb_down
        4: 'delay',       # ok_sign
        5: 'reverb',      # peace_sign
    }
    
    EFFECT_TO_GESTURE = {v: k for k, v in GESTURE_TO_EFFECT.items()}
    
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, 
                 dropout: float = 0.3):
        """
        Initialize the gesture classification model.
        
        Args:
            num_classes: Number of gesture classes (default: 6)
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability for regularization
        """
        super(GestureCNN, self).__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Get the number of features from the last layer
        in_features = self.backbone.last_channel
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_classes)
        )
        
        # Initialize new layers with Kaiming initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize new classifier layers."""
        for m in self.backbone.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class labels and confidences.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224) or (3, 224, 224)
        
        Returns:
            Tuple of (predicted_class_indices, confidence_scores)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
        
        return predictions, confidences
    
    def get_effect_name(self, class_idx: int) -> str:
        """Get effect name from class index."""
        return self.GESTURE_TO_EFFECT.get(class_idx, 'unknown')
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict:
    """
    Get model summary including parameter count and FLOPs estimate.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    
    Returns:
        Dictionary with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Measure inference latency
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Measure
    num_runs = 100
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    
    avg_latency_ms = (end_time - start_time) / num_runs * 1000
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'avg_inference_latency_ms': avg_latency_ms,
        'estimated_fps': 1000 / avg_latency_ms if avg_latency_ms > 0 else float('inf')
    }


def create_model(num_classes: int = 6, pretrained: bool = True) -> GestureCNN:
    """Factory function to create model."""
    return GestureCNN(num_classes=num_classes, pretrained=pretrained)


if __name__ == '__main__':
    # Test model creation and summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    model.eval()
    
    summary = get_model_summary(model)
    print("=" * 50)
    print("Model Summary")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value:,.2f}" if isinstance(value, float) else f"{key}: {value:,}")
    print("=" * 50)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    predictions, confidences = model.predict(dummy_input)
    print(f"Predicted class: {predictions.item()}")
    print(f"Confidence: {confidences.item():.4f}")
