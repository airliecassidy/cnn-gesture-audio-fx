"""
Real-time gesture inference using trained CNN.
"""
import torch
import numpy as np
from typing import Tuple, Optional
import time

from ..models.gesture_cnn import GestureCNN


class GestureInference:
    """
    Real-time gesture classification inference.
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 device: str = 'auto',
                 confidence_threshold: float = 0.6):
        """
        Initialize gesture inference.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda', 'cpu', or 'auto'
            confidence_threshold: Minimum confidence to accept prediction
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load model
        self.model = GestureCNN().to(self.device)
        
        if model_path:
            self.load_checkpoint(model_path)
        
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        
        # Preallocate input tensor for efficiency
        self.input_tensor = torch.zeros(1, 3, 224, 224, device=self.device)
        
        # Timing stats
        self.inference_times = []
        self.max_timing_samples = 100
    
    def load_checkpoint(self, model_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Loaded model from {model_path}")
    
    def preprocess(self, roi: np.ndarray) -> torch.Tensor:
        """
        Preprocess ROI for inference.
        
        Args:
            roi: ROI image (224, 224, 3) normalized
        
        Returns:
            Preprocessed tensor
        """
        # Convert HWC to CHW
        roi_chw = np.transpose(roi, (2, 0, 1))
        
        # Convert to tensor
        tensor = torch.from_numpy(roi_chw).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def predict(self, roi: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Run inference on ROI.
        
        Args:
            roi: Preprocessed ROI image
        
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        # Preprocess
        input_tensor = self.preprocess(roi)
        
        # Inference
        start_time = time.time()
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Update timing stats
        self.inference_times.append(inference_time)
        if len(self.inference_times) > self.max_timing_samples:
            self.inference_times.pop(0)
        
        # Get prediction
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        predicted_class = predicted_class.item()
        confidence = confidence.item()
        probs = probabilities.cpu().numpy()[0]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            # Return -1 to indicate low confidence
            return -1, confidence, probs
        
        return predicted_class, confidence, probs
    
    def get_average_latency(self) -> float:
        """Get average inference latency in ms."""
        if len(self.inference_times) == 0:
            return 0.0
        return np.mean(self.inference_times)
    
    def get_fps(self) -> float:
        """Get estimated FPS based on inference time."""
        avg_latency = self.get_average_latency()
        if avg_latency == 0:
            return 0.0
        return 1000.0 / avg_latency


def benchmark_inference(model_path: str, num_runs: int = 1000):
    """
    Benchmark inference speed.
    
    Args:
        model_path: Path to model checkpoint
        num_runs: Number of inference runs
    """
    inference = GestureInference(model_path)
    
    # Create dummy input
    dummy_roi = np.random.randn(224, 224, 3).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = inference.predict(dummy_roi)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = inference.predict(dummy_roi)
        times.append((time.time() - start) * 1000)
    
    times = np.array(times)
    
    print(f"\nInference Benchmark ({num_runs} runs)")
    print("=" * 40)
    print(f"Mean: {np.mean(times):.2f} ms")
    print(f"Std: {np.std(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")
    print(f"P95: {np.percentile(times, 95):.2f} ms")
    print(f"P99: {np.percentile(times, 99):.2f} ms")
    print(f"Estimated FPS: {1000/np.mean(times):.1f}")


if __name__ == '__main__':
    # Test inference
    inference = GestureInference()
    
    # Create dummy ROI
    dummy_roi = np.random.randn(224, 224, 3).astype(np.float32)
    
    # Run prediction
    pred_class, confidence, probs = inference.predict(dummy_roi)
    
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {probs}")
    
    # Benchmark
    benchmark_inference(None, num_runs=100)
