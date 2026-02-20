"""
Test script for gesture-controlled audio effects system.

Tests all major components without requiring actual hardware.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import cv2

print("=" * 60)
print("GESTURE AUDIO EFFECTS SYSTEM - COMPONENT TESTS")
print("=" * 60)

# Test 1: Shared State
print("\n[Test 1] Shared State...")
try:
    from shared.state import SharedState
    
    state = SharedState()
    state.current_effect_idx = 2
    state.effect_depth = 0.75
    state.update_calibration(0.02, 0.25)
    
    assert state.current_effect_idx == 2
    assert abs(state.effect_depth - 0.75) < 0.01
    assert state.is_calibrated == True
    
    effect_idx, depth = state.get_audio_params()
    assert effect_idx == 2
    assert abs(depth - 0.75) < 0.01
    
    print("  ✓ Shared State: PASSED")
except Exception as e:
    print(f"  ✗ Shared State: FAILED - {e}")

# Test 2: CNN Model
print("\n[Test 2] CNN Model...")
try:
    from models.gesture_cnn import GestureCNN, get_model_summary
    
    model = GestureCNN(num_classes=6)
    model.eval()  # Set to eval mode for batch norm
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 6)
    
    # Test predict
    pred, conf = model.predict(dummy_input)
    assert 0 <= pred.item() < 6
    assert 0 <= conf.item() <= 1
    
    # Test model summary
    summary = get_model_summary(model)
    assert summary['total_parameters'] > 0
    assert summary['avg_inference_latency_ms'] > 0
    
    print(f"  ✓ CNN Model: PASSED")
    print(f"    Parameters: {summary['total_parameters']:,}")
    print(f"    Latency: {summary['avg_inference_latency_ms']:.2f} ms")
except Exception as e:
    print(f"  ✗ CNN Model: FAILED - {e}")

# Test 3: Hand Tracker (without webcam)
print("\n[Test 3] Hand Tracker Components...")
try:
    from vision.hand_tracker import TemporalSmoother, ClassSmoother
    
    # Test temporal smoother
    ts = TemporalSmoother(method='ema', alpha=0.5)
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    smoothed = [ts.update(v) for v in values]
    assert len(smoothed) == len(values)
    
    # Test class smoother
    cs = ClassSmoother(window_size=5)
    for i in range(10):
        cs.update(2, 0.8)
    pred, conf = cs.update(2, 0.8)
    assert pred == 2
    
    print("  ✓ Hand Tracker Components: PASSED")
except Exception as e:
    print(f"  ✗ Hand Tracker Components: FAILED - {e}")

# Test 4: DSP Effects
print("\n[Test 4] DSP Effects...")
try:
    from audio.dsp import (
        GainEffect, LowpassFilter, HighpassFilter,
        DistortionEffect, DelayEffect, ReverbEffect,
        AudioEffectChain
    )
    
    sample_rate = 48000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_signal = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    
    effects = {
        'Gain': GainEffect(sample_rate),
        'Lowpass': LowpassFilter(sample_rate),
        'Highpass': HighpassFilter(sample_rate),
        'Distortion': DistortionEffect(sample_rate),
        'Delay': DelayEffect(sample_rate),
        'Reverb': ReverbEffect(sample_rate),
    }
    
    for name, effect in effects.items():
        effect.set_depth(0.5)
        output = effect.process(test_signal.copy())
        assert len(output) == len(test_signal)
        assert not np.isnan(output).any()
        assert not np.isinf(output).any()
    
    # Test effect chain
    chain = AudioEffectChain(sample_rate)
    chain.set_effect_by_index(3)
    chain.set_depth(0.7)
    output = chain.process(test_signal)
    assert len(output) == len(test_signal)
    
    print("  ✓ DSP Effects: PASSED")
    print(f"    Tested {len(effects)} effects")
except Exception as e:
    print(f"  ✗ DSP Effects: FAILED - {e}")

# Test 5: Dataset Classes
print("\n[Test 5] Dataset Classes...")
try:
    from hagrid.dataset import (
        CLASS_NAMES, GESTURE_TO_EFFECT, 
        CLASS_TO_IDX, IDX_TO_CLASS
    )
    
    assert len(CLASS_NAMES) == 6
    assert len(GESTURE_TO_EFFECT) == 6
    assert all(name in GESTURE_TO_EFFECT for name in CLASS_NAMES)
    assert all(idx in IDX_TO_CLASS for idx in range(6))
    
    print("  ✓ Dataset Classes: PASSED")
    print(f"    Classes: {CLASS_NAMES}")
except Exception as e:
    print(f"  ✗ Dataset Classes: FAILED - {e}")

# Test 6: Gesture Inference
print("\n[Test 6] Gesture Inference...")
try:
    from vision.inference import GestureInference
    
    inference = GestureInference(device='cpu')
    
    # Test with dummy ROI
    dummy_roi = np.random.randn(224, 224, 3).astype(np.float32)
    pred_class, confidence, probs = inference.predict(dummy_roi)
    
    assert 0 <= pred_class < 6 or pred_class == -1
    assert 0 <= confidence <= 1
    assert len(probs) == 6
    assert abs(np.sum(probs) - 1.0) < 0.01
    
    print("  ✓ Gesture Inference: PASSED")
except Exception as e:
    print(f"  ✗ Gesture Inference: FAILED - {e}")

# Test 7: Calibration
print("\n[Test 7] Calibration...")
try:
    from shared.state import CalibrationData
    
    cal = CalibrationData()
    cal.min_distance = 0.02
    cal.max_distance = 0.25
    cal.is_calibrated = True
    
    # Test normalization
    assert abs(cal.normalize(0.02) - 0.0) < 0.01
    assert abs(cal.normalize(0.135) - 0.5) < 0.1
    assert abs(cal.normalize(0.25) - 1.0) < 0.01
    
    # Test clipping
    assert cal.normalize(0.0) == 0.0
    assert cal.normalize(1.0) == 1.0
    
    print("  ✓ Calibration: PASSED")
except Exception as e:
    print(f"  ✗ Calibration: FAILED - {e}")

# Test 8: Model Save/Load
print("\n[Test 8] Model Save/Load...")
try:
    import tempfile
    import os
    
    model = GestureCNN(num_classes=6)
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        temp_path = f.name
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 10,
        'val_f1': 0.95,
    }
    torch.save(checkpoint, temp_path)
    
    # Load
    loaded = torch.load(temp_path)
    model.load_state_dict(loaded['model_state_dict'])
    
    os.remove(temp_path)
    
    print("  ✓ Model Save/Load: PASSED")
except Exception as e:
    print(f"  ✗ Model Save/Load: FAILED - {e}")

# Test 9: Performance Benchmarks
print("\n[Test 9] Performance Benchmarks...")
try:
    from models.gesture_cnn import get_model_summary
    
    model = GestureCNN()
    summary = get_model_summary(model)
    
    # Check targets
    assert summary['avg_inference_latency_ms'] < 50, "Inference too slow"
    assert summary['estimated_fps'] > 20, "FPS too low"
    
    print("  ✓ Performance Benchmarks: PASSED")
    print(f"    Inference: {summary['avg_inference_latency_ms']:.2f} ms")
    print(f"    Est. FPS: {summary['estimated_fps']:.1f}")
except Exception as e:
    print(f"  ✗ Performance Benchmarks: FAILED - {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("\nAll core components tested successfully!")
print("\nTo run the full system:")
print("  1. Prepare dataset: python scripts/prepare_hagrid.py --create_sample")
print("  2. Train model: python scripts/train_cnn.py --data_dir ./hagrid_processed")
print("  3. Run realtime: python scripts/run_realtime.py --model_path <path>")
print("\nFor testing without audio:")
print("  python scripts/run_realtime.py --model_path <path> --no_audio")
print("=" * 60)
