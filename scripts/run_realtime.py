"""
Real-time gesture-controlled audio effects system.

Integrates:
- MediaPipe Hands for dual hand tracking
- CNN for gesture classification (right hand)
- Pinch distance for effect depth (left hand)
- PyAudio for real-time DSP
"""
import os
import sys
import json
import argparse
import threading
import time
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.state import SharedState
from vision.hand_tracker import HandTracker, TemporalSmoother, ClassSmoother
from vision.inference import GestureInference
from audio.stream import AudioStream
from hagrid.dataset import CLASS_NAMES, IDX_TO_CLASS, GESTURE_TO_EFFECT


class CalibrationMode:
    """Calibration routine for pinch distance bounds."""
    
    def __init__(self, shared_state: SharedState):
        self.shared_state = shared_state
        self.min_samples = deque(maxlen=30)
        self.max_samples = deque(maxlen=30)
        self.state = 'idle'  # 'idle', 'capturing_min', 'capturing_max', 'done'
    
    def start(self):
        """Start calibration."""
        self.state = 'capturing_min'
        self.min_samples.clear()
        self.max_samples.clear()
        print("\n=== CALIBRATION MODE ===")
        print("Step 1: Pinch thumb and index finger CLOSED")
        print("Press 'C' when ready to capture MINIMUM")
    
    def capture_min(self):
        """Capture minimum distance samples."""
        self.state = 'capturing_max'
        print("\nStep 2: Spread thumb and index finger OPEN")
        print("Press 'C' when ready to capture MAXIMUM")
    
    def capture_max(self):
        """Capture maximum distance samples."""
        self.state = 'done'
        
        # Calculate bounds
        if len(self.min_samples) > 0 and len(self.max_samples) > 0:
            min_dist = np.median(self.min_samples)
            max_dist = np.median(self.max_samples)
            
            # Ensure reasonable bounds
            if max_dist - min_dist < 0.05:
                print("\nWarning: Range too small, using defaults")
                min_dist = 0.02
                max_dist = 0.25
            
            self.shared_state.update_calibration(min_dist, max_dist)
            
            print(f"\n=== CALIBRATION COMPLETE ===")
            print(f"Minimum distance: {min_dist:.4f}")
            print(f"Maximum distance: {max_dist:.4f}")
            print(f"Range: {max_dist - min_dist:.4f}")
        else:
            print("\nCalibration failed: not enough samples")
        
        self.state = 'idle'
    
    def add_sample(self, distance: float):
        """Add a distance sample."""
        if self.state == 'capturing_min':
            self.min_samples.append(distance)
        elif self.state == 'capturing_max':
            self.max_samples.append(distance)
    
    def is_active(self) -> bool:
        """Check if calibration is in progress."""
        return self.state != 'idle'


class GestureAudioSystem:
    """
    Main system integrating vision and audio.
    """
    
    def __init__(self, 
                 model_path: str,
                 confidence_threshold: float = 0.6,
                 use_audio: bool = True):
        """
        Initialize the gesture-controlled audio system.
        
        Args:
            model_path: Path to trained CNN checkpoint
            confidence_threshold: Minimum confidence for gesture recognition
            use_audio: Whether to enable audio processing
        """
        self.shared_state = SharedState()
        self.use_audio = use_audio
        
        # Initialize components
        print("Initializing hand tracker...")
        self.hand_tracker = HandTracker(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        print("Initializing gesture inference...")
        self.gesture_inference = GestureInference(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Temporal smoothers
        self.pinch_smoother = TemporalSmoother(method='ema', alpha=0.5)
        self.class_smoother = ClassSmoother(window_size=7)
        
        # Calibration
        self.calibration = CalibrationMode(self.shared_state)
        
        # Audio stream
        self.audio_stream: Optional[AudioStream] = None
        if use_audio:
            print("Initializing audio stream...")
            self.audio_stream = AudioStream(self.shared_state)
        
        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.last_effect_change_time = 0
        self.effect_cooldown = 0.3  # seconds
        
        # Visualization colors
        self.effect_colors = {
            'gain': (0, 255, 0),        # Green
            'lowpass': (255, 165, 0),   # Orange
            'highpass': (0, 165, 255),  # Light blue
            'distortion': (0, 0, 255),  # Red
            'delay': (255, 0, 255),     # Magenta
            'reverb': (128, 0, 128),    # Purple
        }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image from webcam
        
        Returns:
            Annotated frame
        """
        start_time = time.perf_counter()
        
        # Detect hands
        right_hand, left_hand = self.hand_tracker.process_frame(frame)
        
        # Update hand presence in shared state
        self.shared_state.right_hand_present = right_hand is not None
        self.shared_state.left_hand_present = left_hand is not None
        
        # Process right hand (gesture classification)
        if right_hand is not None:
            # Extract ROI
            roi = self.hand_tracker.extract_roi(frame, right_hand)
            
            if roi is not None:
                # Run CNN inference
                pred_class, confidence, _ = self.gesture_inference.predict(roi)
                
                # Temporal smoothing
                smoothed_class, smoothed_conf = self.class_smoother.update(
                    pred_class, confidence
                )
                
                # Update effect if valid and not in cooldown
                current_time = time.time()
                if (smoothed_class >= 0 and 
                    current_time - self.last_effect_change_time > self.effect_cooldown):
                    
                    self.shared_state.current_effect_idx = smoothed_class
                    self.shared_state.effect_confidence = smoothed_conf
                    self.last_effect_change_time = current_time
            
            # Draw landmarks
            effect_name = self.shared_state.effect_label
            color = self.effect_colors.get(effect_name, (0, 255, 0))
            frame = self.hand_tracker.draw_landmarks(frame, right_hand, color)
        
        # Process left hand (pinch distance for depth control)
        if left_hand is not None:
            # Get pinch distance
            pinch_dist = left_hand.get_pinch_distance()
            
            # Add to calibration if active
            if self.calibration.is_active():
                self.calibration.add_sample(pinch_dist)
            
            # Smooth and normalize
            smoothed_dist = self.pinch_smoother.update(pinch_dist)
            normalized_depth = self.shared_state.normalize_pinch_distance(smoothed_dist)
            
            # Update shared state
            self.shared_state.effect_depth = normalized_depth
            self.shared_state.left_hand_landmarks = left_hand.landmarks
            
            # Draw landmarks
            frame = self.hand_tracker.draw_landmarks(frame, left_hand, (255, 0, 0))
        else:
            # If left hand not present, keep last depth value
            pass
        
        # Calculate FPS
        frame_time = (time.perf_counter() - start_time) * 1000
        self.frame_times.append(frame_time)
        avg_frame_time = np.mean(self.frame_times)
        fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
        self.shared_state.fps = fps
        
        return frame
    
    def draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw user interface overlay.
        
        Args:
            frame: BGR image
        
        Returns:
            Frame with UI overlay
        """
        h, w = frame.shape[:2]
        
        # Create overlay for semi-transparent UI
        overlay = frame.copy()
        
        # Draw info panel background
        panel_x = 10
        panel_y = 10
        panel_w = 350
        panel_h = 180
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_w, panel_y + panel_h),
                     (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw text
        x = panel_x + 10
        y = panel_y + 30
        
        # Current effect
        effect_name = self.shared_state.effect_label.upper()
        effect_idx = self.shared_state.current_effect_idx
        confidence = self.shared_state.effect_confidence
        color = self.effect_colors.get(effect_name.lower(), (0, 255, 0))
        
        cv2.putText(frame, f"EFFECT: {effect_name}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 30
        
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        
        # Effect depth
        depth = self.shared_state.effect_depth
        cv2.putText(frame, f"Depth: {depth:.2f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw depth bar
        bar_x = x + 120
        bar_y = y - 10
        bar_w = 150
        bar_h = 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     (100, 100, 100), -1)
        filled_w = int(bar_w * depth)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h),
                     (0, 255, 0), -1)
        y += 30
        
        # Hand status
        right_status = "DETECTED" if self.shared_state.right_hand_present else "NOT DETECTED"
        left_status = "DETECTED" if self.shared_state.left_hand_present else "NOT DETECTED"
        
        right_color = (0, 255, 0) if self.shared_state.right_hand_present else (0, 0, 255)
        left_color = (0, 255, 0) if self.shared_state.left_hand_present else (0, 0, 255)
        
        cv2.putText(frame, f"Right Hand: {right_status}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_color, 1)
        y += 25
        
        cv2.putText(frame, f"Left Hand: {left_status}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_color, 1)
        y += 25
        
        # FPS
        cv2.putText(frame, f"FPS: {self.shared_state.fps:.1f}", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        
        # Audio callback time
        audio_time = self.shared_state.audio_callback_time_ms
        cv2.putText(frame, f"Audio: {audio_time:.1f}ms", (x, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw effect legend
        legend_x = w - 200
        legend_y = 10
        legend_h = len(self.effect_colors) * 25 + 20
        
        cv2.rectangle(overlay, (legend_x, legend_y),
                     (legend_x + 190, legend_y + legend_h),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.putText(frame, "EFFECTS:", (legend_x + 10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y = legend_y + 50
        for effect, color in self.effect_colors.items():
            marker = ">" if effect == effect_name.lower() else " "
            cv2.putText(frame, f"{marker} {effect.upper()}", (legend_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 22
        
        # Calibration status
        if self.calibration.is_active():
            cal_text = f"CALIBRATING: {self.calibration.state}"
            cv2.putText(frame, cal_text, (w // 2 - 150, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        elif not self.shared_state.is_calibrated:
            cv2.putText(frame, "Press 'C' to calibrate", (w // 2 - 150, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Controls help
        help_text = "Q:Quit | C:Calibrate | R:Reset | M:Mute"
        cv2.putText(frame, help_text, (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main run loop."""
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start audio
        if self.audio_stream:
            self.audio_stream.start()
        
        print("\n" + "=" * 60)
        print("GESTURE-CONTROLLED AUDIO EFFECTS SYSTEM")
        print("=" * 60)
        print("\nControls:")
        print("  Q - Quit")
        print("  C - Start/Continue calibration")
        print("  R - Reset effects")
        print("  M - Mute/Unmute audio")
        print("\nHand roles:")
        print("  RIGHT HAND - Select effect (gesture)")
        print("  LEFT HAND - Control depth (pinch distance)")
        print("\n" + "=" * 60)
        
        muted = False
        
        try:
            while self.shared_state.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Draw UI
                frame = self.draw_ui(frame)
                
                # Show frame
                cv2.imshow('Gesture Audio Effects', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                
                elif key == ord('c'):
                    if not self.calibration.is_active():
                        self.calibration.start()
                    elif self.calibration.state == 'capturing_min':
                        self.calibration.capture_min()
                    elif self.calibration.state == 'capturing_max':
                        self.calibration.capture_max()
                
                elif key == ord('r'):
                    # Reset effects
                    self.shared_state.current_effect_idx = 0
                    self.shared_state.effect_depth = 0.5
                    self.pinch_smoother.reset()
                    self.class_smoother.reset()
                    print("Effects reset")
                
                elif key == ord('m'):
                    muted = not muted
                    if muted:
                        self.shared_state.effect_depth = 0.0
                        print("Audio muted")
                    else:
                        self.shared_state.effect_depth = 0.5
                        print("Audio unmuted")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()
            
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.print_stats()
                self.audio_stream.release()
            
            print("\nSystem shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time gesture-controlled audio effects'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                       help='Minimum confidence for gesture recognition')
    parser.add_argument('--no_audio', action='store_true',
                       help='Disable audio processing (vision only)')
    
    args = parser.parse_args()
    
    # Create and run system
    system = GestureAudioSystem(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        use_audio=not args.no_audio
    )
    
    system.run()


if __name__ == '__main__':
    main()
