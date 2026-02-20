"""
MediaPipe-based hand tracking with dual hand support.
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List, Dict
from collections import deque
from dataclasses import dataclass


@dataclass
class HandData:
    """Container for hand tracking data."""
    landmarks: np.ndarray  # 21x3 array of (x, y, z) coordinates
    handedness: str  # 'Left' or 'Right'
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    
    @property
    def thumb_tip(self) -> np.ndarray:
        """Get thumb tip landmark (index 4)."""
        return self.landmarks[4]
    
    @property
    def index_tip(self) -> np.ndarray:
        """Get index finger tip landmark (index 8)."""
        return self.landmarks[8]
    
    def get_pinch_distance(self) -> float:
        """Calculate Euclidean distance between thumb and index finger tips."""
        return np.linalg.norm(self.thumb_tip[:2] - self.index_tip[:2])


class TemporalSmoother:
    """
    Temporal smoothing using exponential moving average (EMA) or voting.
    """
    
    def __init__(self, window_size: int = 5, method: str = 'ema', alpha: float = 0.7):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Size of history window for voting method
            method: 'ema' or 'voting'
            alpha: EMA smoothing factor (higher = more responsive)
        """
        self.window_size = window_size
        self.method = method
        self.alpha = alpha
        self.history: deque = deque(maxlen=window_size)
        self.ema_value: Optional[float] = None
    
    def update(self, value: float) -> float:
        """
        Update smoother with new value and return smoothed value.
        
        Args:
            value: New value to add
        
        Returns:
            Smoothed value
        """
        if self.method == 'ema':
            if self.ema_value is None:
                self.ema_value = value
            else:
                self.ema_value = self.alpha * value + (1 - self.alpha) * self.ema_value
            return self.ema_value
        
        elif self.method == 'voting':
            self.history.append(value)
            if len(self.history) == 0:
                return value
            # Return most common value in history
            return max(set(self.history), key=list(self.history).count)
        
        else:
            return value
    
    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.ema_value = None


class ClassSmoother:
    """Temporal smoother for classification outputs."""
    
    def __init__(self, window_size: int = 7, confidence_threshold: float = 0.6):
        """
        Initialize classification smoother.
        
        Args:
            window_size: Number of frames to consider
            confidence_threshold: Minimum confidence to accept prediction
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
    
    def update(self, class_idx: int, confidence: float) -> Tuple[int, float]:
        """
        Update with new prediction and return smoothed result.
        
        Args:
            class_idx: Predicted class index
            confidence: Prediction confidence
        
        Returns:
            Tuple of (smoothed_class, smoothed_confidence)
        """
        # Only add to history if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.history.append(class_idx)
            self.confidence_history.append(confidence)
        
        if len(self.history) == 0:
            return class_idx, confidence
        
        # Majority voting
        from collections import Counter
        vote_counts = Counter(self.history)
        smoothed_class = vote_counts.most_common(1)[0][0]
        
        # Average confidence for winning class
        winning_confidences = [c for c, h in zip(self.confidence_history, self.history) 
                              if h == smoothed_class]
        smoothed_confidence = np.mean(winning_confidences) if winning_confidences else confidence
        
        return smoothed_class, smoothed_confidence
    
    def reset(self):
        """Reset smoother state."""
        self.history.clear()
        self.confidence_history.clear()


class HandTracker:
    """
    MediaPipe-based hand tracker for dual hand detection.
    """
    
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    
    def __init__(self, 
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 roi_margin: float = 0.2):
        """
        Initialize hand tracker.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
            roi_margin: Margin around hand landmarks for ROI (fraction of bbox)
        """
        self.roi_margin = roi_margin
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Temporal smoothers
        self.pinch_smoother = TemporalSmoother(method='ema', alpha=0.6)
        self.class_smoother = ClassSmoother()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[HandData], Optional[HandData]]:
        """
        Process a frame and extract hand data.
        
        Args:
            frame: BGR image from webcam
        
        Returns:
            Tuple of (right_hand_data, left_hand_data)
            Each can be None if hand not detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        right_hand: Optional[HandData] = None
        left_hand: Optional[HandData] = None
        
        if results.multi_hand_landmarks:
            for idx, (hand_landmarks, handedness) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                # Extract landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                
                # Get handedness label
                hand_label = handedness.classification[0].label  # 'Left' or 'Right'
                confidence = handedness.classification[0].score
                
                # Calculate bounding box
                h, w = frame.shape[:2]
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                
                x_min = int(np.min(x_coords) * w)
                x_max = int(np.max(x_coords) * w)
                y_min = int(np.min(y_coords) * h)
                y_max = int(np.max(y_coords) * h)
                
                # Add margin
                margin_x = int((x_max - x_min) * self.roi_margin)
                margin_y = int((y_max - y_min) * self.roi_margin)
                
                x_min = max(0, x_min - margin_x)
                x_max = min(w, x_max + margin_x)
                y_min = max(0, y_min - margin_y)
                y_max = min(h, y_max + margin_y)
                
                bbox = (x_min, y_min, x_max, y_max)
                
                # Create hand data
                hand_data = HandData(
                    landmarks=landmarks,
                    handedness=hand_label,
                    confidence=confidence,
                    bbox=bbox
                )
                
                # Assign to right or left hand
                # Note: MediaPipe's 'Left' means the person's left hand (appears on right of image)
                if hand_label == 'Right':
                    right_hand = hand_data
                else:
                    left_hand = hand_data
        
        return right_hand, left_hand
    
    def extract_roi(self, frame: np.ndarray, hand_data: HandData, 
                    target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract and preprocess hand ROI for CNN inference.
        
        Args:
            frame: Original BGR frame
            hand_data: Hand data with bounding box
            target_size: Target size for CNN input
        
        Returns:
            Preprocessed ROI image (RGB, normalized)
        """
        x_min, y_min, x_max, y_max = hand_data.bbox
        
        # Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            # Return blank image if ROI is empty
            return np.zeros((*target_size, 3), dtype=np.float32)
        
        # Resize to target size
        roi_resized = cv2.resize(roi, target_size)
        
        # Convert BGR to RGB
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        roi_normalized = roi_rgb.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        roi_normalized = (roi_normalized - mean) / std
        
        return roi_normalized
    
    def draw_landmarks(self, frame: np.ndarray, hand_data: HandData, 
                       color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw hand landmarks on frame.
        
        Args:
            frame: BGR image
            hand_data: Hand data
            color: Color for landmarks
        
        Returns:
            Frame with landmarks drawn
        """
        # Convert landmarks to MediaPipe format
        h, w = frame.shape[:2]
        landmarks_proto = self.mp_hands.HandLandmark
        
        # Create landmark list for drawing
        landmark_list = self.mp_hands.HandLandmark
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = hand_data.bbox
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(hand_data.landmarks):
            px, py = int(x * w), int(y * h)
            cv2.circle(frame, (px, py), 5, color, -1)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1)
        
        # Draw connections
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            x1, y1 = int(hand_data.landmarks[start_idx][0] * w), int(hand_data.landmarks[start_idx][1] * h)
            x2, y2 = int(hand_data.landmarks[end_idx][0] * w), int(hand_data.landmarks[end_idx][1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Highlight thumb and index tips
        thumb_px = int(hand_data.thumb_tip[0] * w)
        thumb_py = int(hand_data.thumb_tip[1] * h)
        index_px = int(hand_data.index_tip[0] * w)
        index_py = int(hand_data.index_tip[1] * h)
        
        cv2.circle(frame, (thumb_px, thumb_py), 8, (0, 0, 255), -1)
        cv2.circle(frame, (index_px, index_py), 8, (255, 0, 0), -1)
        cv2.line(frame, (thumb_px, thumb_py), (index_px, index_py), (255, 255, 0), 2)
        
        # Draw pinch distance
        pinch_dist = hand_data.get_pinch_distance()
        mid_x = (thumb_px + index_px) // 2
        mid_y = (thumb_py + index_py) // 2
        cv2.putText(frame, f"{pinch_dist:.3f}", (mid_x, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def get_smoothed_pinch(self, hand_data: HandData) -> float:
        """Get temporally smoothed pinch distance."""
        pinch_dist = hand_data.get_pinch_distance()
        return self.pinch_smoother.update(pinch_dist)
    
    def get_smoothed_class(self, class_idx: int, confidence: float) -> Tuple[int, float]:
        """Get temporally smoothed classification."""
        return self.class_smoother.update(class_idx, confidence)
    
    def release(self):
        """Release resources."""
        self.hands.close()


if __name__ == '__main__':
    # Test hand tracker
    tracker = HandTracker()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 'c' to calibrate")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        right_hand, left_hand = tracker.process_frame(frame)
        
        # Draw results
        if right_hand:
            frame = tracker.draw_landmarks(frame, right_hand, (0, 255, 0))
            cv2.putText(frame, f"Right: {right_hand.confidence:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if left_hand:
            frame = tracker.draw_landmarks(frame, left_hand, (255, 0, 0))
            cv2.putText(frame, f"Left: {left_hand.get_pinch_distance():.3f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Hand Tracker', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
