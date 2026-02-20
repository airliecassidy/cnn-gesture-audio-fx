"""
Thread-safe shared state for communication between vision and audio threads.
"""
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class CalibrationData:
    """Stores calibration bounds for pinch distance."""
    min_distance: float = 0.02  # Default minimum (closed pinch)
    max_distance: float = 0.25  # Default maximum (open pinch)
    is_calibrated: bool = False
    
    def normalize(self, distance: float) -> float:
        """Normalize distance to 0-1 range based on calibration."""
        if not self.is_calibrated:
            # Use default bounds if not calibrated
            normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        else:
            normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        return np.clip(normalized, 0.0, 1.0)


class SharedState:
    """
    Thread-safe shared state for gesture-controlled audio effects.
    
    This class manages all shared data between the vision thread (hand tracking,
    gesture classification) and the audio thread (DSP effects processing).
    """
    
    # Effect labels mapped to indices
    EFFECT_LABELS = ['gain', 'lowpass', 'highpass', 'distortion', 'delay', 'reverb']
    
    def __init__(self):
        self._lock = threading.RLock()
        
        # Effect selection (from right hand gesture)
        self._current_effect_idx: int = 0  # Default to gain
        self._effect_confidence: float = 0.0
        self._effect_label: str = self.EFFECT_LABELS[0]
        
        # Effect depth control (from left hand pinch)
        self._effect_depth: float = 0.5  # Default mid-range
        self._calibration = CalibrationData()
        
        # Hand presence tracking
        self._right_hand_present: bool = False
        self._left_hand_present: bool = False
        
        # Raw landmark data for visualization
        self._right_hand_landmarks: Optional[np.ndarray] = None
        self._left_hand_landmarks: Optional[np.ndarray] = None
        
        # System state
        self._is_running: bool = True
        self._fps: float = 0.0
        self._audio_callback_time_ms: float = 0.0
        
        # Debug info
        self._debug_info: Dict[str, Any] = {}
    
    # Effect selection properties
    @property
    def current_effect_idx(self) -> int:
        with self._lock:
            return self._current_effect_idx
    
    @current_effect_idx.setter
    def current_effect_idx(self, value: int):
        with self._lock:
            self._current_effect_idx = max(0, min(value, len(self.EFFECT_LABELS) - 1))
            self._effect_label = self.EFFECT_LABELS[self._current_effect_idx]
    
    @property
    def effect_label(self) -> str:
        with self._lock:
            return self._effect_label
    
    @property
    def effect_confidence(self) -> float:
        with self._lock:
            return self._effect_confidence
    
    @effect_confidence.setter
    def effect_confidence(self, value: float):
        with self._lock:
            self._effect_confidence = np.clip(value, 0.0, 1.0)
    
    # Effect depth properties
    @property
    def effect_depth(self) -> float:
        with self._lock:
            return self._effect_depth
    
    @effect_depth.setter
    def effect_depth(self, value: float):
        with self._lock:
            self._effect_depth = np.clip(value, 0.0, 1.0)
    
    # Calibration methods
    def update_calibration(self, min_dist: float, max_dist: float):
        """Update calibration bounds."""
        with self._lock:
            self._calibration.min_distance = min_dist
            self._calibration.max_distance = max_dist
            self._calibration.is_calibrated = True
    
    def normalize_pinch_distance(self, distance: float) -> float:
        """Normalize pinch distance using calibration."""
        with self._lock:
            return self._calibration.normalize(distance)
    
    @property
    def is_calibrated(self) -> bool:
        with self._lock:
            return self._calibration.is_calibrated
    
    # Hand presence properties
    @property
    def right_hand_present(self) -> bool:
        with self._lock:
            return self._right_hand_present
    
    @right_hand_present.setter
    def right_hand_present(self, value: bool):
        with self._lock:
            self._right_hand_present = value
    
    @property
    def left_hand_present(self) -> bool:
        with self._lock:
            return self._left_hand_present
    
    @left_hand_present.setter
    def left_hand_present(self, value: bool):
        with self._lock:
            self._left_hand_present = value
    
    # Landmark properties
    @property
    def right_hand_landmarks(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._right_hand_landmarks.copy() if self._right_hand_landmarks is not None else None
    
    @right_hand_landmarks.setter
    def right_hand_landmarks(self, value: Optional[np.ndarray]):
        with self._lock:
            self._right_hand_landmarks = value.copy() if value is not None else None
    
    @property
    def left_hand_landmarks(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._left_hand_landmarks.copy() if self._left_hand_landmarks is not None else None
    
    @left_hand_landmarks.setter
    def left_hand_landmarks(self, value: Optional[np.ndarray]):
        with self._lock:
            self._left_hand_landmarks = value.copy() if value is not None else None
    
    # System state properties
    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._is_running
    
    @is_running.setter
    def is_running(self, value: bool):
        with self._lock:
            self._is_running = value
    
    @property
    def fps(self) -> float:
        with self._lock:
            return self._fps
    
    @fps.setter
    def fps(self, value: float):
        with self._lock:
            self._fps = value
    
    @property
    def audio_callback_time_ms(self) -> float:
        with self._lock:
            return self._audio_callback_time_ms
    
    @audio_callback_time_ms.setter
    def audio_callback_time_ms(self, value: float):
        with self._lock:
            self._audio_callback_time_ms = value
    
    # Debug info
    def set_debug_info(self, key: str, value: Any):
        with self._lock:
            self._debug_info[key] = value
    
    def get_debug_info(self, key: str) -> Any:
        with self._lock:
            return self._debug_info.get(key)
    
    def get_all_debug_info(self) -> Dict[str, Any]:
        with self._lock:
            return self._debug_info.copy()
    
    # Convenience method for audio thread
    def get_audio_params(self) -> tuple[int, float]:
        """Get current effect parameters for audio processing.
        
        Returns:
            Tuple of (effect_index, effect_depth)
        """
        with self._lock:
            return self._current_effect_idx, self._effect_depth
