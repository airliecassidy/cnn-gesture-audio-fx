"""
Real-time DSP effects implemented in NumPy.

All effects expose a depth parameter (0.0 to 1.0) that maps to
meaningful effect parameters.
"""
import numpy as np
from typing import Optional, Tuple
from scipy import signal
from collections import deque
import time


class AudioEffect:
    """Base class for audio effects."""
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.depth = 0.5  # 0.0 to 1.0
    
    def set_depth(self, depth: float):
        """Set effect depth (0.0 to 1.0)."""
        self.depth = np.clip(depth, 0.0, 1.0)
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """
        Process audio buffer.
        
        Args:
            input_buffer: Input audio samples (mono, float32)
        
        Returns:
            Processed audio samples
        """
        raise NotImplementedError
    
    def reset(self):
        """Reset effect state."""
        pass


class GainEffect(AudioEffect):
    """
    Gain/boost effect.
    
    Depth mapping:
        0.0: Unity gain (1.0x)
        1.0: +20dB boost (~10x)
    """
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.min_db = 0.0
        self.max_db = 20.0
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply gain."""
        # Map depth to dB
        gain_db = self.min_db + self.depth * (self.max_db - self.min_db)
        gain_linear = 10 ** (gain_db / 20.0)
        
        return input_buffer * gain_linear


class LowpassFilter(AudioEffect):
    """
    Lowpass filter effect.
    
    Depth mapping:
        0.0: 500 Hz cutoff
        1.0: 8000 Hz cutoff
    """
    
    def __init__(self, sample_rate: int = 48000, order: int = 4):
        super().__init__(sample_rate)
        self.order = order
        self.min_freq = 500.0
        self.max_freq = 8000.0
        
        # Filter state (for continuity)
        self.zi = None
        self._prev_cutoff = None
        self._b = None
        self._a = None
    
    def _design_filter(self, cutoff: float):
        """Design Butterworth lowpass filter."""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = np.clip(cutoff / nyquist, 0.01, 0.99)
        
        b, a = signal.butter(self.order, normalized_cutoff, btype='low')
        return b, a
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply lowpass filter."""
        # Map depth to cutoff frequency
        cutoff = self.min_freq + self.depth * (self.max_freq - self.min_freq)
        
        # Redesign filter if cutoff changed significantly
        if self._prev_cutoff is None or abs(cutoff - self._prev_cutoff) > 50:
            self._b, self._a = self._design_filter(cutoff)
            self._prev_cutoff = cutoff
            # Reset state for new filter
            self.zi = signal.lfiltic(self._b, self._a, [0] * self.order)
        
        # Apply filter
        output, self.zi = signal.lfilter(self._b, self._a, input_buffer, zi=self.zi)
        
        return output.astype(np.float32)
    
    def reset(self):
        """Reset filter state."""
        self.zi = None
        self._prev_cutoff = None
        self._b = None
        self._a = None


class HighpassFilter(AudioEffect):
    """
    Highpass filter effect.
    
    Depth mapping:
        0.0: 50 Hz cutoff
        1.0: 4000 Hz cutoff
    """
    
    def __init__(self, sample_rate: int = 48000, order: int = 4):
        super().__init__(sample_rate)
        self.order = order
        self.min_freq = 50.0
        self.max_freq = 4000.0
        
        self.zi = None
        self._prev_cutoff = None
        self._b = None
        self._a = None
    
    def _design_filter(self, cutoff: float):
        """Design Butterworth highpass filter."""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = np.clip(cutoff / nyquist, 0.001, 0.99)
        
        b, a = signal.butter(self.order, normalized_cutoff, btype='high')
        return b, a
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply highpass filter."""
        cutoff = self.min_freq + self.depth * (self.max_freq - self.min_freq)
        
        if self._prev_cutoff is None or abs(cutoff - self._prev_cutoff) > 50:
            self._b, self._a = self._design_filter(cutoff)
            self._prev_cutoff = cutoff
            self.zi = signal.lfiltic(self._b, self._a, [0] * self.order)
        
        output, self.zi = signal.lfilter(self._b, self._a, input_buffer, zi=self.zi)
        
        return output.astype(np.float32)
    
    def reset(self):
        """Reset filter state."""
        self.zi = None
        self._prev_cutoff = None
        self._b = None
        self._a = None


class DistortionEffect(AudioEffect):
    """
    Distortion/overdrive effect using waveshaping.
    
    Depth mapping:
        0.0: Mild overdrive
        1.0: Heavy distortion
    """
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.min_drive = 1.0
        self.max_drive = 20.0
        self.mix = 0.7  # Wet/dry mix
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply distortion."""
        # Map depth to drive amount
        drive = self.min_drive + self.depth * (self.max_drive - self.min_drive)
        
        # Apply pre-gain
        gained = input_buffer * drive
        
        # Waveshaping (soft clipping)
        # Using hyperbolic tangent for smooth distortion
        distorted = np.tanh(gained)
        
        # Mix wet and dry
        output = (1 - self.mix) * input_buffer + self.mix * distorted
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(output))
        if max_val > 1.0:
            output = output / max_val * 0.9
        
        return output.astype(np.float32)


class DelayEffect(AudioEffect):
    """
    Delay/echo effect.
    
    Depth mapping:
        0.0: 100ms delay, 0.1 feedback
        1.0: 500ms delay, 0.8 feedback
    """
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.min_delay_ms = 100.0
        self.max_delay_ms = 500.0
        self.min_feedback = 0.1
        self.max_feedback = 0.8
        
        # Maximum delay buffer (500ms at 48kHz = 24000 samples)
        max_delay_samples = int(0.5 * sample_rate)
        self.delay_buffer = np.zeros(max_delay_samples, dtype=np.float32)
        self.write_idx = 0
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply delay effect."""
        # Map depth to parameters
        delay_ms = self.min_delay_ms + self.depth * (self.max_delay_ms - self.min_delay_ms)
        feedback = self.min_feedback + self.depth * (self.max_feedback - self.min_feedback)
        
        delay_samples = int(delay_ms * self.sample_rate / 1000.0)
        delay_samples = min(delay_samples, len(self.delay_buffer) - 1)
        
        output = np.zeros_like(input_buffer)
        
        for i in range(len(input_buffer)):
            # Read from delay buffer
            read_idx = (self.write_idx - delay_samples) % len(self.delay_buffer)
            delayed_sample = self.delay_buffer[read_idx]
            
            # Mix input with delayed signal
            output[i] = input_buffer[i] + delayed_sample * feedback
            
            # Write to delay buffer
            self.delay_buffer[self.write_idx] = input_buffer[i] + delayed_sample * feedback
            
            self.write_idx = (self.write_idx + 1) % len(self.delay_buffer)
        
        # Soft clipping to prevent runaway feedback
        output = np.tanh(output)
        
        return output.astype(np.float32)
    
    def reset(self):
        """Clear delay buffer."""
        self.delay_buffer.fill(0.0)
        self.write_idx = 0


class ReverbEffect(AudioEffect):
    """
    Simple Schroeder reverb using comb and all-pass filters.
    
    Depth mapping:
        0.0: Dry signal with minimal reverb
        1.0: 60% wet mix with full reverb
    """
    
    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.min_wet = 0.0
        self.max_wet = 0.6
        
        # Comb filter delays (in samples) - prime numbers for diffusion
        self.comb_delays = [1553, 1613, 1663, 1721]
        self.comb_feedback = [0.805, 0.827, 0.783, 0.764]
        
        # All-pass filter delays
        self.ap_delays = [225, 341, 441]
        self.ap_feedback = 0.7
        
        # Initialize comb filter buffers
        self.comb_buffers = []
        self.comb_indices = []
        for delay in self.comb_delays:
            self.comb_buffers.append(np.zeros(delay, dtype=np.float32))
            self.comb_indices.append(0)
        
        # Initialize all-pass filter buffers
        self.ap_buffers = []
        self.ap_indices = []
        for delay in self.ap_delays:
            self.ap_buffers.append(np.zeros(delay, dtype=np.float32))
            self.ap_indices.append(0)
        
        # Lowpass filter for damping (simple 1-pole)
        self.damping = 0.4
        self.last_comb_outputs = np.zeros(len(self.comb_delays), dtype=np.float32)
    
    def _process_comb(self, input_sample: float, buffer_idx: int) -> float:
        """Process single comb filter."""
        buffer = self.comb_buffers[buffer_idx]
        idx = self.comb_indices[buffer_idx]
        
        # Read delayed sample
        delayed = buffer[idx]
        
        # Apply damping (lowpass)
        filtered = delayed * (1 - self.damping) + self.last_comb_outputs[buffer_idx] * self.damping
        self.last_comb_outputs[buffer_idx] = filtered
        
        # Write to buffer with feedback
        buffer[idx] = input_sample + filtered * self.comb_feedback[buffer_idx]
        
        self.comb_indices[buffer_idx] = (idx + 1) % len(buffer)
        
        return filtered
    
    def _process_allpass(self, input_sample: float, buffer_idx: int) -> float:
        """Process single all-pass filter."""
        buffer = self.ap_buffers[buffer_idx]
        idx = self.ap_indices[buffer_idx]
        
        # Read delayed sample
        delayed = buffer[idx]
        
        # All-pass formula: output = delayed - feedback * (input - output_delayed)
        # Simplified: output = delayed + feedback * (delayed - input)
        output = delayed + self.ap_feedback * (delayed - input_sample)
        
        # Write to buffer
        buffer[idx] = input_sample + self.ap_feedback * (input_sample - output)
        
        self.ap_indices[buffer_idx] = (idx + 1) % len(buffer)
        
        return output
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Apply reverb effect."""
        wet_mix = self.min_wet + self.depth * (self.max_wet - self.min_wet)
        
        output = np.zeros_like(input_buffer)
        
        for i in range(len(input_buffer)):
            sample = input_buffer[i]
            
            # Parallel comb filters
            comb_sum = 0.0
            for j in range(len(self.comb_delays)):
                comb_sum += self._process_comb(sample, j)
            
            # Series all-pass filters
            ap_input = comb_sum / len(self.comb_delays)
            for j in range(len(self.ap_delays)):
                ap_input = self._process_allpass(ap_input, j)
            
            # Mix wet and dry
            output[i] = (1 - wet_mix) * sample + wet_mix * ap_input
        
        return output.astype(np.float32)
    
    def reset(self):
        """Clear all filter buffers."""
        for buffer in self.comb_buffers:
            buffer.fill(0.0)
        for buffer in self.ap_buffers:
            buffer.fill(0.0)
        self.comb_indices = [0] * len(self.comb_delays)
        self.ap_indices = [0] * len(self.ap_delays)
        self.last_comb_outputs.fill(0.0)


class AudioEffectChain:
    """
    Chain of audio effects with selection.
    """
    
    EFFECT_NAMES = ['gain', 'lowpass', 'highpass', 'distortion', 'delay', 'reverb']
    
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        
        # Create all effects
        self.effects = {
            'gain': GainEffect(sample_rate),
            'lowpass': LowpassFilter(sample_rate),
            'highpass': HighpassFilter(sample_rate),
            'distortion': DistortionEffect(sample_rate),
            'delay': DelayEffect(sample_rate),
            'reverb': ReverbEffect(sample_rate),
        }
        
        self.current_effect = 'gain'
        self.current_depth = 0.5
    
    def set_effect(self, effect_name: str):
        """Set active effect."""
        if effect_name in self.effects:
            # Reset previous effect
            self.effects[self.current_effect].reset()
            self.current_effect = effect_name
    
    def set_depth(self, depth: float):
        """Set effect depth."""
        self.current_depth = np.clip(depth, 0.0, 1.0)
        self.effects[self.current_effect].set_depth(self.current_depth)
    
    def set_effect_by_index(self, effect_idx: int):
        """Set effect by index (0-5)."""
        if 0 <= effect_idx < len(self.EFFECT_NAMES):
            self.set_effect(self.EFFECT_NAMES[effect_idx])
    
    def process(self, input_buffer: np.ndarray) -> np.ndarray:
        """Process audio through current effect."""
        return self.effects[self.current_effect].process(input_buffer)
    
    def reset(self):
        """Reset all effects."""
        for effect in self.effects.values():
            effect.reset()


if __name__ == '__main__':
    # Test DSP effects
    sample_rate = 48000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create test signal (mix of frequencies)
    test_signal = (np.sin(2 * np.pi * 440 * t) + 
                   0.5 * np.sin(2 * np.pi * 880 * t) + 
                   0.25 * np.sin(2 * np.pi * 1760 * t))
    test_signal = test_signal.astype(np.float32) * 0.3
    
    print("Testing DSP effects...")
    
    # Test each effect
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
        
        start = time.time()
        output = effect.process(test_signal.copy())
        elapsed = (time.time() - start) * 1000
        
        print(f"{name}: {elapsed:.2f} ms for {len(test_signal)} samples")
        print(f"  Input RMS: {np.sqrt(np.mean(test_signal**2)):.4f}")
        print(f"  Output RMS: {np.sqrt(np.mean(output**2)):.4f}")
