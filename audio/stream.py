"""
Real-time audio I/O using PyAudio callback streaming.
"""
import numpy as np
import pyaudio
import threading
import time
from typing import Optional, Callable
from collections import deque

from .dsp import AudioEffectChain


class AudioStream:
    """
    Real-time audio stream with effect processing.
    
    Uses PyAudio callback for low-latency processing.
    """
    
    # Audio settings
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_CHANNELS = 1
    DEFAULT_FORMAT = pyaudio.paFloat32
    DEFAULT_BUFFER_SIZE = 256
    
    def __init__(self, 
                 shared_state,
                 sample_rate: int = DEFAULT_SAMPLE_RATE,
                 channels: int = DEFAULT_CHANNELS,
                 buffer_size: int = DEFAULT_BUFFER_SIZE,
                 input_device_index: Optional[int] = None,
                 output_device_index: Optional[int] = None):
        """
        Initialize audio stream.
        
        Args:
            shared_state: SharedState object for communication
            sample_rate: Audio sample rate
            channels: Number of channels (1 for mono)
            buffer_size: Buffer size in samples
            input_device_index: Input device index (None for default)
            output_device_index: Output device index (None for default)
        """
        self.shared_state = shared_state
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        self.input_device_index = input_device_index
        self.output_device_index = output_device_index
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        
        # Create effect chain
        self.effect_chain = AudioEffectChain(sample_rate)
        
        # Stream handle
        self.stream: Optional[pyaudio.Stream] = None
        self.is_streaming = False
        
        # Performance monitoring
        self.callback_times = deque(maxlen=100)
        self.buffer_underruns = 0
        self.buffer_overruns = 0
        
        # Preallocate output buffer (avoid allocation in callback)
        self._output_buffer = np.zeros(buffer_size, dtype=np.float32)
    
    def list_devices(self):
        """List available audio devices."""
        print("\nAvailable Audio Devices:")
        print("=" * 50)
        
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            print(f"Device {i}: {info['name']}")
            print(f"  Input channels: {info['maxInputChannels']}")
            print(f"  Output channels: {info['maxOutputChannels']}")
            print(f"  Default sample rate: {info['defaultSampleRate']}")
            print()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio callback function called by PyAudio.
        
        This runs in a separate thread and must be fast!
        """
        start_time = time.perf_counter()
        
        # Check for errors
        if status:
            if status & pyaudio.paInputUnderflow:
                self.buffer_underruns += 1
            if status & pyaudio.paInputOverflow:
                self.buffer_overruns += 1
        
        # Convert input bytes to numpy array
        input_buffer = np.frombuffer(in_data, dtype=np.float32)
        
        # Get current effect parameters from shared state
        effect_idx, effect_depth = self.shared_state.get_audio_params()
        
        # Update effect chain
        self.effect_chain.set_effect_by_index(effect_idx)
        self.effect_chain.set_depth(effect_depth)
        
        # Process audio
        output_buffer = self.effect_chain.process(input_buffer)
        
        # Ensure correct size
        if len(output_buffer) != frame_count:
            output_buffer = np.resize(output_buffer, frame_count)
        
        # Convert back to bytes
        out_data = output_buffer.astype(np.float32).tobytes()
        
        # Update timing stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.callback_times.append(elapsed_ms)
        self.shared_state.audio_callback_time_ms = elapsed_ms
        
        return (out_data, pyaudio.paContinue)
    
    def start(self):
        """Start audio stream."""
        if self.is_streaming:
            print("Audio stream already running")
            return
        
        try:
            self.stream = self.pa.open(
                format=self.DEFAULT_FORMAT,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                output=True,
                frames_per_buffer=self.buffer_size,
                input_device_index=self.input_device_index,
                output_device_index=self.output_device_index,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_streaming = True
            
            print(f"Audio stream started:")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Channels: {self.channels}")
            print(f"  Buffer size: {self.buffer_size} samples")
            print(f"  Latency: {self.buffer_size / self.sample_rate * 1000:.1f} ms")
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            raise
    
    def stop(self):
        """Stop audio stream."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        print("Audio stream stopped")
    
    def get_stats(self) -> dict:
        """Get audio processing statistics."""
        if len(self.callback_times) == 0:
            return {
                'avg_callback_time_ms': 0.0,
                'max_callback_time_ms': 0.0,
                'buffer_underruns': self.buffer_underruns,
                'buffer_overruns': self.buffer_overruns,
            }
        
        times = list(self.callback_times)
        buffer_duration_ms = self.buffer_size / self.sample_rate * 1000
        
        return {
            'avg_callback_time_ms': np.mean(times),
            'max_callback_time_ms': np.max(times),
            'buffer_duration_ms': buffer_duration_ms,
            'cpu_usage_percent': np.mean(times) / buffer_duration_ms * 100,
            'buffer_underruns': self.buffer_underruns,
            'buffer_overruns': self.buffer_overruns,
        }
    
    def print_stats(self):
        """Print audio statistics."""
        stats = self.get_stats()
        print("\nAudio Stream Statistics:")
        print("=" * 40)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
    
    def release(self):
        """Release resources."""
        self.stop()
        self.pa.terminate()


class AudioPassthrough:
    """
    Simple passthrough audio stream for testing without effects.
    """
    
    def __init__(self, buffer_size: int = 256):
        self.buffer_size = buffer_size
        self.pa = pyaudio.PyAudio()
        self.stream = None
    
    def _callback(self, in_data, frame_count, time_info, status):
        """Simple passthrough callback."""
        return (in_data, pyaudio.paContinue)
    
    def start(self):
        """Start passthrough stream."""
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=48000,
            input=True,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self._callback
        )
        self.stream.start_stream()
        print("Passthrough audio started")
    
    def stop(self):
        """Stop passthrough stream."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
    
    def release(self):
        """Release resources."""
        self.stop()
        self.pa.terminate()


if __name__ == '__main__':
    # Test audio stream
    import sys
    sys.path.insert(0, '/mnt/okcomputer/output/gesture_audio_effects')
    from shared.state import SharedState
    
    # List devices
    test_stream = AudioStream(SharedState())
    test_stream.list_devices()
    test_stream.release()
    
    # Test passthrough
    print("\nTesting passthrough (5 seconds)...")
    passthrough = AudioPassthrough()
    passthrough.start()
    time.sleep(5)
    passthrough.stop()
    passthrough.release()
    print("Test complete")
