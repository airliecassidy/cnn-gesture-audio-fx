"""
Audio module for real-time DSP effects processing.
"""
from .dsp import (
    GainEffect, LowpassFilter, HighpassFilter, 
    DistortionEffect, DelayEffect, ReverbEffect,
    AudioEffectChain
)
from .stream import AudioStream

__all__ = [
    'GainEffect', 'LowpassFilter', 'HighpassFilter',
    'DistortionEffect', 'DelayEffect', 'ReverbEffect',
    'AudioEffectChain', 'AudioStream'
]
