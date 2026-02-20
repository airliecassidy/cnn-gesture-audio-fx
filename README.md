# cnn-gesture-audio-fx
Gesture-controlled audio effects engine. Right hand picks the effect (gain, lowpass, highpass, distortion, delay, reverb), left hand pinch distance controls the depth, with live audio, at 30 fps. Built with MediaPipe Hands, a MobileNetV2 CNN trained on HaGRID, and a PyAudio DSP pipeline.

This project is an expansion of my previous MediaPipe-Max-Gesture-Audio-Effects. 

# Hand Role Logic
Right Hand: effect selection - gesture classification via CNN
- Effect selection works, depth frozen at last value
Left Hand: effect depth - pinch distance (euclidean distance between thumb and index finger) 
- Effect selection unchanged, depth control active.

# Gesture to Effect Mapping

| Gesture | Audio Effect | Depth Mapping |
| :--- | :---: | ---: |
| Palm | Gain | 0dB to +20dB boost |
| Fist | Lowpass Filter | 500Hz to 8kHz cutoff |
| Thumbs up | Highpass Filter | 50Hz to 4Hz cutoff |
| Thumbs down | Distortion | Mild to heavy drive |
| OK sign | Delay | 100ms to 500ms, feedback 0.1-0.8 |
| Peace Sign | Reverb | 0% to 60% wet mix |

# Controls 

Q: quit 
C: start/continue calibration 
R: reset effects
M: mute/unmute audio

# Calibration 
This system includes a calibration routine for pinch distance
1. Press C to start calibration
2. Pinch thumb and index finger CLOSED, press C to capture minimum
3. Spread thumb and index finger OPEN, press C to capture maximum
4. Calibration complete - depth control is now personalized


# Model Statistics 

| Metric | Value |
| :--- | :---: |
| Total parameters | ~3.5M | 
| Trainable parameters | ~3.5M |
| Model size | ~14MB |
| CPU | ~5-10ms |
| GPU | ~1-3ms |
| Estimated FPS | 100-200+ |

# DSP Effects

**Gain**
- Depth 0.0: Unity gain (1.0x)
- Depth 1.0: +20dB boost (~10x)

**Lowpass Filter**
- 4th-order Butterworth
- Depth 0.0: 500 Hz cutoff
- Depth 1.0: 8000 Hz cutoff

**Highpass Filter**
- 4th-order Butterworth
- Depth 0.0: 50 Hz cutoff
- Depth 1.0: 4000 Hz cutoff

**Distortion**
- Soft clipping using tanh waveshaping
- Depth 0.0: Mild overdrive (1x drive)
- Depth 1.0: Heavy distortion (20x drive)

**Delay**
- Circular buffer implementation
- Depth 0.0: 100ms delay, 0.1 feedback
- Depth 1.0: 500ms delay, 0.8 feedback

**Reverb**
- Schroeder reverb (4 comb + 3 all-pass filters)
- Depth 0.0: Dry signal (0% wet)
- Depth 1.0: 60% wet mix


# Performance Targets of Model

# Training Results
| Metric | Target | Expected |
| :--- | :---: | ---: |
| Accuracy | 90% | 92-96% |
| Macro F1 | 0.90 | 0.92-0.95 |


# Notes
HaGRID Dataset: hukenovs/hagrid
MediaPipe: Google AI
MobileNetV2: Google Research

