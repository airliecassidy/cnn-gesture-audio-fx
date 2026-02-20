"""
Generate architecture visualization for the gesture-controlled audio effects system.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_module_diagram():
    """Create system architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'Gesture-Controlled Audio Effects System', 
           fontsize=18, fontweight='bold', ha='center')
    ax.text(8, 11.1, 'Module Architecture', fontsize=12, ha='center', style='italic')
    
    # Colors
    vision_color = '#4CAF50'
    audio_color = '#2196F3'
    shared_color = '#FF9800'
    data_color = '#9C27B0'
    model_color = '#E91E63'
    
    # Vision Thread Box
    vision_box = FancyBboxPatch((0.5, 6.5), 5, 4, 
                                boxstyle="round,pad=0.1", 
                                facecolor=vision_color, alpha=0.3,
                                edgecolor=vision_color, linewidth=2)
    ax.add_patch(vision_box)
    ax.text(3, 10.2, 'VISION THREAD', fontsize=11, fontweight='bold', 
           ha='center', color=vision_color)
    
    # Vision components
    components = [
        ('Webcam Capture\n(OpenCV)', 1.5, 9),
        ('MediaPipe Hands\n(Hand Tracking)', 3, 9),
        ('Hand Tracker', 1.5, 7.8),
        ('ROI Extraction', 3, 7.8),
        ('CNN Inference\n(PyTorch)', 2.25, 6.7),
    ]
    for text, x, y in components:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.02",
                            facecolor='white', alpha=0.9,
                            edgecolor=vision_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Audio Thread Box
    audio_box = FancyBboxPatch((10.5, 6.5), 5, 4,
                              boxstyle="round,pad=0.1",
                              facecolor=audio_color, alpha=0.3,
                              edgecolor=audio_color, linewidth=2)
    ax.add_patch(audio_box)
    ax.text(13, 10.2, 'AUDIO THREAD', fontsize=11, fontweight='bold',
           ha='center', color=audio_color)
    
    # Audio components
    audio_components = [
        ('PyAudio Stream\n(Callback)', 13, 9),
        ('Audio I/O\n(48kHz, 256samp)', 11.5, 9),
        ('Effect Chain', 13, 7.8),
        ('DSP Effects\n(NumPy/SciPy)', 11.5, 7.8),
        ('6 Effect Types', 12.25, 6.7),
    ]
    for text, x, y in audio_components:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.02",
                            facecolor='white', alpha=0.9,
                            edgecolor=audio_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Shared State Box
    shared_box = FancyBboxPatch((6, 6.5), 4, 4,
                               boxstyle="round,pad=0.1",
                               facecolor=shared_color, alpha=0.3,
                               edgecolor=shared_color, linewidth=2)
    ax.add_patch(shared_box)
    ax.text(8, 10.2, 'SHARED STATE', fontsize=11, fontweight='bold',
           ha='center', color=shared_color)
    
    # Shared state contents
    state_text = """Thread-safe state:
• Current effect index
• Effect depth (0-1)
• Calibration data
• Hand presence flags
• Debug info"""
    ax.text(8, 8.5, state_text, fontsize=8, ha='center', va='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Data/Dataset Box
    data_box = FancyBboxPatch((0.5, 1.5), 5, 4,
                             boxstyle="round,pad=0.1",
                             facecolor=data_color, alpha=0.3,
                             edgecolor=data_color, linewidth=2)
    ax.add_patch(data_box)
    ax.text(3, 5.2, 'HaGRID DATASET', fontsize=11, fontweight='bold',
           ha='center', color=data_color)
    
    data_components = [
        ('6 Gesture Classes', 1.5, 4.5),
        ('MediaPipe Preprocessing', 3, 4.5),
        ('ROI Extraction', 1.5, 3.5),
        ('Augmentation', 3, 3.5),
        ('Train/Val/Test Split', 2.25, 2.5),
    ]
    for text, x, y in data_components:
        box = FancyBboxPatch((x-0.6, y-0.3), 1.2, 0.6,
                            boxstyle="round,pad=0.02",
                            facecolor='white', alpha=0.9,
                            edgecolor=data_color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=8, ha='center', va='center')
    
    # Model Box
    model_box = FancyBboxPatch((6, 1.5), 4, 4,
                              boxstyle="round,pad=0.1",
                              facecolor=model_color, alpha=0.3,
                              edgecolor=model_color, linewidth=2)
    ax.add_patch(model_box)
    ax.text(8, 5.2, 'CNN MODEL', fontsize=11, fontweight='bold',
           ha='center', color=model_color)
    
    model_text = """MobileNetV2 Backbone:
• ~3.5M parameters
• 224x224 input
• 6-class output

Custom Head:
• Dropout layers
• BatchNorm
• ~5ms inference"""
    ax.text(8, 3.2, model_text, fontsize=8, ha='center', va='center',
           family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scripts Box
    scripts_box = FancyBboxPatch((10.5, 1.5), 5, 4,
                                boxstyle="round,pad=0.1",
                                facecolor='#607D8B', alpha=0.3,
                                edgecolor='#607D8B', linewidth=2)
    ax.add_patch(scripts_box)
    ax.text(13, 5.2, 'SCRIPTS', fontsize=11, fontweight='bold',
           ha='center', color='#607D8B')
    
    scripts = [
        ('prepare_hagrid.py', 11.5, 4.5),
        ('train_cnn.py', 13, 4.5),
        ('eval_cnn.py', 11.5, 3.5),
        ('run_realtime.py', 13, 3.5),
    ]
    for text, x, y in scripts:
        box = FancyBboxPatch((x-0.7, y-0.25), 1.4, 0.5,
                            boxstyle="round,pad=0.02",
                            facecolor='white', alpha=0.9,
                            edgecolor='#607D8B', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, fontsize=7, ha='center', va='center', family='monospace')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='gray', lw=1.5)
    
    # Vision to Shared
    ax.annotate('', xy=(6, 8.5), xytext=(5.5, 8.5),
               arrowprops=arrow_style)
    ax.text(5.75, 8.8, 'effect_idx', fontsize=7, ha='center')
    
    # Shared to Audio
    ax.annotate('', xy=(10, 8.5), xytext=(10.5, 8.5),
               arrowprops=arrow_style)
    ax.text(10.25, 8.8, 'effect_idx\ndepth', fontsize=7, ha='center')
    
    # Data to Model
    ax.annotate('', xy=(6, 3.5), xytext=(5.5, 3.5),
               arrowprops=arrow_style)
    
    # Model to Vision
    ax.annotate('', xy=(6, 7), xytext=(6, 5.5),
               arrowprops=arrow_style)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=vision_color, alpha=0.3, edgecolor=vision_color, label='Vision'),
        mpatches.Patch(facecolor=audio_color, alpha=0.3, edgecolor=audio_color, label='Audio'),
        mpatches.Patch(facecolor=shared_color, alpha=0.3, edgecolor=shared_color, label='Shared State'),
        mpatches.Patch(facecolor=data_color, alpha=0.3, edgecolor=data_color, label='Dataset'),
        mpatches.Patch(facecolor=model_color, alpha=0.3, edgecolor=model_color, label='Model'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/okcomputer/output/gesture_audio_effects/outputs/module_diagram.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Module diagram saved to outputs/module_diagram.png")


def create_hand_roles_diagram():
    """Create hand roles diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Hand Role Assignment', fontsize=16, fontweight='bold', ha='center')
    
    # Right Hand (Effect Selection)
    right_box = FancyBboxPatch((1, 2), 5, 5, boxstyle="round,pad=0.1",
                               facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    ax.add_patch(right_box)
    ax.text(3.5, 6.7, 'RIGHT HAND', fontsize=12, fontweight='bold', 
           ha='center', color='#1976D2')
    ax.text(3.5, 6.2, 'Effect Selection', fontsize=10, ha='center', style='italic')
    
    right_content = [
        ('🖐️ Palm → Gain', 2.2),
        ('✊ Fist → Lowpass', 1.8),
        ('👍 Thumb Up → Highpass', 1.4),
        ('👎 Thumb Down → Distortion', 1.0),
        ('👌 OK → Delay', 0.6),
        ('✌️ Peace → Reverb', 0.2),
    ]
    for text, y in right_content:
        ax.text(3.5, y + 2, text, fontsize=9, ha='center', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Left Hand (Depth Control)
    left_box = FancyBboxPatch((8, 2), 5, 5, boxstyle="round,pad=0.1",
                              facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2)
    ax.add_patch(left_box)
    ax.text(10.5, 6.7, 'LEFT HAND', fontsize=12, fontweight='bold',
           ha='center', color='#7B1FA2')
    ax.text(10.5, 6.2, 'Effect Depth Control', fontsize=10, ha='center', style='italic')
    
    # Pinch illustration
    ax.text(10.5, 5.5, 'Pinch Distance:', fontsize=10, ha='center', fontweight='bold')
    ax.text(10.5, 5.0, 'Thumb Tip ↔ Index Tip', fontsize=9, ha='center')
    
    # Distance visualization
    ax.plot([9.5, 11.5], [4.2, 4.2], 'k-', linewidth=2)
    ax.plot(9.5, 4.2, 'ro', markersize=10)
    ax.plot(11.5, 4.2, 'bo', markersize=10)
    ax.text(9.3, 4.2, '👍', fontsize=14, ha='center', va='center')
    ax.text(11.7, 4.2, '☝️', fontsize=14, ha='center', va='center')
    
    # Depth scale
    ax.text(10.5, 3.4, 'Depth Range: 0.0 → 1.0', fontsize=9, ha='center')
    
    depth_labels = [
        ('0.0 (Closed)', 3.0),
        ('0.5 (Mid)', 2.6),
        ('1.0 (Open)', 2.2),
    ]
    for text, y in depth_labels:
        ax.text(10.5, y, text, fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Single hand behavior
    info_box = FancyBboxPatch((3.5, 0.3), 7, 1.2, boxstyle="round,pad=0.05",
                              facecolor='#FFF3E0', edgecolor='#E65100', linewidth=1)
    ax.add_patch(info_box)
    ax.text(7, 1.2, 'Single Hand Behavior:', fontsize=9, fontweight='bold', ha='center')
    ax.text(7, 0.7, 'Right only: Effect selection works, depth frozen | Left only: Depth control works, effect unchanged',
           fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('/mnt/okcomputer/output/gesture_audio_effects/outputs/hand_roles_diagram.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("Hand roles diagram saved to outputs/hand_roles_diagram.png")


def create_dsp_effects_diagram():
    """Create DSP effects mapping diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'DSP Effects Depth Mapping', fontsize=16, fontweight='bold', ha='center')
    
    effects = [
        ('Gain', '0.0: 0dB', '1.0: +20dB', '#4CAF50'),
        ('Lowpass', '0.0: 500Hz', '1.0: 8kHz', '#FF9800'),
        ('Highpass', '0.0: 50Hz', '1.0: 4kHz', '#03A9F4'),
        ('Distortion', '0.0: Mild', '1.0: Heavy', '#F44336'),
        ('Delay', '0.0: 100ms/0.1fb', '1.0: 500ms/0.8fb', '#E91E63'),
        ('Reverb', '0.0: 0% wet', '1.0: 60% wet', '#9C27B0'),
    ]
    
    y_start = 8.5
    for i, (name, min_val, max_val, color) in enumerate(effects):
        y = y_start - i * 1.3
        
        # Effect box
        box = FancyBboxPatch((0.5, y-0.4), 13, 1, boxstyle="round,pad=0.05",
                            facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Effect name
        ax.text(2, y, name, fontsize=11, fontweight='bold', ha='center', va='center')
        
        # Min value
        ax.text(5, y, min_val, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Arrow
        ax.annotate('', xy=(8.5, y), xytext=(6.5, y),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
        
        # Depth bar
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        ax.imshow(gradient, extent=[8.5, 11, y-0.15, y+0.15], aspect='auto',
                 cmap='RdYlGn', alpha=0.7)
        ax.plot([8.5, 11], [y-0.15, y-0.15], 'k-', linewidth=1)
        ax.plot([8.5, 11], [y+0.15, y+0.15], 'k-', linewidth=1)
        ax.plot([8.5, 8.5], [y-0.15, y+0.15], 'k-', linewidth=1)
        ax.plot([11, 11], [y-0.15, y+0.15], 'k-', linewidth=1)
        
        # Max value
        ax.text(12, y, max_val, fontsize=9, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/mnt/okcomputer/output/gesture_audio_effects/outputs/dsp_effects_diagram.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("DSP effects diagram saved to outputs/dsp_effects_diagram.png")


def main():
    """Generate all architecture diagrams."""
    import os
    os.makedirs('/mnt/okcomputer/output/gesture_audio_effects/outputs', exist_ok=True)
    
    print("Generating architecture diagrams...")
    create_module_diagram()
    create_hand_roles_diagram()
    create_dsp_effects_diagram()
    print("\nAll diagrams generated successfully!")


if __name__ == '__main__':
    main()
