"""
Prepare HaGRID dataset for training.

Downloads and preprocesses the HaGRID dataset for gesture classification.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hagrid.dataset import (
    download_hagrid_sample, 
    prepare_hagrid_dataset,
    GESTURE_TO_EFFECT,
    CLASS_NAMES
)


def main():
    parser = argparse.ArgumentParser(description='Prepare HaGRID dataset')
    
    parser.add_argument('--input_dir', type=str, default='./hagrid_data',
                       help='Input directory with raw HaGRID data')
    parser.add_argument('--output_dir', type=str, default='./hagrid_processed',
                       help='Output directory for processed data')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample dataset structure only')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Minimum hand detection confidence')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HaGRID DATASET PREPARATION")
    print("=" * 60)
    
    print("\nGesture to Effect Mapping:")
    print("-" * 40)
    for gesture, effect in GESTURE_TO_EFFECT.items():
        print(f"  {gesture:12} -> {effect}")
    
    if args.create_sample:
        # Create sample structure only
        download_hagrid_sample(args.input_dir)
    else:
        # Full preprocessing
        print(f"\nInput directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Min confidence: {args.min_confidence}")
        
        # Check if input exists
        input_path = Path(args.input_dir)
        if not input_path.exists():
            print(f"\nError: Input directory '{args.input_dir}' does not exist!")
            print("Run with --create_sample first to create the structure.")
            print("Then download HaGRID data from:")
            print("  https://github.com/hukenovs/hagrid")
            return
        
        # Process dataset
        stats = prepare_hagrid_dataset(
            args.input_dir,
            args.output_dir,
            args.min_confidence
        )
        
        print("\n" + "=" * 60)
        print("Preparation Complete!")
        print("=" * 60)
        print(f"\nNext steps:")
        print(f"  1. Train model: python scripts/train_cnn.py --data_dir {args.output_dir}")
        print(f"  2. Evaluate: python scripts/eval_cnn.py --model_path outputs/training/best_model.pth")


if __name__ == '__main__':
    main()
