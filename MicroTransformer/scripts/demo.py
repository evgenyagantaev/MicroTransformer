"""
Demo script for the MicroTransformer model.

This script loads a trained model and demonstrates text generation.
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import tokenizer
from models.model import create_model

def main():
    """Main demo function."""
    print("=== MicroTransformer Demo ===\n")

    # Load model
    print("Loading trained model...")
    model = create_model()
    model_path = "MicroTransformer/models/microtransformer.pth"

    if os.path.exists(model_path):
        model.load_model(model_path)
        print("✓ Model loaded successfully!")
    else:
        print("✗ No trained model found. Please run training first.")
        return

    # Generate multiple samples
    print("\nGenerating text samples:")
    print("-" * 40)

    start_token = tokenizer.vocabulary['<start>']

    for i in range(5):
        generated = model.generate(start_token, max_length=15, temperature=0.7)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"{i+1}. {text}")

    print("\nDemo completed!")

if __name__ == "__main__":
    main()
