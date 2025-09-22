"""
Test script for the MicroTransformer project.

This script tests all components of the project: tokenizer, dataset, model, and training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import tokenizer
from src.dataset import generate_sample_sentences, DersuUzalaDataset, create_data_loader
from models.model import create_model
import torch

def test_tokenizer():
    """Test the tokenizer functionality."""
    print("Testing Tokenizer...")

    # Test encoding/decoding
    test_text = "амба ходи"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Test batch processing
    batch_texts = ["амба ходи", "моя тихо смотри"]
    batch_encoded = tokenizer.batch_encode(batch_texts)
    batch_decoded = tokenizer.batch_decode(batch_encoded)

    print(f"Batch input: {batch_texts}")
    print(f"Batch decoded: {batch_decoded}")

def test_dataset():
    """Test dataset generation."""
    print("\nTesting Dataset...")

    # Generate sample sentences
    samples = generate_sample_sentences(3)
    print("Sample sentences:")
    for sample in samples:
        print(f"  {sample}")

    # Create small dataset
    dataset = DersuUzalaDataset(num_samples=10)
    print(f"Dataset size: {len(dataset)}")

    # Test data loading
    loader = create_data_loader(dataset, batch_size=2)
    for inputs, targets in loader:
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Sample input: {tokenizer.decode(inputs[0])}")
        print(f"Sample target: {tokenizer.decode(targets[0])}")
        break

def test_model():
    """Test model functionality."""
    print("\nTesting Model...")

    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Test forward pass
    batch_size, seq_len = 2, 10
    test_input = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len))
    output = model(test_input)

    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test generation
    start_token = tokenizer.vocabulary['<start>']
    generated = model.generate(start_token, max_length=15, temperature=0.8)
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"Generated text: {generated_text}")

def main():
    """Run all tests."""
    print("=== MicroTransformer Test Suite ===\n")

    test_tokenizer()
    test_dataset()
    test_model()

    print("\n=== All tests completed successfully! ===")

if __name__ == "__main__":
    main()
