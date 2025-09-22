"""
Dataset generator for the Dersu Uzala language.

This module provides functionality to generate training data for the Transformer model.
"""

import torch
from torch.utils.data import Dataset
from .language_spec import generate_sentence, MAX_LENGTH
from .tokenizer import tokenizer
import random

class DersuUzalaDataset(Dataset):
    """Dataset for training the Transformer model on Dersu Uzala language."""

    def __init__(self, num_samples=10000, max_length=MAX_LENGTH, split='train'):
        """
        Initialize the dataset.

        Args:
            num_samples (int): Number of samples to generate
            max_length (int): Maximum sequence length
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.num_samples = num_samples
        self.max_length = max_length
        self.split = split
        self.data = []

        # Generate data
        self._generate_data()

    def _generate_data(self):
        """Generate training data."""
        for _ in range(self.num_samples):
            # Generate input sequence
            input_tokens = generate_sentence(self.max_length)

            # For language modeling, input is the sequence and target is shifted by one
            # Add <start> and <end> tokens
            input_tokens = [tokenizer.vocabulary['<start>']] + input_tokens + [tokenizer.vocabulary['<end>']]

            # Pad to max_length
            if len(input_tokens) < self.max_length:
                input_tokens += [tokenizer.vocabulary['<pad>']] * (self.max_length - len(input_tokens))
            else:
                input_tokens = input_tokens[:self.max_length]

            # Target is the same as input but shifted by one position
            target_tokens = input_tokens[1:] + [tokenizer.vocabulary['<pad>']]

            # Convert to tensors
            input_tensor = torch.tensor(input_tokens, dtype=torch.long)
            target_tensor = torch.tensor(target_tokens, dtype=torch.long)

            self.data.append((input_tensor, target_tensor))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_data_loader(dataset, batch_size=32, shuffle=True):
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: DersuUzalaDataset instance
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def collate_fn(batch):
    """
    Collate function for DataLoader.

    Args:
        batch: List of (input, target) tuples

    Returns:
        tuple: Batched inputs and targets
    """
    inputs, targets = zip(*batch)
    inputs = torch.stack(inputs)
    targets = torch.stack(targets)
    return inputs, targets

def generate_sample_sentences(num_sentences=10):
    """Generate sample sentences for testing."""
    sentences = []
    for _ in range(num_sentences):
        tokens = generate_sentence(MAX_LENGTH)
        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        sentences.append(sentence)
    return sentences

if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset generation...")

    # Generate sample sentences
    samples = generate_sample_sentences(5)
    print("Sample sentences:")
    for i, sample in enumerate(samples):
        print(f"{i+1}: {sample}")

    # Create small dataset
    test_dataset = DersuUzalaDataset(num_samples=100)
    print(f"\nDataset size: {len(test_dataset)}")

    # Test data loader
    test_loader = create_data_loader(test_dataset, batch_size=10)
    for inputs, targets in test_loader:
        print(f"\nInput shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Sample input: {tokenizer.decode(inputs[0])}")
        print(f"Sample target: {tokenizer.decode(targets[0])}")
        break
