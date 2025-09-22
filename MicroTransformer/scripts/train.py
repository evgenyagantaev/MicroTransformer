"""
Training script for the MicroTransformer model.

This script handles the complete training pipeline for the Transformer model
on the Dersu Uzala language dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import json

# Import project modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import DersuUzalaDataset, create_data_loader
from src.tokenizer import tokenizer
from models.model import create_model, MicroTransformer

class Trainer:
    """Training class for the MicroTransformer model."""

    def __init__(self, model, train_loader, val_loader=None, lr=0.001, device='cpu'):
        """
        Initialize the trainer.

        Args:
            model: MicroTransformer instance
            train_loader: Training data loader
            val_loader: Validation data loader
            lr: Learning rate
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr

        # Move model to device
        self.model.to(device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocabulary['<pad>'])

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': []
        }

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Reshape for loss calculation
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self):
        """Validate the model."""
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, num_epochs=50, save_path=None, log_interval=10):
        """
        Train the model.

        Args:
            num_epochs (int): Number of epochs
            save_path (str): Path to save the model
            log_interval (int): Logging interval
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log
            if epoch % log_interval == 0:
                log_str = f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.3f}"
                if val_loss is not None:
                    log_str += f", Val Loss: {val_loss:.3f}"
                print(log_str)

                # Generate sample
                self.generate_sample()

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['epochs'].append(epoch + 1)

        # Save model
        if save_path:
            self.save_model(save_path)

        # Save training history
        self.save_history()

    def generate_sample(self):
        """Generate a sample text."""
        start_token = tokenizer.vocabulary['<start>']
        generated = self.model.generate(start_token, max_length=20, temperature=0.8)
        generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"Sample generation: {generated_text}")

    def save_model(self, path):
        """Save the model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to {path}")

    def save_history(self):
        """Save training history."""
        history_path = "MicroTransformer/models/training_history.json"
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MicroTransformer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train_size', type=int, default=10000, help='Training set size')
    parser.add_argument('--val_size', type=int, default=1000, help='Validation set size')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--save_path', type=str, default='MicroTransformer/models/microtransformer.pth', help='Model save path')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    # Create datasets
    print("Creating datasets...")
    train_dataset = DersuUzalaDataset(num_samples=args.train_size, split='train')
    val_dataset = DersuUzalaDataset(num_samples=args.val_size, split='val')

    # Create data loaders
    train_loader = create_data_loader(train_dataset, batch_size=args.batch_size)
    val_loader = create_data_loader(val_dataset, batch_size=args.batch_size)

    # Create model
    print("Creating model...")
    model = create_model()

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, lr=args.lr, device=args.device)

    # Train
    print("Starting training...")
    trainer.train(num_epochs=args.epochs, save_path=args.save_path)

    print("Training completed!")

if __name__ == "__main__":
    main()
