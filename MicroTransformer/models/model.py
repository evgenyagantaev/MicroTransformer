"""
Transformer model for the Dersu Uzala language.

This module implements a simple Transformer architecture for language modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import tokenizer

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Create div_term for all positions
        half_dim = (d_model + 1) // 2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sin for even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Cos for odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class MicroTransformer(nn.Module):
    """Simple Transformer model for language modeling."""

    def __init__(self, vocab_size, d_model=5, n_heads=1, n_layers=2, d_ff=20, max_len=50, dropout=0.1):
        """
        Initialize the Transformer model.

        Args:
            vocab_size (int): Size of vocabulary
            d_model (int): Embedding dimension (must be 5 as per requirements)
            n_heads (int): Number of attention heads
            n_layers (int): Number of transformer layers
            d_ff (int): Feed-forward dimension
            max_len (int): Maximum sequence length
            dropout (float): Dropout rate
        """
        super().__init__()

        assert d_model == 5, "Embedding dimension must be 5 as per requirements"

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output layer
        self.fc = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def generate_square_subsequent_mask(self, sz):
        """Generate mask for causal attention."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            mask (torch.Tensor, optional): Attention mask

        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()

        # Create causal mask
        if mask is None:
            mask = self.generate_square_subsequent_mask(seq_len).to(x.device)

        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        output = self.fc(x)

        return output

    def generate(self, start_token, max_length=50, temperature=1.0):
        """
        Generate text autoregressively.

        Args:
            start_token (int): Starting token ID
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature

        Returns:
            list: Generated token IDs
        """
        self.eval()
        with torch.no_grad():
            # Start with start token
            generated = [start_token]
            current_seq = torch.tensor([generated], dtype=torch.long)

            for _ in range(max_length - 1):
                # Get model output
                logits = self(current_seq)[:, -1, :]  # Get last token's logits
                logits = logits / temperature

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()

                # Stop if end token is generated
                if next_token == tokenizer.vocabulary['<end>']:
                    break

                generated.append(next_token)
                current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)

        return generated

    def save_model(self, path):
        """Save model to file."""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """Load model from file."""
        self.load_state_dict(torch.load(path))

def create_model():
    """Create and return the model."""
    vocab_size = tokenizer.vocab_size
    model = MicroTransformer(
        vocab_size=vocab_size,
        d_model=5,
        n_heads=1,
        n_layers=2,
        d_ff=20,
        max_len=tokenizer.max_length
    )
    return model

if __name__ == "__main__":
    # Test the model
    print("Testing MicroTransformer model...")

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
    generated = model.generate(start_token, max_length=20)
    generated_text = tokenizer.decode(generated)
    print(f"Generated: {generated_text}")
