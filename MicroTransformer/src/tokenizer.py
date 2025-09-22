"""
Tokenizer for the Dersu Uzala language.

This module provides functionality to tokenize text into token IDs and vice versa
based on the vocabulary defined in language_spec.py.
"""

import torch
from .language_spec import VOCABULARY, ID_TO_TOKEN, MAX_LENGTH

class DersuUzalaTokenizer:
    """Tokenizer for the Dersu Uzala language."""

    def __init__(self):
        self.vocabulary = VOCABULARY
        self.id_to_token = ID_TO_TOKEN
        self.vocab_size = len(VOCABULARY)
        self.max_length = MAX_LENGTH

    def tokenize(self, text):
        """
        Tokenize a text string into token IDs.

        Args:
            text (str): Input text

        Returns:
            list: List of token IDs
        """
        tokens = text.strip().split()
        token_ids = []

        for token in tokens:
            if token in self.vocabulary:
                token_ids.append(self.vocabulary[token])
            else:
                token_ids.append(self.vocabulary['<unk>'])

        return token_ids

    def detokenize(self, token_ids):
        """
        Convert token IDs back to text.

        Args:
            token_ids (list or torch.Tensor): List of token IDs

        Returns:
            str: Detokenized text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append('<unk>')

        return ' '.join(tokens)

    def encode(self, text, add_special_tokens=True):
        """
        Encode text to token IDs with optional special tokens.

        Args:
            text (str): Input text
            add_special_tokens (bool): Whether to add <start> and <end> tokens

        Returns:
            torch.Tensor: Encoded token IDs
        """
        token_ids = self.tokenize(text)

        if add_special_tokens:
            token_ids = [self.vocabulary['<start>']] + token_ids + [self.vocabulary['<end>']]

        # Pad to max_length
        if len(token_ids) < self.max_length:
            token_ids += [self.vocabulary['<pad>']] * (self.max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.max_length]

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs back to text.

        Args:
            token_ids (torch.Tensor or list): Token IDs
            skip_special_tokens (bool): Whether to skip special tokens

        Returns:
            str: Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<unk>')
            if skip_special_tokens and token in ['<start>', '<end>', '<pad>']:
                continue
            tokens.append(token)

        return ' '.join(tokens)

    def batch_encode(self, texts, add_special_tokens=True):
        """
        Encode a batch of texts.

        Args:
            texts (list): List of text strings
            add_special_tokens (bool): Whether to add special tokens

        Returns:
            torch.Tensor: Batch of encoded token IDs
        """
        encoded_batch = []
        for text in texts:
            encoded = self.encode(text, add_special_tokens)
            encoded_batch.append(encoded)

        return torch.stack(encoded_batch)

    def batch_decode(self, token_ids_batch, skip_special_tokens=True):
        """
        Decode a batch of token IDs.

        Args:
            token_ids_batch (torch.Tensor): Batch of token IDs
            skip_special_tokens (bool): Whether to skip special tokens

        Returns:
            list: List of decoded texts
        """
        decoded_batch = []
        for token_ids in token_ids_batch:
            decoded = self.decode(token_ids, skip_special_tokens)
            decoded_batch.append(decoded)

        return decoded_batch

# Global tokenizer instance
tokenizer = DersuUzalaTokenizer()
