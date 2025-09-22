"""
Демонстрация преобразования токенов в эмбеддинги.

Этот скрипт показывает, как токены преобразуются в векторные представления
с помощью Embedding Layer в MicroTransformer.
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import tokenizer
from models.model import create_model

def demonstrate_embedding_layer():
    """Демонстрация Embedding Layer."""
    print("=== EMBEDDING LAYER ДЕМОНСТРАЦИЯ ===")
    print("=" * 50)

    # Создаем модель
    model = create_model()

    # Пример входных токенов
    example_text = "амба ходи"
    print(f"Исходный текст: '{example_text}'")

    # Токенизация
    tokens = tokenizer.tokenize(example_text)
    print(f"Токены: {tokens}")
    print(f"Соответствующие слова: {[tokenizer.id_to_token.get(t, '<unk>') for t in tokens]}")

    # Создаем батч токенов
    token_tensor = torch.tensor([tokens], dtype=torch.long)
    print(f"Форма токенов: {token_tensor.shape}")  # (batch_size=1, seq_len=2)

    # Применяем Embedding Layer
    embeddings = model.embedding(token_tensor)
    print(f"Форма эмбеддингов: {embeddings.shape}")  # (1, 2, 5)

    print("\nЭмбеддинги для каждого токена:")
    for i, token_id in enumerate(tokens):
        word = tokenizer.id_to_token.get(token_id, '<unk>')
        embedding = embeddings[0, i]
        print(f"  '{word}' (ID={token_id}): {embedding}")

    scaled = embeddings * 5**0.5
    print(f"\nМасштабирование: embeddings * sqrt(d_model) = {scaled}")

    return token_tensor, embeddings

def demonstrate_positional_encoding(model, token_tensor, embeddings):
    """Демонстрация позиционного кодирования."""
    print("\n=== POSITIONAL ENCODING ===")
    print("=" * 30)

    # Позиционное кодирование
    pos_encoded = model.pos_encoding(embeddings)
    print(f"Форма после позиционного кодирования: {pos_encoded.shape}")

    print("\nДобавленные позиционные значения (первые 3 позиции):")
    for pos in range(3):
        if pos < pos_encoded.shape[1]:
            pos_embedding = pos_encoded[0, pos]
            print(f"  Позиция {pos}: {pos_embedding}")

    return pos_encoded

def demonstrate_full_forward(model, token_tensor):
    """Демонстрация полного forward pass через модель."""
    print("\n=== ПОЛНЫЙ FORWARD PASS ===")
    print("=" * 30)

    print(f"Входные токены: {token_tensor}")
    print(f"Форма: {token_tensor.shape}")

    # Полный forward pass
    output = model(token_tensor)
    print(f"Выходная форма: {output.shape}")  # (batch_size, seq_len, vocab_size)

    print("\nЛогиты для первого токена 'амба':")
    first_token_logits = output[0, 0]  # Логиты для первого токена в последовательности
    top_5_logits, top_5_indices = torch.topk(first_token_logits, 5)

    for i in range(5):
        token_id = top_5_indices[i].item()
        word = tokenizer.id_to_token.get(token_id, '<unk>')
        logit = top_5_logits[i].item()
        print(f"  {word} (ID={token_id}): {logit:.4f}")

def demonstrate_embedding_weights():
    """Демонстрация весов Embedding слоя."""
    print("\n=== ВЕСА EMBEDDING СЛОЯ ===")
    print("=" * 30)

    model = create_model()
    embedding_weights = model.embedding.weight

    print(f"Форма весов: {embedding_weights.shape}")  # (vocab_size, d_model)
    min_val = embedding_weights.min().item()
    max_val = embedding_weights.max().item()
    print(f"Диапазон весов: [{min_val:.4f}, {max_val:.4f}]")

    # Показать эмбеддинги для нескольких токенов
    example_tokens = [1, 12, 113, 2]  # START, амба, ходи, END

    print("\nЭмбеддинги для специальных токенов:")
    for token_id in example_tokens:
        if token_id < embedding_weights.shape[0]:
            embedding = embedding_weights[token_id]
            word = tokenizer.id_to_token.get(token_id, '<unk>')
            print(f"  '{word}' (ID={token_id}): {embedding}")

def main():
    """Основная функция демонстрации."""
    print("🎯 ДЕМОНСТРАЦИЯ ПРЕОБРАЗОВАНИЯ ТОКЕНОВ В ЭМБЕДДИНГИ")
    print("=" * 60)

    # Создаем модель
    model = create_model()

    # Демонстрация шаг за шагом
    token_tensor, embeddings = demonstrate_embedding_layer()
    pos_encoded = demonstrate_positional_encoding(model, token_tensor, embeddings)
    demonstrate_full_forward(model, token_tensor)
    demonstrate_embedding_weights()

    print("\n✅ Демонстрация завершена!")
    print("\nКлючевые моменты:")
    print("- Токены → Целые числа (0-239)")
    print("- Embedding Layer → Векторы размерности 5")
    print("- Масштабирование: * sqrt(d_model)")
    print("- Позиционное кодирование добавляет позиционную информацию")

if __name__ == "__main__":
    main()
