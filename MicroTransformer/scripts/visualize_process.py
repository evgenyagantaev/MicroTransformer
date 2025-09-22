"""
Visualization script for the MicroTransformer process.

This script demonstrates the step-by-step process of text generation
from input to output with detailed tensor shapes and intermediate results.
"""

import sys
import os
import torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import tokenizer
from models.model import create_model

def visualize_tokenization():
    """Visualize the tokenization process."""
    print("=== 1. ТОКЕНИЗАЦИЯ ===")
    input_text = "амба ходи"

    print(f"Входной текст: '{input_text}'")

    # Токенизация
    tokens = tokenizer.tokenize(input_text)
    print(f"Токены: {tokens}")

    # Кодирование с специальными токенами
    encoded = tokenizer.encode(input_text, add_special_tokens=True)
    print(f"Закодированная последовательность: {encoded}")
    print(f"Форма тензора: {encoded.shape}")
    print()

def visualize_embeddings(model):
    """Visualize the embedding process."""
    print("=== 2. ЭМБЕДДИНГИ ===")

    # Простой пример
    test_tokens = torch.tensor([[1, 12, 113, 2, 0]], dtype=torch.long)  # [START, амба, ходи, END, PAD]

    print(f"Входные токены: {test_tokens}")
    print(f"Форма: {test_tokens.shape}")

    # Получаем эмбеддинги
    embeddings = model.embedding(test_tokens)
    print(f"Эмбеддинги форма: {embeddings.shape}")
    print(f"Эмбеддинг для 'амба' (токен 12): {embeddings[0, 1]}")
    print()

def visualize_positional_encoding(model):
    """Visualize positional encoding."""
    print("=== 3. ПОЗИЦИОННОЕ КОДИРОВАНИЕ ===")

    # Создаем тестовый батч
    batch_size, seq_len = 2, 5
    test_tokens = torch.randint(1, 10, (batch_size, seq_len))

    embeddings = model.embedding(test_tokens)
    pos_encoded = model.pos_encoding(embeddings)

    print(f"Токены: {test_tokens}")
    print(f"Эмбеддинги + позиционное кодирование форма: {pos_encoded.shape}")
    print(f"Разница (первые 3 позиции): {pos_encoded[0, :3, 0] - embeddings[0, :3, 0]}")
    print()

def visualize_attention(model):
    """Visualize attention mechanism."""
    print("=== 4. МЕХАНИЗМ ВНИМАНИЯ ===")

    # Создаем тестовый батч
    batch_size, seq_len = 1, 4
    x = torch.randn(batch_size, seq_len, model.d_model)

    print(f"Вход трансформеру: {x.shape}")

    # Пропускаем через первый слой
    attention_output = model.layers[0].self_attn(x, x, x)[0]

    print(f"Выход внимания: {attention_output.shape}")
    print("Внимание помогает модели фокусироваться на разных частях последовательности")
    print()

def visualize_generation(model):
    """Visualize the generation process."""
    print("=== 5. ГЕНЕРАЦИЯ ТЕКСТА ===")

    start_token = tokenizer.vocabulary['<start>']
    print(f"Начинаем с токена: {start_token} ('<start>')")

    generated = [start_token]
    current_seq = torch.tensor([generated], dtype=torch.long)

    print("\nШаг генерации:")
    for step in range(5):
        # Получаем выход модели
        logits = model(current_seq)[:, -1, :]  # Последний токен
        probs = F.softmax(logits, dim=-1)

        # Семплируем следующий токен
        next_token = torch.multinomial(probs, 1).item()

        print(f"Шаг {step+1}:")
        print(f"  Текущая последовательность: {current_seq.tolist()}")
        print(f"  Логиты форма: {logits.shape}")
        print(f"  Топ-3 вероятности: {torch.topk(probs, 3)[1].tolist()}")
        print(f"  Выбран токен: {next_token} ({tokenizer.id_to_token.get(next_token, '<unk>')})")

        if next_token == tokenizer.vocabulary['<end>']:
            break

        generated.append(next_token)
        current_seq = torch.cat([current_seq, torch.tensor([[next_token]])], dim=1)

    # Декодируем результат
    final_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"\nИтоговый текст: '{final_text}'")
    print()

def visualize_model_architecture(model):
    """Visualize model architecture."""
    print("=== АРХИТЕКТУРА МОДЕЛИ ===")

    print(f"Общее количество параметров: {sum(p.numel() for p in model.parameters())}")

    print("\nСлои модели:")
    for i, layer in enumerate(model.layers):
        print(f"  Трансформерный слой {i+1}:")
        print(f"    - Self-Attention: {layer.self_attn.num_heads} голов")
        print(f"    - Feed-Forward: {layer.feed_forward[0].out_features} нейронов")
        print()

    print("Размеры:")
    print(f"  - Embedding: {model.embedding.weight.shape}")
    print(f"  - Выходной слой: {model.fc.weight.shape}")
    print()

def main():
    """Main visualization function."""
    print("🎯 ВИЗУАЛИЗАЦИЯ ПРОЦЕССА MicroTransformer")
    print("=" * 50)

    # Создаем модель
    model = create_model()

    # Визуализируем этапы
    visualize_tokenization()
    visualize_embeddings(model)
    visualize_positional_encoding(model)
    visualize_attention(model)
    visualize_generation(model)
    visualize_model_architecture(model)

    print("✅ Визуализация завершена!")

if __name__ == "__main__":
    main()
