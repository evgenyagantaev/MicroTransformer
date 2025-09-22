"""
Пример кода: Преобразование токенов в эмбеддинги

Этот файл содержит код, который показывает, как токены преобразуются
в векторные представления с помощью Embedding Layer.
"""

import torch
import torch.nn as nn

# Пример создания Embedding слоя
vocab_size = 240  # Размер словаря (как в MicroTransformer)
embedding_dim = 5  # Размерность эмбеддингов (d_model)

# Создание Embedding слоя
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# Пример токенов (ID из словаря)
tokens = torch.tensor([1, 12, 113, 2], dtype=torch.long)  # [START, амба, ходи, END]
print("Входные токены:", tokens)
print("Форма токенов:", tokens.shape)

# Преобразование токенов в эмбеддинги
embeddings = embedding_layer(tokens)
print("Эмбеддинги:")
print(embeddings)
print("Форма эмбеддингов:", embeddings.shape)

# Масштабирование (как в трансформерах)
import math
scaled_embeddings = embeddings * math.sqrt(embedding_dim)
print("\nМасштабированные эмбеддинги:")
print(scaled_embeddings)

# Показать эмбеддинг для конкретного токена
print("\nЭмбеддинг для токена 'амба' (ID=12):")
print(embedding_layer.weight[12])

# Показать как получить эмбеддинг для одного токена
single_token = torch.tensor([12], dtype=torch.long)
single_embedding = embedding_layer(single_token)
print("\nЭмбеддинг для одного токена:")
print(single_embedding)

# Батч эмбеддингов
batch_tokens = torch.tensor([[1, 12], [113, 2]], dtype=torch.long)
batch_embeddings = embedding_layer(batch_tokens)
print("\nБатч токенов:")
print(batch_tokens)
print("Батч эмбеддингов:")
print(batch_embeddings)
print("Форма батча:", batch_embeddings.shape)
