# MicroTransformer Project

## Overview

This project implements a complete Transformer-based language model for the artificial language "Dersu Uzala" (Язык Дерсу Узала). The model is designed as a micro-transformer with an embedding dimension of 5, making it suitable for educational purposes and understanding the fundamentals of Transformer architectures.

## Features

- **Complete Pipeline**: From data generation to model training and inference
- **Custom Language**: Based on the Dersu Uzala language specification
- **Small Model**: Embedding dimension of 5 for lightweight implementation
- **Full Training Loop**: Complete training pipeline with validation
- **Text Generation**: Autoregressive text generation capabilities

## Project Structure

```
MicroTransformer/
├── src/
│   ├── __init__.py
│   ├── language_spec.py    # Language specification and vocabulary
│   ├── tokenizer.py        # Text tokenization
│   └── dataset.py          # Dataset generation and loading
├── models/
│   ├── __init__.py
│   └── model.py            # Transformer model implementation
├── scripts/
│   └── train.py            # Training script
├── data/                   # Generated datasets
├── __init__.py
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Language Specification

The model is trained on the "Dersu Uzala" language, which includes:

- **240 tokens** across 8 categories:
  - Special tokens: `<start>`, `<end>`, `<pad>`, `<unk>`
  - Subjects: Living beings and natural entities (60 tokens)
  - Objects: Physical items (50 tokens)
  - Actions: Verbs in infinitive form (60 tokens)
  - Qualities: Adjectives and adverbs (40 tokens)
  - Pronouns: Personal pronouns (8 tokens)
  - Conjunctions and Particles (8 tokens)
  - Questions and Answers (6 tokens)

## Grammar Rules

The language follows strict but simple grammar rules:

1. **Simple sentence**: `[Subject] + [Action]`
2. **With quality**: `[Subject] + [Quality] + [Action]`
3. **With object**: `[Subject] + [Object] + [Action]`
4. **Modal constructions**: `надо + [Subject] + [Action]`
5. **Possessive**: `[Owner] + [Object] + его`
6. **Negative**: `[Subject] + [Object/Quality] + нету`
7. **Questions**: `[Question] + [Subject] + [Action]`
8. **Complex sentences**: Connected with conjunctions

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MicroTransformer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the training script:
```bash
python scripts/train.py --epochs 50 --batch_size 32 --train_size 10000
```

## Usage

### Training

To train the model with custom parameters:

```bash
python scripts/train.py \
    --batch_size 32 \
    --lr 0.001 \
    --epochs 50 \
    --train_size 10000 \
    --val_size 1000 \
    --device cpu \
    --save_path models/microtransformer.pth
```

### Inference

To generate text with the trained model:

```python
from models.model import create_model
from src.tokenizer import tokenizer

# Load model
model = create_model()
model.load_model('models/microtransformer.pth')

# Generate text
start_token = tokenizer.vocabulary['<start>']
generated = model.generate(start_token, max_length=20, temperature=0.8)
text = tokenizer.decode(generated, skip_special_tokens=True)
print(text)
```

### Dataset Generation

To generate sample sentences:

```python
from src.dataset import generate_sample_sentences

samples = generate_sample_sentences(5)
for sample in samples:
    print(sample)
```

## Model Architecture

- **Embedding Dimension**: 5 (as required)
- **Attention Heads**: 1
- **Transformer Layers**: 2
- **Feed-Forward Dimension**: 20
- **Vocabulary Size**: 240
- **Maximum Sequence Length**: 20

## Training Details

- **Loss Function**: Cross-Entropy Loss with padding ignore
- **Optimizer**: Adam with learning rate 0.001
- **Scheduler**: StepLR with step size 10 and gamma 0.1
- **Batch Size**: 32 (configurable)

## Process Flow Diagram

```
ВХОДНОЙ ТЕКСТ → ТОКЕНИЗАЦИЯ → ЭМБЕДДИНГИ → ПОЗИЦИОННОЕ КОДИРОВАНИЕ → ТРАНСФОРМЕР → ВЫХОДНЫЕ ЛОГИТЫ → ДЕКОДИРОВАНИЕ → ВЫХОДНОЙ ТЕКСТ
```

### Detailed Process Flow:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ВХОДНОЙ ТЕКСТ  │ -> │   ТОКЕНИЗАТОР    │ -> │   ЭМБЕДДИНГИ    │ -> │  ПОЗИЦИОННОЕ    │
│                 │    │                 │    │   (d_model=5)   │    │   КОДИРОВАНИЕ   │
│ "амба ходи"     │    │ амба=12,        │    │ [emb_амба]      │    │ [emb + pos]     │
│                 │    │ ходи=113        │    │ [emb_ходи]      │    │ [emb + pos]     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ТРАНСФОРМЕРНЫЕ    │ -> │   ВНИМАНИЕ      │ -> │  FEED-FORWARD   │ -> │ ВЫХОДНЫЕ ЛОГИТЫ │
│    СЛОИ (2)     │    │  (Multi-Head)   │    │   NETWORK       │    │                 │
│                 │    │                 │    │                 │    │ [logit_1]       │
│ Self-Attention  │    │ Query/Key/Value │    │ Linear → ReLU   │    │ [logit_2]       │
│ Layer Norm      │    │                 │    │ Linear          │    │ ...             │
│ Feed-Forward    │    │                 │    │                 │    │ [logit_240]     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

         │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   АВТОРЕГРЕСС.   │ -> │   СЕМПЛИНГ      │ -> │   ДЕТОКЕНИЗАТОР  │ -> │  ВЫХОДНОЙ ТЕКСТ  │
│   ГЕНЕРАЦИЯ     │    │   (Top-k)       │    │                 │    │                 │
│                 │    │                 │    │ 12 → "амба"     │    │ "амба ходи"     │
│ Итеративно      │    │ Температура 0.7 │    │ 113 → "ходи"    │    │                 │
│ Добавляем       │    │                 │    │ ...             │    │                 │
│ Токены          │    │                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Process Description:

1. **Input Text**: Исходный текст на языке Дерсу Узала
2. **Tokenization**: Разбиение текста на токены с использованием словаря (240 токенов)
3. **Embeddings**: Преобразование токенов в векторы размерности 5
4. **Positional Encoding**: Добавление позиционной информации к эмбеддингам
5. **Transformer Layers** (2 слоя):
   - Multi-Head Self-Attention (1 голова)
   - Layer Normalization
   - Feed-Forward Network (20 → 5)
6. **Output Logits**: Получение вероятностей для каждого токена словаря
7. **Autoregressive Generation**: Итеративное добавление токенов
8. **Sampling**: Выбор следующего токена с использованием температуры
9. **Detokenization**: Преобразование токенов обратно в текст

### Key Parameters:
- **Embedding Dimension**: 5
- **Attention Heads**: 1
- **Transformer Layers**: 2
- **Feed-Forward Dimension**: 20
- **Vocabulary Size**: 240
- **Maximum Sequence Length**: 20
- **Temperature**: 0.7 (для семплинга)

## Example Output

Sample generated sentences:
- "амба ходи"
- "моя тихо смотри"
- "капитан ружье его"
- "надо наша работай"
- "солнце уходи и ветер приходи"

## Contributing

This is an educational project demonstrating a complete Transformer implementation. Feel free to experiment with different hyperparameters or extend the language specification.

## License

This project is for educational purposes only.
