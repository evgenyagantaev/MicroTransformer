"""
Language specification for the Dersu Uzala language (Язык Дерсу Узала).

This module defines the vocabulary and grammar rules for the Dersu Uzala language
used to generate training data for the Transformer model.

Based on the specification from DersuUzalaLanguageSpecification.md
"""

# Vocabulary based on the specification
VOCABULARY = {
    # Special tokens
    '<pad>': 0,   # Padding token
    '<start>': 1, # Start of sequence
    '<end>': 2,   # End of sequence
    '<unk>': 3,   # Unknown token

    # Subjects (60)
    'капитан': 4, 'Дерсу': 5, 'солдат': 6, 'жена': 7, 'сын': 8, 'девчонка': 9,
    'охотник': 10, 'гольд': 11, 'амба': 12, 'медведь': 13, 'козуля': 14, 'олень': 15,
    'рыба': 16, 'соболь': 17, 'птица': 18, 'зверь': 19, 'человек': 20, 'люди': 21,
    'старик': 22, 'мальчик': 23, 'казак': 24, 'собака': 25, 'лошадь': 26, 'змея': 27,
    'волк': 28, 'заяц': 29, 'белка': 30, 'какой-то люди': 31,
    # Natural entities (30)
    'вода': 32, 'огонь': 33, 'дождь': 34, 'ветер': 35, 'солнце': 36, 'луна': 37,
    'звезда': 38, 'небо': 39, 'земля': 40, 'сопка': 41, 'лес': 42, 'река': 43,
    'море': 44, 'тайга': 45, 'след': 46, 'день': 47, 'ночь': 48, 'утро': 49,
    'вечер': 50, 'туман': 51, 'снег': 52, 'лёд': 53, 'камень': 54, 'дерево': 55,
    'трава': 56, 'дорога': 57, 'путь': 58, 'место': 59, 'жизнь': 60, 'смерть': 61,

    # Objects (50)
    'ружье': 62, 'палатка': 63, 'балаган': 64, 'дрова': 65, 'трубка': 66, 'топор': 67,
    'нож': 68, 'лодка': 69, 'котелок': 70, 'спички': 71, 'патрон': 72, 'пуля': 73,
    'рюкзак': 74, 'одежда': 75, 'рубашка': 76, 'штаны': 77, 'шапка': 78, 'дом': 79,
    'берлога': 80, 'костер': 81, 'дым': 82, 'еда': 83, 'кушай': 84, 'мясо': 85,
    'чай': 86, 'соль': 87, 'хлеб': 88, 'письмо': 89, 'карта': 90, 'компас': 91,
    'деньги': 92, 'золото': 93, 'подарок': 94, 'капкан': 95, 'сеть': 96, 'удочка': 97,
    'весло': 98, 'лыжи': 99, 'нарты': 100, 'седло': 101, 'веревка': 102, 'цепь': 103,
    'замок': 104, 'ключ': 105, 'окно': 106, 'дверь': 107, 'стена': 108, 'крыша': 109,
    'печка': 110, 'стол': 111, 'какой-то штука': 112,

    # Actions (60)
    'ходи': 113, 'беги': 114, 'стой': 115, 'сиди': 116, 'лежи': 117, 'спи': 118,
    'кушай': 119, 'пей': 120, 'думай': 121, 'понимай': 122, 'говори': 123, 'слушай': 124,
    'смотри': 125, 'работай': 126, 'делай': 127, 'стреляй': 128, 'лови': 129, 'ищи': 130,
    'найди': 131, 'теряй': 132, 'живи': 133, 'помирай': 134, 'бояться': 135, 'сердись': 136,
    'играй': 137, 'плачь': 138, 'кричи': 139, 'молчи': 140, 'жди': 141, 'помни': 142,
    'забывай': 143, 'знай': 144, 'умей': 145, 'могу': 146, 'хоти': 147, 'надо': 148,
    'давай': 149, 'бери': 150, 'клади': 151, 'неси': 152, 'тащи': 153, 'руби': 154,
    'режь': 155, 'вяжи': 156, 'гоняй': 157, 'прячься': 158, 'верь': 159, 'обмани': 160,
    'помогай': 161, 'мешай': 162, 'учи': 163, 'уходи': 164, 'приходи': 165, 'вернись': 166,
    'останься': 167, 'садись': 168, 'вставай': 169, 'плыви': 170, 'лети': 171, 'падай': 172,
    'что-то делай': 173,

    # Qualities (40)
    'хорошо': 174, 'худо': 175, 'страшно': 176, 'скоро': 177, 'быстро': 178, 'медленно': 179,
    'много': 180, 'мало-мало': 181, 'шибко': 182, 'слабо': 183, 'прямо': 184, 'криво': 185,
    'близко': 186, 'далеко': 187, 'тут': 188, 'там': 189, 'сегодня': 190, 'завтра': 191,
    'вчера': 192, 'раньше': 193, 'потом': 194, 'постоянно': 195, 'один': 196, 'два': 197,
    'три': 198, 'большой': 199, 'маленький': 200, 'старый': 201, 'молодой': 202,
    'сильный': 203, 'слабый': 204, 'добрый': 205, 'злой': 206, 'хитрый': 207, 'глупый': 208,
    'теплый': 209, 'холодный': 210, 'сухой': 211, 'мокрый': 212, 'острый': 213, 'тупой': 214,

    # Pronouns (8)
    'моя': 215, 'тебе': 216, 'его': 217, 'наша': 218, 'ваша': 219, 'кто': 220, 'что': 221, 'все': 222,

    # Conjunctions and Particles (8)
    'и': 223, 'а': 224, 'но': 225, 'не': 226, 'или': 227, 'тоже': 228, 'однако': 229, 'если': 230, 'потому': 231,

    # Questions and Answers (6)
    'да': 232, 'есть': 233, 'нету': 234, 'почему': 235, 'когда': 236, 'где': 237, 'как': 238, 'какой': 239,
}

# Reverse vocabulary for decoding
ID_TO_TOKEN = {v: k for k, v in VOCABULARY.items()}

# Token categories
SUBJECTS = list(range(4, 62))  # 4 to 61
NATURAL_ENTITIES = list(range(32, 62))  # 32 to 61 (subset of subjects)
OBJECTS = list(range(62, 112))  # 62 to 111
ACTIONS = list(range(113, 173))  # 113 to 172
QUALITIES = list(range(174, 214))  # 174 to 213
PRONOUNS = list(range(215, 222))  # 215 to 221
CONJUNCTIONS = list(range(223, 231))  # 223 to 230
QUESTIONS = list(range(232, 239))  # 232 to 238

# Maximum sentence length
MAX_LENGTH = 20

# Grammar rules for sentence generation
def generate_sentence(max_length=MAX_LENGTH):
    """Generate a random valid sentence in Dersu Uzala language."""
    import random

    def random_subject():
        return random.choice([random.choice(list(range(4, 31))), random.choice(list(range(32, 62)))])  # Living or Natural

    def random_object():
        return random.choice(OBJECTS)

    def random_action():
        return random.choice(ACTIONS)

    def random_quality():
        return random.choice(QUALITIES)

    def random_pronoun():
        return random.choice(PRONOUNS)

    def random_conjunction():
        return random.choice(CONJUNCTIONS)

    def random_question():
        return random.choice(QUESTIONS)

    # Generate sentence based on grammar rules
    sentence_type = random.choice(['simple', 'with_quality', 'with_object', 'modal', 'possessive', 'negative', 'question', 'complex'])

    tokens = []

    if sentence_type == 'simple':
        # [Subject] + [Action]
        tokens = [random_subject(), random_action()]

    elif sentence_type == 'with_quality':
        # [Subject] + [Quality] + [Action]
        tokens = [random_subject(), random_quality(), random_action()]

    elif sentence_type == 'with_object':
        # [Subject] + [Object] + [Action]
        tokens = [random_subject(), random_object(), random_action()]

    elif sentence_type == 'modal':
        # надо + [Subject] + [Action] or надо + [Action]
        if random.random() < 0.5:
            tokens = [148, random_subject(), random_action()]  # надо = 148
        else:
            tokens = [148, random_action()]

    elif sentence_type == 'possessive':
        # [Owner] + [Object] + его
        tokens = [random_subject(), random_object(), 217]  # его = 217

    elif sentence_type == 'negative':
        # [Subject] + [Object/Quality] + нету
        if random.random() < 0.5:
            tokens = [random_subject(), random_object(), 234]  # нету = 234
        else:
            tokens = [random_subject(), random_quality(), 234]

    elif sentence_type == 'question':
        # [Question] + [Subject] + [Action]
        tokens = [random_question(), random_subject(), random_action()]

    elif sentence_type == 'complex':
        # Two simple sentences connected by conjunction
        sent1 = generate_sentence(max_length=5)
        sent2 = generate_sentence(max_length=5)
        conj = random_conjunction()
        tokens = sent1 + [conj] + sent2

    # Ensure length doesn't exceed max
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    return tokens

def is_valid_sentence(tokens):
    """Basic validation for sentence structure."""
    # Simple check: ensure it starts with a subject-like token and has an action
    if not tokens:
        return False

    # Check for basic structure
    if len(tokens) >= 2:
        first_token = tokens[0]
        last_token = tokens[-1]
        if first_token in SUBJECTS or first_token in PRONOUNS:
            if last_token in ACTIONS:
                return True

    return False
