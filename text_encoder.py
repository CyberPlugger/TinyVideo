import torch


VOCAB_SIZE = 256
TEXT_DIM = 128


def encode_text(text):

    text = text.lower()

    vector = torch.zeros(
        TEXT_DIM,
        dtype=torch.float32
    )

    # Превращаем символы текста
    # в простое числовое представление.

    for i, char in enumerate(text):

        if i >= TEXT_DIM:
            break

        vector[i] = ord(char) / 255.0

    return vector
