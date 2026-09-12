import torch


# ============================================================
# TinyVideo V1.4 - Text Encoder
# ============================================================

TEXT_DIM = 128
VOCAB_SIZE = 256
MAX_TEXT_LENGTH = TEXT_DIM


RESOLUTION_IDS = {
    "unknown": 0,
    "1080p": 1,
    "720p": 2,
    "420p": 3,
}


def tokenize_text(text):
    """
    Convert text into fixed-length integer character tokens.

    Returns:
        torch.LongTensor with shape [128]
    """
    text = str(text).lower()

    tokens = torch.zeros(
        TEXT_DIM,
        dtype=torch.long,
    )

    for i, char in enumerate(text):
        if i >= MAX_TEXT_LENGTH:
            break

        tokens[i] = ord(char) % VOCAB_SIZE

    return tokens


def encode_text(text):
    """
    Convert text into the floating-point embedding expected
    by the TinyVideo diffusion model.

    Returns:
        torch.FloatTensor with shape [128]
    """
    tokens = tokenize_text(text)

    return tokens.float() / 255.0


def extract_resolution_id(prompt):
    """
    Extract the resolution-conditioning ID from a prompt.

    Examples:
        'a car driving, 1080p' -> 1
        'a car driving, 720p'  -> 2
        'a car driving, 420p'  -> 3
        'a car driving'        -> 0
    """
    prompt = str(prompt).lower()

    if "1080p" in prompt:
        return RESOLUTION_IDS["1080p"]

    if "720p" in prompt:
        return RESOLUTION_IDS["720p"]

    if "420p" in prompt:
        return RESOLUTION_IDS["420p"]

    return RESOLUTION_IDS["unknown"]