import re
from ftfy import fix_text

def normalize_quotes(text):
    if not isinstance(text, str):
        return text

    replacements = {
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"'
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    return text

import re
from ftfy import fix_text


def normalize_quotes(text: str) -> str:
    """Normalize quotation marks and apostrophes."""

    replacements = {
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "«": '"',
        "»": '"'
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def fix_encoding(text: str) -> str:
    """Repair common text encoding corruption issues."""

    text = fix_text(text)
    text = normalize_quotes(text)

    return text


def remove_urls(text: str) -> str:
    """Remove URLs from text."""

    return re.sub(r'https?://\S+|www\.\S+', ' ', text)


def remove_mentions(text: str) -> str:
    """Remove user mentions from text."""

    return re.sub(r'@\w+', ' ', text)


def normalize_hashtags(text: str) -> str:
    """Remove hashtag symbols while preserving the word."""

    return re.sub(r'#(\w+)', r'\1', text)


def remove_emojis(text: str) -> str:
    """Remove emojis and supplementary Unicode symbols."""

    return re.sub(r'[\U00010000-\U0010ffff]', ' ', text)


def remove_special_characters(text: str) -> str:
    """Remove unsupported special characters."""

    return re.sub(r"[^a-zA-Z0-9\s']", ' ', text)


def normalize_spaces(text: str) -> str:
    """Normalize consecutive whitespace characters."""

    return re.sub(r'\s+', ' ', text).strip()


def clean_text(text: str) -> str:
    """
    Execute the preprocessing pipeline for classic
    machine learning text classification tasks.

    Steps
    -----
    1. Repair encoding issues.
    2. Normalize quotes and apostrophes.
    3. Convert text to lowercase.
    4. Remove URLs.
    5. Remove user mentions.
    6. Normalize hashtags.
    7. Remove emojis.
    8. Remove special characters.
    9. Normalize whitespace.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Fully preprocessed text.
    """

    if not isinstance(text, str):
        return ""

    text = fix_encoding(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_mentions(text)
    text = normalize_hashtags(text)
    text = remove_emojis(text)
    text = remove_special_characters(text)
    text = normalize_spaces(text)

    return text

def tokenize_text(sentence: str) -> list:
    