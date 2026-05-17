# ============================================================================
# Variantes de preprocesamiento de texto
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import re
from functools import lru_cache

import pandas as pd
from ftfy import fix_text


# --- NLTK lazy loading -----------------------------------------------------
#
# The stopword/stemming variants need NLTK resources. We import the heavy
# pieces lazily inside the helpers below so that simply importing this module
# doesn't force a download. `_ensure_nltk_resource` downloads quietly only if
# the resource isn't already present locally.

def _ensure_nltk_resource(resource: str, download_name: str | None = None) -> None:
    """
    Make sure an NLTK resource is available; download it quietly if not.

    Parameters
    ----------
    resource : str
        Path of the resource as nltk.data.find expects it
        (e.g., "corpora/stopwords", "tokenizers/punkt").
    download_name : str, optional
        Identifier passed to `nltk.download`. Defaults to the last segment
        of `resource` (e.g., "stopwords" for "corpora/stopwords").
    """
    import nltk
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name or resource.split("/")[-1], quiet=True)

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

    return re.sub(r'@\w+|u/\w+', ' ', text)


def normalize_hashtags(text: str) -> str:
    """Remove hashtag symbols while preserving the word."""

    return re.sub(r'#\s*(\w+)', r'\1', text)


def remove_emojis(text: str) -> str:
    """Remove unicode emojis and common ASCII emoticons from text."""
    text = re.sub(r'[\U00010000-\U0010ffff]', ' ', text)
    text = re.sub(r'[:;=][\'\-]?[)(D/\\|PpOo]', ' ', text)
    return text

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
# posible de aplicar en futuro
#def remove_stop_words(sentence:str) -> list:
#    return

def preprocessing(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Apply `clean_text` to a text column of a DataFrame in place.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    text_column : str
        Name of the column to clean.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with the cleaned column, or None on failure.
    """
    try:
        df[text_column] = df[text_column].apply(clean_text)
        return df
    except Exception as e:
        print(f"dataframe unable to preprocess: {e}")
        return None
    
## codigo muerto de momento




# Negations that must NEVER be removed: they flip the semantics of a sentence
# and are especially load-bearing in suicide-ideation text
# (e.g., "I do not want to live" vs. "I do want to live").
_NEGATION_WHITELIST = frozenset({
    "no", "not", "never", "nor", "neither",
    "without", "nothing", "nobody", "nowhere", "none",
})

# Pure-grammar tokens (articles, prepositions, conjunctions, basic auxiliaries).
# Intentionally excludes negations and any token with emotional content
# ("alone", "hate", "die" etc. are NOT here).
_DOMAIN_STOPWORDS = frozenset({
    # articles
    "a", "an", "the",
    # prepositions
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "into", "onto", "upon", "over", "under", "about", "above",
    "below", "between", "through", "during", "before", "after",
    # conjunctions
    "and", "or", "but", "so", "if", "because", "although", "though",
    "while", "since", "unless", "until", "whereas",
    # auxiliaries / be / have / do (no negation forms)
    "is", "am", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may",
    "might", "must",
    # pronouns / determiners that are pure grammar
    "this", "that", "these", "those", "it", "its",
})


@lru_cache(maxsize=1)
def _nltk_stopwords_no_negations() -> frozenset[str]:
    """
    NLTK English stopwords minus the negation whitelist.

    Cached because (a) loading NLTK on every call is wasteful and (b) the
    result is immutable.
    """
    _ensure_nltk_resource("corpora/stopwords")
    from nltk.corpus import stopwords
    return frozenset(stopwords.words("english")) - _NEGATION_WHITELIST


@lru_cache(maxsize=1)
def _porter_stemmer():
    """
    Return a memoized PorterStemmer instance.

    The class is stateless so a single shared instance is safe.
    """
    from nltk.stem.porter import PorterStemmer
    return PorterStemmer()


def clean_text_with_stopwords_nltk(text: str) -> str:
    """
    Preprocessing variant: baseline `clean_text` followed by removal of
    NLTK English stopwords, **preserving the negation whitelist**.

    Rationale
    ---------
    Standard stopword lists strip out tokens like "not" and "never" — which is
    catastrophic for suicide-ideation classification because they invert
    meaning. We keep them.

    Authors (equipo del proyecto TC3002B):
        Aislinn Ruiz Sandoval, Iván Alexander Ramos Ramírez,
        Miguel Ángel Galicia Sánchez, Víctor Alejandro Morales García.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text with stopwords (except negations) removed.
    """
    base = clean_text(text)
    if not base:
        return ""

    stops = _nltk_stopwords_no_negations()
    return " ".join(tok for tok in base.split() if tok not in stops)


def clean_text_with_stopwords_domain(text: str) -> str:
    """
    Preprocessing variant: baseline `clean_text` followed by removal of a
    **curated domain-aware stopword list** that contains only purely
    grammatical tokens (articles, prepositions, conjunctions, auxiliaries).

    Rationale
    ---------
    Tokens carrying emotional or experiential content ("alone", "die", "hate",
    "hopeless", "tired", and every negation) are explicitly kept — even when
    they would be on a standard stopword list — because they are precisely
    the discriminative signal in this domain.

    Authors (equipo del proyecto TC3002B):
        Aislinn Ruiz Sandoval, Iván Alexander Ramos Ramírez,
        Miguel Ángel Galicia Sánchez, Víctor Alejandro Morales García.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text with only pure-grammar tokens removed.
    """
    base = clean_text(text)
    if not base:
        return ""

    return " ".join(tok for tok in base.split() if tok not in _DOMAIN_STOPWORDS)


def clean_text_with_stemming(text: str) -> str:
    """
    Preprocessing variant: baseline `clean_text` followed by Porter stemming
    applied token-by-token.

    Rationale
    ---------
    Collapses morphological variants ("die" / "dying" / "died" / "dies") to a
    single stem so they share a TF-IDF feature. Trades fine-grained morphology
    for vocabulary reduction — useful when the corpus is small (~1.5k docs).

    Authors (equipo del proyecto TC3002B):
        Aislinn Ruiz Sandoval, Iván Alexander Ramos Ramírez,
        Miguel Ángel Galicia Sánchez, Víctor Alejandro Morales García.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text where each whitespace-separated token has been stemmed.
    """
    base = clean_text(text)
    if not base:
        return ""

    stemmer = _porter_stemmer()
    return " ".join(stemmer.stem(tok) for tok in base.split())