# ============================================================================
# Tests — Variantes de preprocesamiento
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
"""
Tests for the preprocessing variants:
- clean_text_with_stopwords_nltk
- clean_text_with_stopwords_domain
- clean_text_with_stemming

These tests guard the contract each variant promises: negations preserved,
emotional content preserved, stems collapsed, etc.
"""

import pytest

from src.preprocessing import (
    clean_text_with_stopwords_nltk,
    clean_text_with_stopwords_domain,
    clean_text_with_stemming,
)


# --- clean_text_with_stopwords_nltk ---------------------------------------

NEGATIONS = ("no", "not", "never", "nor", "neither",
             "without", "nothing", "nobody", "nowhere", "none")


@pytest.mark.parametrize("negation", NEGATIONS)
def test_nltk_variant_preserves_negation(negation):
    """Every negation in the whitelist must survive the stopword filter."""
    text = f"I am {negation} alone in this world"
    result = clean_text_with_stopwords_nltk(text)
    assert negation in result.split(), (
        f"Negation {negation!r} was unexpectedly removed; got: {result!r}"
    )


@pytest.mark.parametrize("stopword", ["the", "a", "is", "an", "of", "to"])
def test_nltk_variant_removes_standard_stopwords(stopword):
    """Standard non-negation stopwords must be filtered out."""
    text = f"this is {stopword} test of the system"
    result = clean_text_with_stopwords_nltk(text)
    assert stopword not in result.split(), (
        f"Stopword {stopword!r} should have been removed; got: {result!r}"
    )


def test_nltk_variant_preserves_content_words():
    """Emotional / content words must pass through untouched."""
    text = "I feel hopeless alone and tired every day"
    result = clean_text_with_stopwords_nltk(text)
    for word in ("feel", "hopeless", "alone", "tired", "every", "day"):
        assert word in result.split(), f"Content word {word!r} dropped: {result!r}"


def test_nltk_variant_returns_nonempty_for_meaningful_input():
    text = "I want to end my life tonight please help"
    result = clean_text_with_stopwords_nltk(text)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_nltk_variant_empty_input():
    assert clean_text_with_stopwords_nltk("") == ""


def test_nltk_variant_only_stopwords_input():
    """If the input collapses to only stopwords, output is empty (no negations either)."""
    result = clean_text_with_stopwords_nltk("the a is of to and or but")
    assert result == ""


# --- clean_text_with_stopwords_domain -------------------------------------

@pytest.mark.parametrize("emotional_word", [
    "alone", "hopeless", "tired", "die", "hate",
    "scared", "afraid", "sad", "lonely", "empty",
])
def test_domain_variant_preserves_emotional_words(emotional_word):
    """The curated domain list must NEVER strip emotional / experiential content."""
    text = f"I feel {emotional_word} every single day"
    result = clean_text_with_stopwords_domain(text)
    assert emotional_word in result.split(), (
        f"Emotional word {emotional_word!r} was dropped; got: {result!r}"
    )


@pytest.mark.parametrize("negation", NEGATIONS)
def test_domain_variant_preserves_negations(negation):
    text = f"I do {negation} want to feel this anymore"
    result = clean_text_with_stopwords_domain(text)
    assert negation in result.split(), (
        f"Negation {negation!r} was dropped by domain variant; got: {result!r}"
    )


@pytest.mark.parametrize("grammar_token", ["the", "a", "of", "to", "and", "is", "was"])
def test_domain_variant_removes_pure_grammar(grammar_token):
    text = f"this is {grammar_token} sentence in english"
    result = clean_text_with_stopwords_domain(text)
    assert grammar_token not in result.split(), (
        f"Grammar token {grammar_token!r} survived; got: {result!r}"
    )


def test_domain_variant_returns_nonempty_for_meaningful_input():
    text = "I cannot keep living like this anymore"
    result = clean_text_with_stopwords_domain(text)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_domain_variant_empty_input():
    assert clean_text_with_stopwords_domain("") == ""


# --- clean_text_with_stemming ---------------------------------------------

def test_stemming_collapses_die_family():
    """The whole point of stemming: morphological variants share a stem."""
    s_die    = clean_text_with_stemming("die")
    s_dying  = clean_text_with_stemming("dying")
    s_died   = clean_text_with_stemming("died")
    assert s_die == s_dying == s_died, (
        f"Expected die/dying/died to share a stem; got "
        f"die={s_die!r} dying={s_dying!r} died={s_died!r}"
    )


def test_stemming_collapses_run_family():
    s_run     = clean_text_with_stemming("run")
    s_running = clean_text_with_stemming("running")
    s_runs    = clean_text_with_stemming("runs")
    assert s_run == s_running == s_runs


def test_stemming_returns_nonempty_for_meaningful_input():
    text = "I am running out of options"
    result = clean_text_with_stemming(text)
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_stemming_empty_input():
    assert clean_text_with_stemming("") == ""


def test_stemming_preserves_word_count():
    """Stemming must operate token-by-token, not collapse the sentence."""
    text = "running runs ran runner"
    result = clean_text_with_stemming(text)
    assert len(result.split()) == 4


# --- Smoke check: all variants chain correctly with clean_text ------------

@pytest.mark.parametrize("variant_fn", [
    clean_text_with_stopwords_nltk,
    clean_text_with_stopwords_domain,
    clean_text_with_stemming,
])
def test_variant_applies_base_cleanup(variant_fn):
    """Every variant must inherit clean_text's URL/mention/case normalization."""
    dirty = "Check https://reddit.com/r/test @mod_bot THIS is FINE"
    result = variant_fn(dirty)
    # base cleanup: lowercase + URLs/mentions stripped
    assert "https" not in result
    assert "@mod_bot" not in result
    assert "THIS" not in result
