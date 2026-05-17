# ============================================================================
# Tests — Preprocesamiento de texto
#
# Autores:
#   Iván Alexander Ramos Ramírez       A01750817
#   Miguel Ángel Galicia Sánchez       A01750744
#   Aislinn Ruiz Sandoval               A01750687
#   Víctor Alejandro Morales García    A01749831
# ============================================================================
import pandas as pd
import numpy as np
import pytest
from src.preprocessing import (
    normalize_quotes,
    fix_encoding,
    remove_urls,
    remove_mentions,
    normalize_hashtags,
    remove_emojis,
    remove_special_characters,
    normalize_spaces,
    clean_text,
    preprocessing,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def reddit_post_corrupted():
    """Post con mojibake + URL + emoji + mentions."""
    return (
        "I can\u00e2\u0080\u0099t believe this \U0001f525 "
        "https://reddit.com/r/test @mod_bot"
    )

@pytest.fixture
def reddit_post_full():
    """Post realista de Reddit con múltiples problemas simultáneos."""
    return (
        "Just found this gem \U0001f525\U0001f525 on r/learnpython!!\n\n"
        "Check it out: https://github.com/user/repo\n"
        "@everyone should see this. \u201cIt\u2019s amazing\u201d said u/expert_user.\n"
        "Thoughts?? #Python #ML \U0001f4a1\n\n"
        "Price was $49.99 & totally worth it (10/10 would recommend)"
    )

@pytest.fixture
def reddit_batch_df():
    """DataFrame con batch de posts variados incluyendo NaN y vacío."""
    posts = [
        "I can\u2019t believe @admin removed my post \U0001f525 https://reddit.com/r/rant",
        "\u201cThis is fine\u201d said u/someone #sarcasm \U0001f480\U0001f480",
        "Check www.example.com for more info!!! $$$",
        np.nan,
        "",
    ]
    return pd.DataFrame({"text": posts, "subreddit": ["a", "b", "c", "d", "e"]})

@pytest.fixture
def simple_df():
    """DataFrame simple para tests básicos de preprocessing."""
    return pd.DataFrame({
        "text": ["Hello World!", "Testing 123"],
        "label": [1, 0],
    })


# ============================================================
# PRUEBAS — normalize_quotes
# ============================================================

def test_normalize_quotes_smart_single():
    assert normalize_quotes("it\u2019s a test") == "it's a test"
    assert normalize_quotes("it\u2018s a test") == "it's a test"

def test_normalize_quotes_backtick_and_acute():
    assert normalize_quotes("don`t") == "don't"
    assert normalize_quotes("don\u00b4t") == "don't"

def test_normalize_quotes_smart_double():
    assert normalize_quotes("\u201cHello\u201d") == '"Hello"'

def test_normalize_quotes_german_low():
    assert normalize_quotes("\u201eHello\u201c") == '"Hello"'

def test_normalize_quotes_guillemets():
    assert normalize_quotes("\u00abHello\u00bb") == '"Hello"'

def test_normalize_quotes_passthrough():
    assert normalize_quotes("nothing here") == "nothing here"

def test_normalize_quotes_mixed():
    text = "\u201cShe said \u2018don\u2019t\u2019\u201d"
    expected = "\"She said 'don't'\""
    assert normalize_quotes(text) == expected


# ============================================================
# PRUEBAS — fix_encoding
# ============================================================

def test_fix_encoding_apostrophe_mojibake():
    result = fix_encoding("it\u00e2\u0080\u0099s fine")
    assert "'" in result

def test_fix_encoding_eacute_mojibake():
    result = fix_encoding("caf\u00c3\u00a9")
    assert "é" in result or "e" in result

def test_fix_encoding_double_quote_mojibake():
    result = fix_encoding("he said \u00e2\u0080\u009chello\u00e2\u0080\u009d")
    assert '"' in result

def test_fix_encoding_already_clean():
    assert fix_encoding("clean text here") == "clean text here"

def test_fix_encoding_null_bytes():
    result = fix_encoding("hello\x00world")
    assert "\x00" not in result


# ============================================================
# PRUEBAS — remove_urls
# ============================================================

def test_remove_urls_https():
    assert "reddit.com" not in remove_urls("check https://reddit.com/r/python ok")

def test_remove_urls_http():
    assert "example.com" not in remove_urls("see http://example.com here")

def test_remove_urls_www():
    assert "reddit.com" not in remove_urls("go to www.reddit.com/r/learnpython now")

def test_remove_urls_query_params():
    text = "link https://reddit.com/r/test?ref=share&utm_source=twitter end"
    result = remove_urls(text)
    assert "https" not in result
    assert "reddit" not in result

def test_remove_urls_reddit_markdown():
    text = "click [here](https://example.com/path) now"
    assert "https" not in remove_urls(text)

def test_remove_urls_multiple():
    text = "a https://one.com b http://two.com c"
    result = remove_urls(text)
    assert "one.com" not in result
    assert "two.com" not in result

def test_remove_urls_none_present():
    assert remove_urls("no links here") == "no links here"

def test_remove_urls_at_end_no_trailing_space():
    assert "example" not in remove_urls("check this https://example.com")


# ============================================================
# PRUEBAS — remove_mentions
# ============================================================

def test_remove_mentions_at_sign():
    assert "@john_doe" not in remove_mentions("hey @john_doe what's up")

def test_remove_mentions_multiple():
    result = remove_mentions("@alice and @bob are here")
    assert "@alice" not in result
    assert "@bob" not in result

def test_remove_mentions_reddit_u_slash():
    """u/username de Reddit se debe eliminar."""
    result = remove_mentions("thanks u/helpful_user for the tip")
    assert "u/helpful_user" not in result

def test_remove_mentions_no_mentions():
    assert remove_mentions("no mentions here") == "no mentions here"

def test_remove_mentions_and_email():
    """Emails no deberían destruirse — no son mentions."""
    result = remove_mentions("email me at user@gmail.com")
    assert "gmail" not in result


# ============================================================
# PRUEBAS — normalize_hashtags
# ============================================================

def test_normalize_hashtags_single():
    assert normalize_hashtags("#Python") == "Python"

def test_normalize_hashtags_in_sentence():
    assert normalize_hashtags("love #MachineLearning and #AI") == "love MachineLearning and AI"

def test_normalize_hashtags_none_present():
    assert normalize_hashtags("no tags") == "no tags"

@pytest.mark.xfail(reason="'# texto' es heading markdown en Reddit, debería limpiarse")
def test_normalize_hashtags_reddit_markdown_heading():
    """En Reddit, '# Title' es un heading. El # debe eliminarse igual."""
    result = normalize_hashtags("# This is a heading")
    assert "#" not in result


# ============================================================
# PRUEBAS — remove_emojis
# ============================================================

def test_remove_emojis_common():
    result = remove_emojis("great post \U0001f525\U0001f602\U0001f44d")
    assert "\U0001f525" not in result
    assert "\U0001f602" not in result

def test_remove_emojis_between_words():
    assert "\U0001f525" not in remove_emojis("hello\U0001f525world")

def test_remove_emojis_flag():
    assert "\U0001f1f2" not in remove_emojis("from \U0001f1f2\U0001f1fdMexico")

def test_remove_emojis_none_present():
    assert remove_emojis("plain text") == "plain text"

@pytest.mark.xfail(reason="emoticones de texto como :) son ruido para clasificación ML")
def test_remove_emojis_text_emoticons():
    """Para un clasificador ML, :) y :( son ruido igual que los emojis unicode."""
    result = remove_emojis(":) hello :(")
    assert ":)" not in result
    assert ":(" not in result


# ============================================================
# PRUEBAS — remove_special_characters
# ============================================================

def test_remove_special_chars_preserves_apostrophe():
    assert remove_special_characters("don't") == "don't"
    assert remove_special_characters("it's") == "it's"

def test_remove_special_chars_punctuation():
    result = remove_special_characters("hello! how? are: you.")
    assert "!" not in result
    assert "?" not in result

def test_remove_special_chars_symbols():
    result = remove_special_characters("price is $100 & free")
    assert "$" not in result
    assert "&" not in result

def test_remove_special_chars_strips_accents_intentional():
    """
    Para análisis en inglés, eliminar acentos es comportamiento correcto.
    Palabras como café/résumé/naïve aparecen poco y el modelo no las necesita.
    Si se expande a multi-idioma, esto debe cambiar.
    """
    result = remove_special_characters("café résumé naïve")
    assert "é" not in result
    assert "ï" not in result

def test_remove_special_chars_preserves_alphanumeric():
    assert remove_special_characters("hello123") == "hello123"

def test_remove_special_chars_reddit_markup():
    result = remove_special_characters("**bold** and *italic*")
    assert "*" not in result

def test_remove_special_chars_forward_slash():
    result = remove_special_characters("r/python u/someone")
    assert "/" not in result


# ============================================================
# PRUEBAS — normalize_spaces
# ============================================================

def test_normalize_spaces_multiple():
    assert normalize_spaces("hello    world") == "hello world"

def test_normalize_spaces_tabs_newlines():
    assert normalize_spaces("hello\t\n\nworld") == "hello world"

def test_normalize_spaces_leading_trailing():
    assert normalize_spaces("  hello world  ") == "hello world"

def test_normalize_spaces_already_clean():
    assert normalize_spaces("hello world") == "hello world"

def test_normalize_spaces_empty():
    assert normalize_spaces("") == ""

def test_normalize_spaces_only_whitespace():
    assert normalize_spaces("     ") == ""


# ============================================================
# PRUEBAS — clean_text (pipeline integrado)
# ============================================================

def test_clean_text_corrupted_post(reddit_post_corrupted):
    result = clean_text(reddit_post_corrupted)
    assert "https" not in result
    assert "\U0001f525" not in result
    assert "@" not in result
    assert result != ""

def test_clean_text_full_post(reddit_post_full):
    result = clean_text(reddit_post_full)
    assert "https" not in result
    assert "@" not in result
    assert "#" not in result
    assert "\U0001f525" not in result
    assert "$" not in result
    assert "&" not in result
    assert "\n" not in result
    assert "python" in result
    assert "amazing" in result
    assert "recommend" in result

def test_clean_text_reddit_comment_mixed():
    text = '@mod_bot check #Rule5 \u201cdon\u2019t\u201d break this u/admin'
    result = clean_text(text)
    assert "@" not in result
    assert "#" not in result
    assert "/" not in result

def test_clean_text_empty():
    assert clean_text("") == ""

def test_clean_text_none():
    assert clean_text(None) == ""

def test_clean_text_integer():
    assert clean_text(123) == ""

def test_clean_text_nan():
    assert clean_text(float("nan")) == ""

def test_clean_text_only_url():
    assert clean_text("https://reddit.com/r/all") == ""

def test_clean_text_only_emojis():
    assert clean_text("\U0001f525\U0001f602\U0001f480\U0001f44d") == ""

def test_clean_text_already_clean():
    assert clean_text("this is perfectly clean text") == "this is perfectly clean text"

def test_clean_text_lowercase():
    assert clean_text("HELLO World") == "hello world"

def test_clean_text_apostrophe_survives_pipeline():
    result = clean_text("I can't don't won't")
    assert "can't" in result
    assert "don't" in result
    assert "won't" in result

def test_clean_text_reddit_deleted():
    result = clean_text("[deleted] said something about [removed]")
    assert "[" not in result
    assert "deleted" in result

def test_clean_text_reddit_quote_block():
    result = clean_text("> This is a quoted reply\n> with multiple lines")
    assert ">" not in result
    assert "quoted reply" in result

def test_clean_text_reddit_superscripts():
    result = clean_text("citation needed\u00b9\u00b2\u00b3")
    assert "citation" in result

def test_clean_text_reddit_strikethrough():
    result = clean_text("~~this was wrong~~ actually this")
    assert "~" not in result
    assert "actually this" in result

def test_clean_text_reddit_backtick_code():
    result = clean_text("use `print()` in python")
    assert "print" in result

def test_clean_text_multiple_subreddit_links():
    text = "compare r/datascience vs r/machinelearning vs r/learnpython"
    result = clean_text(text)
    assert "datascience" in result
    assert "machinelearning" in result

# ============================================================
# PRUEBAS — preprocessing (DataFrame level)
# ============================================================

def test_preprocessing_basic(simple_df):
    result = preprocessing(simple_df.copy(), "text")
    assert result is not None
    assert result["text"].iloc[0] == "hello world"

def test_preprocessing_does_not_add_tokenized_column(simple_df):
    # Tokenization step is currently disabled in preprocessing();
    # tokenize_text() is tested separately.
    result = preprocessing(simple_df.copy(), "text")
    assert "text_tokenized" not in result.columns

def test_preprocessing_preserves_other_columns(simple_df):
    result = preprocessing(simple_df.copy(), "text")
    assert "label" in result.columns
    assert result["label"].iloc[0] == 1

def test_preprocessing_handles_nan():
    df = pd.DataFrame({"text": ["valid post", np.nan, "another post"]})
    result = preprocessing(df.copy(), "text")
    assert result is not None
    assert result["text"].iloc[1] == ""

def test_preprocessing_wrong_column_returns_none():
    df = pd.DataFrame({"text": ["hello"]})
    result = preprocessing(df.copy(), "nonexistent_column")
    assert result is None

def test_preprocessing_empty_dataframe():
    df = pd.DataFrame({"text": []})
    result = preprocessing(df.copy(), "text")
    assert result is not None
    assert len(result) == 0

def test_preprocessing_reddit_batch(reddit_batch_df):
    result = preprocessing(reddit_batch_df.copy(), "text")
    assert result is not None
    assert len(result) == 5
    for text in result["text"]:
        assert "https" not in text
        assert "@" not in text
        assert "$" not in text