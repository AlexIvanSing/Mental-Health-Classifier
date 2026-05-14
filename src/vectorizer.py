from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(config: dict) -> TfidfVectorizer:
    """
    Build a TF-IDF vectorizer from a configuration dictionary.

    Reads the `vectorizer` section of the config and instantiates a
    `TfidfVectorizer` with sublinear term-frequency scaling. The `max_df`
    cap effectively filters corpus-wide stopwords, so no explicit stopword
    list is provided.

    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML. Must contain a
        `vectorizer` key with: `ngram_range`, `min_df`, `max_df`,
        `sublinear_tf`, `max_features`.

    Returns
    -------
    sklearn.feature_extraction.text.TfidfVectorizer
        Unfitted vectorizer ready to be plugged into a Pipeline.
    """
    return TfidfVectorizer(
        ngram_range=tuple(config["vectorizer"]["ngram_range"]),
        min_df=config["vectorizer"]["min_df"],
        max_df=config["vectorizer"]["max_df"],
        sublinear_tf=config["vectorizer"]["sublinear_tf"],
        max_features=config["vectorizer"]["max_features"],
    )

