from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(config: dict) -> TfidfVectorizer:
    
    return TfidfVectorizer(
        ngram_range= tuple(config["vectorizer"]["ngram_range"]),
        min_df= config["vectorizer"]["min_df"],
        max_df= config["vectorizer"]["max_df"], # por esto no necesitamos stopwords, o al menos no de momento
        sublinear_tf = config["vectorizer"]["sublinear_tf"],
        max_features= config["vectorizer"]["max_features"]
    )

