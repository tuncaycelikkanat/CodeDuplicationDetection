from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer():
    """Token-level TF-IDF vectorizer with trigrams."""
    return TfidfVectorizer(
        min_df=3,
        max_df=0.95,
        ngram_range=(1, 3),
        token_pattern=r"[^ ]+",
        sublinear_tf=True,
        max_features=2000  # Select the top 2000 most important/frequent features
    )


def build_char_tfidf_vectorizer():
    """Character-level n-gram TF-IDF vectorizer (wider range)."""
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 6),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        max_features=5000
    )
