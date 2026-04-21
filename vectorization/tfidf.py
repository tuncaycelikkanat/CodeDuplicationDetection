from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF


def build_tfidf_vectorizer():
    """Token-level TF-IDF vektörleyici (trigram'lı)."""
    return TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        token_pattern=r"[^ ]+",
        sublinear_tf=True,
        max_features=TFIDF_MAX_FEATURES  # Type-4 tespiti için semantik feature'lara ağırlık kaydırılmış
    )


def build_char_tfidf_vectorizer():
    """Karakter seviyesi n-gram TF-IDF vektörleyici."""
    return TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 6),
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
        max_features=5000
    )
