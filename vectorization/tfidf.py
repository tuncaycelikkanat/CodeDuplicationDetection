from sklearn.feature_extraction.text import TfidfVectorizer
from config import TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF
from utils.logger import Log



def build_tfidf_vectorizer():
    """Token-level TF-IDF vektörleyici (trigram'lı)."""
    return TfidfVectorizer(
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE,
        token_pattern=r"[^ ]+",
        sublinear_tf=True,
        norm=None,  # UYARI: norm=None olmalı! L2 normalization kısa ve uzun kodların TF-IDF mesafesini (euclidean) yok eder.
        max_features=TFIDF_MAX_FEATURES  # Type-4 tespiti için semantik feature'lara ağırlık kaydırılmış
    )
