from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer():
    return TfidfVectorizer(
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2),
        token_pattern=r"[^ ]+"
    )
