"""
Merkezi yapılandırma — CodeDuplicationDetection
================================================
Tüm sabit değerler ve magic number'lar burada tanımlanır.
Kodun herhangi bir yerinde hard-coded değer kullanmak yerine
bu modülden import edin.
"""

# ── Cascade Architecture ──────────────────────────────────────────────────────
# Kelime benzerliği bu eşiğin üzerindeyse XGBoost'a sormadan direkt klon say.
CASCADE_THRESHOLD = 0.85
# Daha agresif Type-3 tespiti için 0.70'e düşürülebilir (FP riski artar).
# Daha yüksek Precision için 0.90'a çıkarılabilir (Type-3 Recall düşer).

# ── Boyut Azaltma ─────────────────────────────────────────────────────────────
SVD_N_COMPONENTS = 50  # TruncatedSVD bileşen sayısı (LSA)

# ── TF-IDF Vektörizasyon ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 500    # Token TF-IDF kelime hazinesi boyutu
TFIDF_NGRAM_RANGE  = (1, 3) # Unigram, bigram, trigram
TFIDF_MIN_DF       = 3      # Minimum döküman frekansı
TFIDF_MAX_DF       = 0.95   # Maksimum döküman frekansı (oransal)

# ── Eğitim Varsayılanları ─────────────────────────────────────────────────────
DEFAULT_PAIRS      = 800_000
DEFAULT_SEED       = 42
DEFAULT_TEST_SIZE  = 0.2
DEFAULT_CV_FOLDS   = 5
# Eğitim çiftlerinde klon oranı. Varsayılan 0.5 (dengeli).
# Gerçekçi sınıf dağılımı simülasyonu için 0.1 kullanın.
DEFAULT_POSITIVE_RATIO = 0.5

# ── Donanım ───────────────────────────────────────────────────────────────────
OMP_NUM_THREADS = 8
MKL_NUM_THREADS = 8
