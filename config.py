from utils.logger import Log
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

# Stage-1 (HistGradientBoosting) modelinin "kolay klon" kararı için eşik.
# Bu değer main.py, test_automation.py ve web_demo/app.py tarafından paylaşılır.
CASCADE_STAGE1_THRESHOLD = 0.85

# ── Boyut Azaltma ─────────────────────────────────────────────────────────────
SVD_N_COMPONENTS = 100  # TruncatedSVD bileşen sayısı (LSA) - Daha fazla varyans açıklamak için artırıldı

# ── TF-IDF Vektörizasyon ──────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 500    # Token TF-IDF kelime hazinesi boyutu
TFIDF_NGRAM_RANGE  = (1, 3) # Unigram, bigram, trigram
TFIDF_MIN_DF       = 1      # Test otomasyonu gibi küçük datasetlerde çökmemesi için 1'e düşürüldü
TFIDF_MAX_DF       = 0.95   # Maksimum döküman frekansı (oransal)

# ── Eğitim Varsayılanları ─────────────────────────────────────────────────────
DEFAULT_PAIRS      = 800_000
DEFAULT_SEED       = 42
DEFAULT_TEST_SIZE  = 0.2
DEFAULT_CV_FOLDS   = 5
# Eğitim çiftlerinde klon oranı. Varsayılan 0.5 (dengeli).
# Gerçekçi sınıf dağılımı simülasyonu için 0.1 kullanın.
DEFAULT_POSITIVE_RATIO = 0.5
HARD_MINING_RATIO  = 0.3  # Hard positive/negative mining oranı

# ── Ensemble Mimarisi ─────────────────────────────────────────────────────────
# Feature vektöründeki SVD bloğunun başlangıç indeksi.
# pair_generator.py feature sırası:
#   [0..3]    Lexical (cos_token, length_ratio, manhattan, euclidean)
#   [4..43]   AST ratios (20) + AST diffs (20)
#   [44]      CF pattern similarity
#   [45..51]  Semantic Jaccard x7
#   [52]      Type profile cosine
#   [53..152] SVD diff (SVD_N_COMPONENTS boyutunda)
#   [153..216] SSL PCA diff (SSL_PCA_COMPONENTS boyutunda) — sadece --use-ssl ile
ENSEMBLE_SVD_START_IDX = 53

# Stage-1 model için kullanılan özellik sayısı (lexical + AST + CF).
STAGE1_FEATURE_COUNT = 45

# ── SSL Gömme Boyutu Azaltma ──────────────────────────────────────────────────
# CodeBERT 768-D embedding'leri PCA ile bu boyuta indirgenir.
# Bellek: SSL_PCA_COMPONENTS × N_pairs × 4 byte
#   64  → ~200 MB (800K çift için) — önerilen
#   128 → ~400 MB — daha iyi Type-4 için denenebilir
SSL_PCA_COMPONENTS = 64

# ── Donanım ───────────────────────────────────────────────────────────────────
OMP_NUM_THREADS = 8
MKL_NUM_THREADS = 8
