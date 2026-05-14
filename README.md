# CodeDuplicationDetection

C/C++ kaynak kodları için **4 farklı klon tipini** (Type 1–4) tespit eden makine öğrenmesi tabanlı bir sistem.

## Proje Yapısı

```
CodeDuplicationDetection/
├── config.py                    # Merkezi sabitler (CASCADE, STAGE1, SSL_PCA, ...)
├── main.py                      # AKTiF EGITIM (Two-Stage Cascade Egitim Betigi)
├── requirements.txt             # Temel bagimliliklar
├── requirements-gpu.txt         # GPU / CodeBERT SSL bagimlilikları
│
├── preprocessing/
│   ├── tokenizer.py               # C/C++ tokenizer ve normalizer
│   ├── code_features.py           # AST / CF / semantik ozellik cikariımı
│   └── tree_sitter_parser.py      # Tree-sitter C++ parser (pre-compiled queries)
│
├── pairing/
│   └── pair_generator.py          # Kod cifti olusturma (O(1) hard mining)
│
├── vectorization/
│   ├── tfidf.py                   # Token/karakter TF-IDF vektorleyiciler
│   └── ssl_encoder.py             # CodeBERT embedding cikariımı (disk cache)
│
├── models/
│   ├── ensemble.py                # Feature-Partitioned Stacking Ensemble
│   └── xgboost.py                 # XGBoost model sarmalayici (GPU destekli)
│
├── utils/
│   ├── similarity_utils.py        # Jaccard/bigram helper'lar
│   ├── feature_pipeline.py        # Tekli cift icin ozellik cikariımı (demo/test)
│   ├── experiment_logger.py       # Deney kaydetme, metrik ve gorsellesirme
│   ├── hyperparameter_tuner.py    # Optuna ile hiperparametre arama
│   ├── test_automation.py         # Otomatik klon tipi basarim testi
│   ├── compare_experiments.py     # Deney metrik karsilastirma tablosu
│   ├── generate_test_clones.py    # Test klonu verisi olusturma
│   └── find_best_threshold.py     # Threshold optimizasyon araci
│
├── web_demo/
│   ├── app.py                     # FastAPI servisi (/predict + /predict_batch, SHAP)
│   └── index.html
│
├── data/poj104/                  # POJ-104 dataset (104 sinif, .txt dosyaları)
├── experiments/                  # Kaydedilen deneyler (exp_NNN_...)
└── test_results/                 # Otomasyon test sonuclari
```

## Kurulum

```bash
# Python 3.11 veya 3.12
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Temel bagimliliklar
pip install -r requirements.txt

# GPU / CodeBERT SSL ozellikleri icin (opsiyonel)
pip install -r requirements-gpu.txt
```

## Dataset Formatı

POJ-104 dataset'i `data/poj104/` altında şu yapıda olmalıdır:
- Her alt klasör bir problem sınıfıdır (104 sınıf: `1/`, `2/`, ..., `104/`)
- Her sınıfın altında `.txt` uzantılı C/C++ kaynak dosyaları bulunur

### Ensemble Mimarisi (Type-4 odaklı)
```bash
python main.py --pairs 800000 --model ensemble
```

### Hiperparametre Ayarı
```bash
python main.py --pairs 1000000 --tune --tune-trials 50
```

### Cross-Validation
```bash
python main.py --cv --cv-folds 5 --pairs 400000
```

## Klon Tipi Testi

```bash
# Test klonlarini olustur
python utils/generate_test_clones.py

# En son deneyle otomatik test
python utils/test_automation.py

# Belirli bir deneyle test
python utils/test_automation.py --exp-id 55 --threshold 0.95
```

## Deney Karsilastirma

```bash
# Tum deneyleri karsilastir
python utils/compare_experiments.py

# Belirli deneyleri karsilastir
python utils/compare_experiments.py --exp-ids 54 55 56

# Baska bir metrige gore sirala
python utils/compare_experiments.py --metric mcc
```

## Threshold Optimizasyonu

```bash
python utils/find_best_threshold.py
```

## Web Demo

```bash
# En son deneyi yukle
uvicorn web_demo.app:app --reload

# Belirli bir deneyi yukle
EXP_ID=55 uvicorn web_demo.app:app --reload

# Produksiyon CORS ayari
ALLOWED_ORIGINS="https://example.com" uvicorn web_demo.app:app
```

Demo `http://localhost:8000` adresinde acilir.

**API Endpoints:**
- `POST /predict` — Tek cift karsilastirma (SHAP aciklamali)
- `POST /predict_batch` — Toplu karsilastirma (maks. 500 cift)

## Deney Yönetimi

Her egitim otomatik olarak `experiments/exp_NNN_<model>_<pairs>k/` altina kaydedilir:
- `config.json` — Model ve vektorleyici parametreleri
- `metrics_train.json`, `metrics_test.json`, `metrics_val.json` — Basarim metrikleri
- `classification_report_*.txt` — Detayli siniflandirma raporu
- `confusion_matrix_*.png` — Karisiklik matrisi gorseli
- `model.pkl`, `tfidf.pkl`, `svd.pkl` — Egitimli model ve vektorleyiciler
- `stage1_model.pkl` — Stage-1 HistGBM modeli
- `ssl_pca.pkl` — SSL embedding PCA modeli (sadece `--use-ssl` ile)
- `notes.txt` — Timing ve tarih bilgisi

## Klon Tipleri

| Tip | Açıklama | Örnek |
|-----|----------|-------|
| Type 1 | Birebir kopyalar (yorum/boşluk farkı) | Kod kopyalanıp yorum eklenmiş |
| Type 2 | Yeniden adlandırılmış (değişken/fonksiyon isimleri) | `n` → `num_elements` |
| Type 3 | Yakın-kopya (küçük ekleme/çıkarma) | Gereksiz değişken eklenmesi |
| Type 4 | Semantik klon (farklı uygulama, aynı mantık) | İteratif vs özyinelemeli Fibonacci |

## Yapilandirma

Tum sabit degerler `config.py` icindedir:

```python
CASCADE_THRESHOLD         = 0.85   # Kural bazli klon esigi (token cosine)
CASCADE_STAGE1_THRESHOLD  = 0.95   # Stage-1 model esigi (HistGBM)
STAGE1_FEATURE_COUNT      = 32     # Stage-1'in kullandigi ozellik sayisi
SVD_N_COMPONENTS          = 50     # TruncatedSVD bilesenleri
TFIDF_MAX_FEATURES        = 500    # TF-IDF kelime hazinesi boyutu
ENSEMBLE_SVD_START_IDX    = 41     # Ensemble SVD sutun baslangici
SSL_PCA_COMPONENTS        = 64     # CodeBERT 768-D -> 64-D PCA indirgeme
DEFAULT_PAIRS             = 800_000
DEFAULT_SEED              = 42
```

## SSL Embedding Cache ve PCA

CodeBERT 768-D embeddingleri PCA ile `SSL_PCA_COMPONENTS=64` boyutuna indirgenir.
Her kod cifti icin `|emb_A - emb_B|` (64-D abs diff) feature vektorune eklenir.
Bu sayede model, kosinüs benzerligi gibi tek bir skalerden degil, semantik
uzayin her boyutundaki farktan ogrenir -> Type-4 dogrulugu ciddi artisi.

```bash
# Ilk calistirmada embedding'leri cikar, PCA fit et ve kaydet
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 200000

# Sonraki calistirmalarda cache'ten yukle (PCA da ssl_pca.pkl'den)
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 800000

# SSL ile tam gucte egitim (onceri cache hazir olmali)
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 800000 --model ensemble
```

> [!TIP]
> `SSL_PCA_COMPONENTS = 128` ile daha iyi Type-4 recall elde edilebilir (~400 MB).
> Deger `config.py`'de kolayca degistirilebilir.

## Unit Testler

```bash
pip install pytest
pytest tests/ -v
```
