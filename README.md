# CodeDuplicationDetection

C/C++ kaynak kodları için **4 farklı klon tipini** (Type 1–4) tespit eden makine öğrenmesi tabanlı bir sistem.

## Proje Yapısı

```
CodeDuplicationDetection/
├── config.py                   # Merkezi sabitler ve magic number'lar
├── main.py                     # AKTİF EĞİTİM ALANI (Two-Stage Cascade Eğitim Betiği)
├── main_deprecated.py          # DEPRECATED (Eski tek aşamalı mimari, kullanılmaz)
├── requirements.txt
│
├── preprocessing/
│   ├── tokenizer.py            # C/C++ tokenizer ve normalizer
│   └── code_features.py        # AST / CF / semantik özellik çıkarımı
│
├── pairing/
│   └── pair_generator.py       # Kod çifti oluşturma (hard mining dahil)
│
├── vectorization/
│   └── tfidf.py               # Token/karakter TF-IDF vektörleyiciler
│
├── models/
│   └── xgboost.py             # XGBoost model sarmalayıcı (GPU desteği)
│
├── utils/
│   ├── similarity_utils.py    # Paylaşılan Jaccard/bigram helper'ları
│   ├── feature_pipeline.py    # Tekli çift için özellik çıkarımı (demo/test)
│   ├── experiment_logger.py   # Deney kaydetme, metrik ve görselleştirme
│   ├── hyperparameter_tuner.py # Optuna ile hiperparametre arama
│   ├── test_automation.py     # Otomatik klon tipi başarım testi
│   ├── generate_test_clones.py # Test klonu verisi oluşturma
│   └── find_best_threshold.py # Threshold optimizasyon aracı
│
├── web_demo/
│   ├── app.py                 # FastAPI servisi (SHAP açıklamalı)
│   └── index.html
│
├── data/
│   └── poj104/               # POJ-104 dataset (104 sınıf, .txt dosyaları)
│       ├── 1/
│       ├── 2/
│       └── ...
│
├── experiments/               # Kaydedilen deneyler (exp_NNN_...)
└── test_results/              # Otomasyon test sonuçları
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

Her eğitim otomatik olarak `experiments/exp_NNN_<model>_<pairs>k/` altına kaydedilir:
- `config.json` — Model ve vektörleyici parametreleri
- `metrics_train.json`, `metrics_test.json`, `metrics_val.json` — Başarım metrikleri
- `classification_report_*.txt` — Detaylı sınıflandırma raporu
- `confusion_matrix_*.png` — Karışıklık matrisi görseli
- `model.pkl`, `tfidf.pkl` — Eğitimli model ve vektörleyici
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
CASCADE_THRESHOLD         = 0.85   # Kolay klon esigi (token cosine)
CASCADE_STAGE1_THRESHOLD  = 0.95   # Stage-1 model esiği (HistGBM)
STAGE1_FEATURE_COUNT      = 32     # Stage-1'in kullandigi ozellik sayisi
SVD_N_COMPONENTS          = 50     # TruncatedSVD bilesenleri
TFIDF_MAX_FEATURES        = 500    # TF-IDF kelime hazinesi boyutu
ENSEMBLE_SVD_START_IDX    = 41     # Ensemble SVD sutun baslangici
DEFAULT_PAIRS             = 800_000
DEFAULT_SEED              = 42
```

## SSL Embedding Cache (opsiyonel)

```bash
# Ilk calistirmada embedding'leri cikar ve cache'e kaydet
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 200000

# Sonraki calistirmalarda cache'ten yukle
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 800000
```

## Unit Testler

```bash
pip install pytest
pytest tests/ -v
```
