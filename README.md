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
# Python 3.11 önerilir
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# GPU (NVIDIA CUDA) için opsiyonel:
pip install torch>=2.1.0
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
# Test klonlarını oluştur
python utils/generate_test_clones.py

# En son deneyle otomatik test
python utils/test_automation.py

# Belirli bir deneyle test
python utils/test_automation.py --exp-id 55 --threshold 0.95
```

## Threshold Optimizasyonu

```bash
python utils/find_best_threshold.py
```

## Web Demo

```bash
# En son deneyi yükle
uvicorn web_demo.app:app --reload

# Belirli bir deneyi yükle
EXP_ID=55 uvicorn web_demo.app:app --reload
```

Demo `http://localhost:8000` adresinde açılır.

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

## Yapılandırma

Tüm sabit değerler `config.py` içindedir:

```python
CASCADE_THRESHOLD = 0.85   # Kolay klon eşiği
SVD_N_COMPONENTS  = 50     # TruncatedSVD bileşenleri
TFIDF_MAX_FEATURES = 500   # TF-IDF kelime hazinesi boyutu
DEFAULT_PAIRS     = 800_000
DEFAULT_SEED      = 42
```

## Unit Testler

```bash
pip install pytest
pytest tests/ -v
```
