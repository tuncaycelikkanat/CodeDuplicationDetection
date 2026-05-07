# 🚀 CodeDuplicationDetection — Komut Referansı

> Tüm komutlar proje kökünden (`/home/tuncay/PycharmProjects/CodeDuplicationDetection`) çalıştırılır.
> venv aktif olmalı: `source .venv/bin/activate`

---

## 1. ~~Standart Eğitim — `main.py`~~ ⚠️ DEPRECATED

> **Bu dosya artık kullanılmıyor.** Yerine `cascade_experiment/cascade_main.py` kullanın.
> `main.py` çalıştırılırsa hata verir ve çıkar.

---

## 2. Cascade Mimarisi — `cascade_experiment/cascade_main.py`

> Aktif eğitim betiği. Dense (89-boyutlu) özellik vektörü, Cascade filtresi, XGBoost veya Ensemble.

### Temel çalıştırma
```bash
python cascade_experiment/cascade_main.py
python cascade_experiment/cascade_main.py --pairs 500000
python cascade_experiment/cascade_main.py --pairs 1000000
```

### Model seçimi
```bash
# Varsayılan: XGBoost
python cascade_experiment/cascade_main.py --pairs 1000000 --model xgboost

# Stacking Ensemble (XGBoost + RandomForest + HistGradientBoosting)
python cascade_experiment/cascade_main.py --pairs 1000000 --model ensemble
python cascade_experiment/cascade_main.py --pairs 500000 --model ensemble --device cpu
```

### Sınıf dengesi (positive-ratio)
```bash
# Dengeli eğitim (varsayılan: %50 klon, %50 klon-değil)
python cascade_experiment/cascade_main.py --positive-ratio 0.5

# Gerçekçi dengesiz sınıf dağılımı (%10 klon)
python cascade_experiment/cascade_main.py --positive-ratio 0.1

# Precision odaklı eğitim (%20 klon)
python cascade_experiment/cascade_main.py --positive-ratio 0.2
```

### Device
```bash
python cascade_experiment/cascade_main.py --pairs 1000000 --device cuda
python cascade_experiment/cascade_main.py --pairs 1000000 --device cpu
```

### Hiperparametre ayarı (sadece XGBoost)
```bash
python cascade_experiment/cascade_main.py --pairs 1000000 --tune
python cascade_experiment/cascade_main.py --pairs 1000000 --tune --tune-trials 50
python cascade_experiment/cascade_main.py --pairs 2000000 --tune --tune-trials 100 --device cuda
```

### Cross-Validation (Cascade filtreliyle)
```bash
python cascade_experiment/cascade_main.py --cv
python cascade_experiment/cascade_main.py --cv --cv-folds 5
python cascade_experiment/cascade_main.py --cv --cv-folds 5 --cv-pairs 200000 --model ensemble
```

### Tam kombinasyonlar
```bash
# Standart cascade eğitimi
python cascade_experiment/cascade_main.py --pairs 1000000 --device auto --seed 42

# Ensemble + gerçekçi sınıf dengesi
python cascade_experiment/cascade_main.py --pairs 1000000 --model ensemble --positive-ratio 0.2

# Büyük veri + GPU + tune
python cascade_experiment/cascade_main.py --pairs 2000000 --device cuda --tune --tune-trials 50

# Hızlı deneme
python cascade_experiment/cascade_main.py --pairs 5000 --device cpu
```

---

## 3. Test Otomasyonu — `utils/test_automation.py`

> Eğitimli modeli tüm klon tipleri (Type1–4) üzerinde test eder.

### En son deneyi test et
```bash
python utils/test_automation.py
```

### Belirli deneyği test et
```bash
python utils/test_automation.py --exp-id 55
python utils/test_automation.py --exp-id 56
```

### Threshold değiştir
```bash
python utils/test_automation.py --threshold 0.90
python utils/test_automation.py --threshold 0.95   # varsayılan
python utils/test_automation.py --threshold 0.85
python utils/test_automation.py --threshold 0.50
```

### Kombinasyonlar
```bash
# Cascade deneyi, 0.90 eşiğiyle
python utils/test_automation.py --exp-id 56 --threshold 0.90

# En son deney, düşük eşik
python utils/test_automation.py --threshold 0.80

# Belirli deney, yüksek precision için yüksek eşik
python utils/test_automation.py --exp-id 54 --threshold 0.99
```

---

## 4. Test Klonu Verisi Oluştur — `utils/generate_test_clones.py`

> `test_clones/` dizininde Type1–4 klon çiftleri üretir.

```bash
# Varsayılan dizine oluştur
python utils/generate_test_clones.py

# Özel çıktı dizini
python utils/generate_test_clones.py --output test_clones_v2

# Mevcut dizini sıfırla ve yeniden oluştur
python utils/generate_test_clones.py --overwrite
```

---

## 5. Threshold Optimizasyonu — `utils/find_best_threshold.py`

> En iyi F1 ve MCC değerlerini veren threshold değerini bulur.
> Önce `test_automation.py` çalıştırılmış olmalı.

```bash
python utils/find_best_threshold.py
```

---

## 6. Web Demo — `web_demo/app.py`

### Varsayılan (en son deney)
```bash
uvicorn web_demo.app:app --reload
```

### Belirli deney
```bash
EXP_ID=55 uvicorn web_demo.app:app --reload
EXP_ID=56 uvicorn web_demo.app:app --reload
```

### Port değiştir
```bash
uvicorn web_demo.app:app --reload --port 8080
EXP_ID=56 uvicorn web_demo.app:app --reload --port 8000
```

### Production modu (reload yok)
```bash
uvicorn web_demo.app:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 7. Unit Testler ve Integration Testleri — `tests/`

```bash
# Tüm testleri çalıştır (41 test)
python -m pytest tests/ -v

# Sadece similarity testleri
python -m pytest tests/test_similarity_utils.py -v

# Sadece tokenizer testleri
python -m pytest tests/test_tokenizer.py -v

# Sadece feature pipeline testleri (dense mimari)
python -m pytest tests/test_feature_pipeline.py -v

# Ensemble + Cascade integration testleri
python -m pytest tests/test_ensemble_pipeline.py -v

# Hata çıktısını göster
python -m pytest tests/ -v --tb=short

# Belirli test sınıfı
python -m pytest tests/test_ensemble_pipeline.py::TestEnsemblePipeline -v

# Belirli tek test
python -m pytest tests/test_tokenizer.py::TestNormalizeTokens::test_keyword_preserved -v
```

---

## 8. Import / Sağlık Kontrolü

```bash
# Tüm modülleri import et ve kontrol et
python -c "
from config import CASCADE_THRESHOLD
from utils.similarity_utils import _jaccard_sim
from vectorization.tfidf import build_tfidf_vectorizer
from preprocessing.tokenizer import tokenize, normalize_tokens
from utils.feature_pipeline import build_pair_vector
from pairing.pair_generator import generate_pairs
from models.xgboost import build_xgboost
print('✅ Tüm modüller OK')
"

# Config değerlerini gör
python -c "import config; [print(f'{k}={v}') for k,v in vars(config).items() if not k.startswith('_')]"
```

---

## 9. Hızlı Referans Tablosu

| Görev | Komut |
|---|---|
| ~~Eski eğitim~~ | `main.py` **DEPRECATED** |
| Cascade eğitimi (XGBoost) | `python cascade_experiment/cascade_main.py --pairs 1000000` |
| Cascade eğitimi (Ensemble) | `python cascade_experiment/cascade_main.py --pairs 1000000 --model ensemble` |
| Gerçekçi sınıf dengesi | `... --positive-ratio 0.1` |
| GPU eğitimi | `... --device cuda` |
| Tune + eğitim | `... --tune --tune-trials 50` |
| Cross-validation | `... --cv --cv-folds 5` |
| Test otomasyonu | `python utils/test_automation.py --exp-id 56 --threshold 0.90` |
| Web demo başlat | `EXP_ID=56 uvicorn web_demo.app:app --reload` |
| Test klonu oluştur | `python utils/generate_test_clones.py` |
| Threshold bul | `python utils/find_best_threshold.py` |
| Tüm testler | `python -m pytest tests/ -v` |
| Integration test | `python -m pytest tests/test_ensemble_pipeline.py -v` |

---

## 10. Tüm Parametreler Referansı

### `cascade_experiment/cascade_main.py`
| Parametre | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `--model` | str | `xgboost` | `xgboost` veya `ensemble` |
| `--dataset` | str | `data/poj104` | Dataset dizini |
| `--pairs` | int | `800000` | Üretilecek çift sayısı |
| `--positive-ratio` | float | `0.5` | Klon çifti oranı (0.1 = gerçekçi) |
| `--test-size` | float | `0.2` | Test ayrım oranı |
| `--seed` | int | `42` | Rastgele tohum |
| `--tune` | flag | — | Optuna tuning aç (sadece xgboost) |
| `--tune-trials` | int | `30` | Optuna deneme sayısı |
| `--device` | str | `auto` | `cpu` / `cuda` / `xpu` / `auto` |
| `--cv` | flag | — | Cross-validation modu |
| `--cv-folds` | int | `5` | K-fold sayısı |
| `--cv-pairs` | int | `None` | CV fold başına çift sayısı |

### `utils/test_automation.py`
| Parametre | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `--exp-id` | int | `None` (en son) | Test edilecek deney ID'si |
| `--threshold` | float | `0.95` | Sınıflandırma eşiği |

### `web_demo/app.py` (env vars)
| Değişken | Açıklama |
|---|---|
| `EXP_ID` | Yüklenecek deney numarası (yoksa en son) |
