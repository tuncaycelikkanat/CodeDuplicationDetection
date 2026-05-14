# 🚀 CodeDuplicationDetection — Komut Referansı

> Tüm komutlar proje kökünden (`/home/tuncay/PycharmProjects/CodeDuplicationDetection`) çalıştırılır.
> venv aktif olmalı: `source .venv/bin/activate`

---

## 1. ~~Standart Eğitim — `main_deprecated.py`~~ ⚠️ DEPRECATED

> **Bu dosya artık kullanılmıyor.** Yerine `main_deprecated.py` kullanın.
> `main_deprecated.py` çalıştırılırsa hata verir ve çıkar.

---

## 2. Ana Proje — `main.py`

> Aktif eğitim betiği. Dense (89-boyutlu) özellik vektörü, Cascade filtresi, XGBoost veya Ensemble.

### Temel çalıştırma
```bash
python main.py
python main.py --pairs 500000
python main.py --pairs 1000000
```

### Model seçimi
```bash
# Varsayılan: XGBoost
python main.py --pairs 1000000 --model xgboost

# Stacking Ensemble (XGBoost + RandomForest + HistGradientBoosting)
python main.py --pairs 1000000 --model ensemble
python main.py --pairs 500000 --model ensemble --device cpu
```

### Sınıf dengesi (positive-ratio)
```bash
# Dengeli eğitim (eski laboratuvar senaryosu: %50 klon, %50 klon-değil)
python main.py --positive-ratio 0.5

# Gerçekçi dengesiz sınıf dağılımı (varsayılan: %5 klon)
python main.py --positive-ratio 0.05

# Precision odaklı eğitim (%20 klon)
python main.py --positive-ratio 0.2
```

### Transformer SSL Entegrasyonu
```bash
# CodeBERT üzerinden derin öğrenme özellikleri çıkarma (768-boyut)
python main.py --pairs 500000 --use-ssl --device cuda
```

### Device
```bash
python main.py --pairs 1000000 --device cuda
python main.py --pairs 1000000 --device cpu
```

### Hiperparametre ayarı (sadece XGBoost)
```bash
python main.py --pairs 1000000 --tune
python main.py --pairs 1000000 --tune --tune-trials 50
python main.py --pairs 2000000 --tune --tune-trials 100 --device cuda
```

### Cross-Validation (Cascade filtreliyle)
```bash
python main.py --cv
python main.py --cv --cv-folds 5
python main.py --cv --cv-folds 5 --cv-pairs 200000 --model ensemble
```

### Tam kombinasyonlar
```bash
# Standart cascade eğitimi
python main.py --pairs 1000000 --device auto --seed 42

# Ensemble + gerçekçi sınıf dengesi
python main.py --pairs 1000000 --model ensemble --positive-ratio 0.2

# Büyük veri + GPU + tune
python main.py --pairs 2000000 --device cuda --tune --tune-trials 50

# Hızlı deneme
python main.py --pairs 5000 --device cpu
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
| ~~Eski eğitim~~ | `main_deprecated.py` **DEPRECATED** |
| Cascade eğitimi (XGBoost) | `python main.py --pairs 1000000` |
| Cascade eğitimi (Ensemble) | `python main.py --pairs 1000000 --model ensemble` |
| SSL Eğitimi (CodeBERT) | `python main.py --pairs 500000 --use-ssl` |
| Gerçekçi sınıf dengesi | `... --positive-ratio 0.05` |
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

### `main.py`
| Parametre | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `--model` | str | `xgboost` | `xgboost` veya `ensemble` |
| `--dataset` | str | `data/poj104` | Dataset dizini |
| `--pairs` | int | `800000` | Üretilecek çift sayısı |
| `--positive-ratio` | float | `0.05` | Klon çifti oranı (0.05 = gerçekçi dünya senaryosu) |
| `--use-ssl` | flag | — | CodeBERT gömülü (embedding) özelliklerini açar |
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
