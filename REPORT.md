# Code Duplication Detection — Teknik Rapor

**Proje:** CodeDuplicationDetection  
**Tarih:** Mayıs 2026  
**Son Güncelleme:** Mayıs 2026 (kapsamlı kod kalitesi ve güvenlik iyileştirmeleri)  
**Veri Seti:** POJ-104 (C/C++ kaynak kodu, 104 problem sınıfı, ~52.000 dosya)  
**En İyi Model:** CASCADE Ensemble (Feature-Partitioned Stacking) / CASCADE XGBoost — Global F1: ~%92+ | Type 1-2-3 Recall: %100 | Type 4 Precision: Yüksek

---

## 1. Proje Genel Akışı

Sistem, iki C/C++ kod parçasını karşılaştırarak aralarında anlamsal (semantik) benzerlik bulunup bulunmadığını tahmin eden bir ikili sınıflandırma (binary classification) sistemidir. Tahmin edilen sınıflar şunlardır:

- **1 (Duplicate / Clone):** İki kod aynı problemi çözmekte; kaynak kodu, mantık veya algoritma düzeyinde benzeşmektedir.
- **0 (Not Duplicate):** İki kod farklı problemlere aittir.

```text
Ham Kaynak Kod (C/C++)
        │
        ▼
[1] Tokenizasyon + Normalizasyon (preprocessing/tokenizer.py)
        │
        ▼
[2] Yapısal & Semantik Özellik Çıkarımı (preprocessing/code_features.py)
    ├── Halstead, McCabe, Karmaşıklık, Yoğunluk (Density) Metrikleri
    ├── Kod İskeleti (Skeleton), Soyut Kontrol Akışı (Abstract CF)
    └── Algoritmik Parmak İzi (Kütüphane çağrıları, veri yapıları, IO pattern)
        │
        ▼
[3] TF-IDF Vektörizasyon & SVD (vectorization/tfidf.py)
    Token-level TF-IDF (1-3 gram) ve TruncatedSVD ile 50 boyuta sıkıştırma
        │
        ▼
[4] Veri Seti Bölme — Kod düzeyinde, veri sızıntısı önleyici
    Train %64 | Validation %16 | Test %20 (stratified)
        │
        ▼
[5] Çift (Pair) Üretimi (pairing/pair_generator.py)
    Train: %70 | Val: %15 | Test: %15
    ├── Dengeli veya gerçekçi veri dağılımı (positive_ratio)
    ├── Hard Negative Mining (%30 oranında — benzer uzunlukta farklı sınıflar)  [O(1) dict ile]
    └── Hard Positive Mining (%30 oranında — Type-4 zorlaması, en farklı uzunlukta aynı sınıflar)
        │
        ▼
[6] Vektör Oluşturma (Dense Feature Array)
    [cos_token | length_ratio | manhattan | euclidean | AST Ratios & Diffs | CF Sim | Semantic Jaccards | SVD Diffs]
    Feature sırası config.py'de (STAGE1_FEATURE_COUNT, ENSEMBLE_SVD_START_IDX) belgelenmiştir.
    (Toplam ~91 yoğun özellik. Seyrek TF-IDF fark matrisi kaldırılmıştır.)
        │
        ▼
[7] Model Eğitimi (Two-Stage Cascade Mimarisi)
    ├── Stage-1: HistGradientBoosting (İlk STAGE1_FEATURE_COUNT=32 özellik ile eğitilir)
    ├── Kaskad Filtreleme: _apply_cascade_filter() ile Stage-1'in kolayca klon
    │   bulduğu (CASCADE_STAGE1_THRESHOLD=0.95+ prob) örnekler eğitim setinden silinir.
    └── Stage-2 (Uzman): XGBoost veya Stacking Ensemble (LightGBM + RF + LinearSVC)
        "Zor/Type-4" klonlara odaklanır.
        │
        ▼
[8] Değerlendirme & Deney Kayıt (utils/test_automation.py)
    Type 1-2-3-4 bazlı ayrık değerlendirme, Precision, Recall, F1, MCC, AUC-ROC
        │
        ▼
[9] Çıktı
    experiments/exp_NNN_CASCADE_Model_Xk/
```

---

## 2. İki Aşamalı (Two-Stage) Cascade Mimarisi ve Modeller

Projenin en büyük evrimi, basit tek aşamalı tahminden **İki Aşamalı (Two-Stage)** filtreleme ve sınıflandırma sistemine geçişidir. Eski mimariyi içeren `main_deprecated.py` projeden **kaldırılmıştır**; eğitim tamamen `main.py` (Cascade) içerisinde yönetilmektedir.

### Aşama 1: Lexical Filter (Kolay Klon Yakalayıcı)

Model eğitiminin başında, ilk `STAGE1_FEATURE_COUNT = 32` leksikal ve yapısal özellik (Token Kosinüs Benzerliği, Uzunluk Oranı, Manhattan/Öklid mesafeleri, AST ratios/diffs, CF sim) kullanılarak hızlı bir **HistGradientBoosting** (Stage-1) eğitilir.

- **Amacı:** Eğitim verisindeki birbirine çok benzeyen Type-1/2/3 klonları `CASCADE_STAGE1_THRESHOLD = 0.95` olasılık eşiğiyle tespit edip Stage-2'nin eğitim setinden çıkarmak.
- **Uygulama:** `_apply_cascade_filter(X, y, stage1_model, threshold, feature_count)` merkezi helper fonksiyonu ile hem CV hem de normal eğitim modunda paylaşılır.
- **Faydası:** Karmaşık modelin (Stage-2) kolay örnekleri ezberlemesi engellenir; model yalnızca **Type-4** klonları öğrenmeye zorlanır. Çıkarım (inference) sırasında `X[:, :STAGE1_FEATURE_COUNT]` slice'ı kullanılır — bu değer tüm kodlarda config'den gelir.

### Aşama 2: Type-4 Uzmanı (XGBoost veya Ensemble)

Zorlu örnekler üzerinde iki farklı uzman model kullanılabilir:

1. **GPU XGBoost (`models/xgboost.py`):** Derin ağaçlarla (`max_depth=10`) semantik örüntüleri yakalayan geleneksel uzman.
2. **Feature-Partitioned Stacking Ensemble (`models/ensemble.py`):**  
   Özellik gruplarının ilgili uzman modellere dağıtıldığı mimari. SVD başlangıç indeksi `ENSEMBLE_SVD_START_IDX` sabiti ile `config.py`'den alınır; `build_ensemble(svd_start_idx=...)` parametresi ile çalışma zamanında da geçersiz kılınabilir:
   - **LightGBM (HistGradientBoosting):** Lexical (0–3) + SVD (ENSEMBLE_SVD_START_IDX+) özellikleri
   - **Random Forest:** Yapısal / AST+CF (4–32) özellikleri
   - **LinearSVC (Calibrated):** Semantik Jaccard / tip profili (33–40) özellikleri
   - **Meta-Classifier (Logistic Regression):** Üç modelin olasılıklarını birleştirerek nihai kararı verir.

---

## 3. Özellik Mühendisliği (Feature Engineering)

Proje, bellek tüketen seyrek (sparse) TF-IDF fark matrisini bırakıp tamamen yoğun (dense) özellik vektörlerine geçmiştir. Feature sırası `config.py`'de belgelenmiş ve tüm modüllerde sabitlerle yönetilmektedir.

### 3.1 Feature Vektörü Düzeni

```
İndeks   Açıklama
-------  ------------------------------------------------------------------
[0]      cos_token         — Token kosinüs benzerliği (Stage-1 başvurusu)
[1]      length_ratio      — min/max token uzunluk oranı
[2]      manhattan_token   — TF-IDF fark L1 normu
[3]      euclidean_token   — TF-IDF fark L2 normu
[4..17]  AST ratios        — 14 yapısal metrik oranı (min/max)
[18..31] AST diffs         — 14 yapısal metrik mutlak farkı
[32]     cf_sim            — Kontrol akışı desen benzerliği
[33..38] Semantic Jaccard  — lib, lib_cat, data_struct, io, math, skeleton
[39]     abstract_cf       — Soyut kontrol akışı Levenshtein benzerliği
[40]     type_profile_cos  — Değişken tip profili kosinüs benzerliği
[41..]   svd_diff          — SVD bileşen farklarıı (SVD_N_COMPONENTS=50 boyut)
```

### 3.2 Lexical (Yüzeysel) Özellikler
- Token tabanlı kosinüs benzerliği (`cos_token`)
- Kod uzunluk oranı (`length_ratio`)
- TF-IDF mutlak farklarının L1 (Manhattan) ve L2 (Öklid) normları

### 3.3 AST ve Karmaşıklık Özellikleri
Regex ile soyutlanmış hızlı metriklerin oranları (`min/max`) ve farkları (`abs(a - b)`):
- Döngü, dal (branch), fonksiyon çağrısı, operatör ve parametre yoğunlukları
- **Halstead Hacmi / Zihinsel Çaba:** Algoritmik karmaşıklık ve satır yoğunluğu
- **McCabe Karmaşıklığı:** Karar noktaları yoğunluğu

### 3.4 Semantik "Algoritmik Parmak İzi" (Type-4 İçin Kritik)
Jaccard/Kosinüs benzerlikleri ile kodun niyetini okuyan özellikler:
- **Kütüphane / Kategori Benzerliği:** `printf`, `malloc`, `sort` gibi çağrıların Jaccard benzerliği
- **Veri Yapıları:** Array, stack, queue, map, vector kullanım benzerliği
- **IO Deseni:** Girdi/çıktı sırasının (`I-O-I-O`) string bigram eşleşmesi
- **Matematiksel Operatörler:** `+`, `-`, `*`, `%` kümesinin benzerliği
- **Kontrol Akış İskeleti & Abstract CF:** Saf döngü/şart dizilimlerinin Levenshtein mesafesi
- **Değişken Tip Profili:** int/float/char/pointer/array kullanım oranlarının kosinüs benzerliği

### 3.5 SVD Boyut İndirgeme
Token seviyesindeki TF-IDF temsili, **TruncatedSVD** ile `SVD_N_COMPONENTS = 50` boyuta düşürülür. Çiftler arasındaki SVD bileşen farkları gürültüden arındırılmış semantik sinyal olarak modele verilir.

---

## 4. Çift Üretimi ve Hard Mining (`pairing/pair_generator.py`)

Sistem eğitim setini hazırlarken NumPy tabanlı vektörize operasyonlar ve `joblib` paralel işleme kullanır. Hard mining döngüsünde etiket araması O(1)'e indirilmiştir:

```python
label_to_idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(unique_labels)}
# Önceden: unique_labels.index(src_lbl)  → O(n_labels) her iterasyonda
# Şimdi:   label_to_idx[src_lbl]         → O(1)
```

- **Hard Negative Mining (%30):** En yakın uzunluktaki farklı sınıf kodu seçilerek aldatıcı negatifler
- **Hard Positive Mining (%30):** En farklı uzunluktaki aynı sınıf kodu ile Type-4 zorlaması
- **Esnek Positive Ratio:** `positive_ratio=0.5` (dengeli) veya `0.1` (gerçekçi dağılım)

---

## 5. Yapılandırma (`config.py`)

Tüm kritik sabitler merkezi olarak `config.py`'de tanımlıdır — kodun herhangi bir yerinde hardcoded değer kullanılmaz:

| Sabit | Değer | Açıklama |
|---|---|---|
| `CASCADE_THRESHOLD` | 0.85 | Token kosinüs eşiği (kural bazlı ön filtre) |
| `CASCADE_STAGE1_THRESHOLD` | 0.95 | Stage-1 model eşiği (HistGBM) |
| `STAGE1_FEATURE_COUNT` | 32 | Stage-1'in kullandığı özellik sayısı |
| `SVD_N_COMPONENTS` | 50 | TruncatedSVD bileşen sayısı |
| `TFIDF_MAX_FEATURES` | 500 | TF-IDF kelime hazinesi boyutu |
| `ENSEMBLE_SVD_START_IDX` | 41 | Ensemble'da SVD bloğunun başlangıç indeksi |
| `DEFAULT_PAIRS` | 800_000 | Varsayılan çift sayısı |
| `DEFAULT_SEED` | 42 | Rastgele tohum (üretilebilirlik) |

---

## 6. Değerlendirme & Test Otomasyonu

**`utils/test_automation.py`**

Sistem, test setini tüm klon tiplerine (Type 1, 2, 3, 4) ayırarak bağımsız test eder. Cascade inference her iki noktada da `STAGE1_FEATURE_COUNT` ve `CASCADE_STAGE1_THRESHOLD` config sabitlerini kullanır. Raporlar `test_results/run_<timestamp>/` altında JSON, metin ve Karışıklık Matrisi formatlarında arşivlenir.

**`utils/find_best_threshold.py`** — F1 ve MCC'yi maksimize eden ideal eşik belirlenir.

**`utils/compare_experiments.py`** — Birden fazla deneyin metriklerini yan yana karşılaştırır:

```bash
python utils/compare_experiments.py --exp-ids 54 55 56
python utils/compare_experiments.py --metric mcc
```

| Klon Tipi | Yaklaşım ve Başarım |
|---|---|
| **Type-1** (Birebir Kopya) | Stage-1 Lexical filtre veya Kaskad eşiği (cos_sim > 0.85) anında **%100** doğrulukla yakalar. |
| **Type-2** (Değişken Adı Farklı) | Token normalizasyonu (VAR/FUNC/NUM) sayesinde Stage-1 **%100** oranında işaretler. |
| **Type-3** (Satır Ekleme/Silme) | Levenshtein ve CF desen benzerliği ile Stage-1 büyük çoğunluğu yakalar. |
| **Type-4** (Farklı Algoritma) | Yalnızca Hard Mining ile eğitilmiş Stage-2 (XGBoost / Ensemble) devreye girer. **%50+ Recall ve Yüksek Precision** dengesi sağlanır. |

---

## 7. Web Demo (`web_demo/app.py`)

FastAPI tabanlı demo arayüzü SHAP TreeExplainer ile açıklanabilir tahmin sunar.

**API Endpoints:**
- `POST /predict` — Tek çift karşılaştırma, SHAP açıklamalı
- `POST /predict_batch` — Toplu karşılaştırma (maks. 500 çift / istek)

**Güvenlik:**
- CORS kaynakları `ALLOWED_ORIGINS` env değişkeni ile kontrol edilir
- `ALLOWED_ORIGINS=https://example.com uvicorn web_demo.app:app`

**SSL Singleton:** CodeBERT pipeline uygulama başlangıcında bir kez yüklenir; her request'te yeniden yüklenmez.

---

## 8. SSL Embedding Cache

CodeBERT embedding çıkarımı büyük datasetlerde saatlerce sürebilir. `--ssl-cache` parametresi ile sonuçlar diske kaydedilir:

```bash
# İlk çalıştırma — çıkar ve kaydet
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 200000

# Sonraki çalıştırmalar — cache'ten yükle
python main.py --use-ssl --ssl-cache ssl_cache.npy --pairs 800000
```

Cache boyut uyuşmazlığı otomatik olarak tespit edilir ve yeniden extraction yapılır.

---

## 9. Test Kapsamı

**Son Test Sonucu: `54 / 54 PASSED ✅`**

| Test Dosyası | Test Sayısı | Kapsam |
|---|---|---|
| `test_tokenizer.py` | 11 | Tokenizer normalizasyonu |
| `test_similarity_utils.py` | 13 | Jaccard / bigram benzerlik |
| `test_feature_pipeline.py` | 12 | `build_pair_vector` + SVD + edge case |
| `test_ensemble_pipeline.py` | 5 | Cascade + ensemble fit/predict |
| `test_edge_cases.py` | 13 | Sinir durumları (positive_ratio, boş kod, determinizm, cascade helper) |

---

## 10. Dizin Yapısı ve Modüller

```text
CodeDuplicationDetection/
├── config.py                      # Merkezi sabitler (CASCADE_STAGE1_THRESHOLD, STAGE1_FEATURE_COUNT, ...)
├── main.py                        # AKTİF EĞİTİM ALANI (Two-Stage Cascade Eğitim Betiği)
├── requirements.txt               # Temel bağımlılıklar (Python 3.11/3.12)
├── requirements-gpu.txt           # GPU / CodeBERT SSL bağımlılıkları (torch, transformers)
├── models/
│   ├── ensemble.py                # Feature-Partitioned Stacking Ensemble (svd_start_idx parametreli)
│   └── xgboost.py                 # GPU XGBoost Mimarisi
├── pairing/
│   └── pair_generator.py          # Vektörize Çift ve Hard-Mining Üreticisi (O(1) label lookup)
├── preprocessing/
│   ├── tokenizer.py               # Kod tokenizasyon ve VAR/FUNC/NUM normalizasyonu
│   ├── code_features.py           # AST, Metric ve Semantik özellikler
│   └── tree_sitter_parser.py      # Tree-sitter C/C++ parser (pre-compiled queries, singleton)
├── vectorization/
│   ├── tfidf.py                   # Token ve Karakter TF-IDF (SVD destekli)
│   └── ssl_encoder.py             # CodeBERT embedding çıkarımı (disk cache destekli)
├── utils/
│   ├── feature_pipeline.py        # Çıkarım (Inference) için tekil çift özellik oluşturucu
│   ├── test_automation.py         # Type 1-4 ayrık model test aracı
│   ├── find_best_threshold.py     # Optima eşik arayıcı (F1 & MCC)
│   ├── compare_experiments.py     # Deney metrik karşılaştırma tablosu [YENİ]
│   ├── similarity_utils.py        # Merkezi Jaccard/N-Gram benzerlik fonksiyonları
│   ├── experiment_logger.py       # Deney kaydı, CV sonuçları, config.json oluşturucu
│   └── hyperparameter_tuner.py    # Optuna ile XGBoost/RF/LinearSVM hiperparametre arama
├── web_demo/
│   ├── app.py                     # FastAPI servisi (/predict + /predict_batch, SHAP açıklamalı)
│   └── index.html                 # Web arayüzü
├── tests/                         # Unit ve Integration testleri (54 test)
└── commands.md                    # Komut Referansı (Nasıl Çalıştırılır?)
```

> **Nasıl Çalıştırılır?**  
> Tüm işlemler `commands.md` dosyasında detaylandırılmıştır.  
> Hızlı bir Ensemble eğitim başlatmak için:
> ```bash
> python main.py --pairs 1000000 --model ensemble
> ```

---

*Bu rapor, CodeDuplicationDetection projesinin "Two-Stage Cascade ve Feature-Partitioned Stacking Ensemble" mimarisini ve Mayıs 2026'da gerçekleştirilen kapsamlı kod kalitesi / güvenlik iyileştirmelerini yansıtmaktadır.*
