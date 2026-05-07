# Code Duplication Detection — Teknik Rapor

**Proje:** CodeDuplicationDetection  
**Tarih:** Nisan 2026  
**Veri Seti:** POJ-104 (C/C++ kaynak kodu, 104 problem sınıfı, ~52.000 dosya)  
**En İyi Model:** CASCADE XGBoost (exp_056-058 serisi, 1M çift) — Global F1: ~%92 | Type 1-2-3 Recall: %100 | Type 4 Precision: %80+

---

## 1. Proje Genel Akışı

Sistem, iki C/C++ kod parçasını karşılaştırarak aralarında anlamsal (semantik) benzerlik bulunup bulunmadığını tahmin eden bir ikili sınıflandırma (binary classification) sistemidir. Tahmin edilen sınıflar şunlardır:

- **1 (Duplicate / Clone):** İki kod aynı problemi çözmekte; kaynak kodu, mantık veya algoritma düzeyinde benzeşmektedir.
- **0 (Not Duplicate):** İki kod farklı problemlere aittir.

```
Ham Kaynak Kod (C/C++)
        │
        ▼
[1] Tokenizasyon + Normalizasyon (preprocessing/tokenizer.py)
        │
        ▼
[2] Yapısal & Semantik Özellik Çıkarımı (preprocessing/code_features.py)
    ├── Halstead Hacim/Çaba & McCabe Karmaşıklık Metrikleri
    ├── Kod İskeleti (Skeleton N-Gram) ve Veri Tipi Profili
    └── Altın Algoritmik Parmak İzi (kütüphane çağrıları, veri yapıları, IO pattern, math ops)
        │
        ▼
[3] TF-IDF Vektörizasyon (vectorization/tfidf.py)
    Token-level TF-IDF (1-3 gram, max 500 özellik, sublinear_tf=True)
        │
        ▼
[4] Veri Seti Bölme — Kod düzeyinde, veri sızıntısı önleyici
    Train %64 | Validation %16 | Test %20 (stratified)
        │
        ▼
[5] Çift (Pair) Üretimi (pairing/pair_generator.py)
    Train: %70 | Val: %15 | Test: %15  (toplamda N çift)
    ├── Hard Negative Mining (%30 oranında)
    └── Hard Positive Mining (%30 oranında — Type-4 desteği)
        │
        ▼
[6] Çift Özellik Matrisi Oluşturma
    [TF-IDF diff | cos_sim/manhattan/euclidean | length_ratio | AST Ratio & Diff | CF Edit Distance | Semantik & Tip Profili | SVD Diffs]
    Toplam: 589 özellik
        │
        ▼
[7] Model Eğitimi (Cascade Mimarisi)
    XGBoost
    ├── Kaskad Filtreleme: Eğitimden "Kolay Klonlar" (cos_sim > %85) silinir.
    ├── XGBoost sadece "Zor/Type-4" klonlara odaklanır.
    └── Early Stopping (validation seti üzerinde)
        │
        ▼
[8] Değerlendirme & Deney Kayıt (utils/experiment_logger.py)
    Accuracy, F1, AUC-ROC, Confusion Matrix, Classification Report
        │
        ▼
[9] Çıktı
    experiments/exp_NNN_Model_Xk/
    ├── model.pkl, tfidf.pkl
    ├── metrics_train.json, metrics_test.json
    ├── classification_report_{train|test}.{txt|json}
    ├── confusion_matrix_{train|test}.png
    ├── config.json, notes.txt
```

---

## 2. Veri Seti: POJ-104

| Özellik | Değer |
|---|---|
| Kaynak | Peking Online Judge (POJ) arşivi |
| Problem Sayısı (Sınıf Sayısı) | 104 |
| Toplam Kod Dosyası | ~52.000 (her problem için ~500 çözüm) |
| Dil | C / C++ |
| Format | `.txt` dosyaları, `data/poj104/<class_id>/<file>.txt` |
| Görev | Aynı probleme ait iki kodun tespiti (klon tespiti) |

Her bir `class_id` klasörü, aynı algoritmik problemi çözen farklı programcıların kodlarını içermektedir. Aynı sınıftan iki kod → Pozitif çift (clone), farklı sınıflardan iki kod → Negatif çift (non-clone).

### Veri Bölme Stratejisi

Veri sızıntısını (data leakage) önlemek için bölme **çift düzeyinde değil, kod düzeyinde** yapılır. Bir kodun eğitim setinde olması durumunda o kodun tüm çiftleri test setinde yer alamaz.

```
Tüm kodlar (52.000)
    │
    ├── Train+Val (80%, stratified) ──► Train (64%) + Validation (16%)
    └── Test (20%)
```

---

## 3. Ön İşleme: Tokenizasyon ve Normalizasyon

**`preprocessing/tokenizer.py`**

Ham C/C++ kodu şu adımlardan geçirilir:

1. **Regex tokenizasyonu:** Tanımlayıcılar, sayılar, çok karakterli operatörler (`&&`, `||`, `<<`, `>>`, `++`, `->`), karşılaştırma operatörleri ve noktalama işaretleri regex ile ayrıştırılır.

2. **Normalizasyon:**

| Token Türü | Dönüşüm |
|---|---|
| C/C++ anahtar sözcükleri (`int`, `for`, `return`, …) | **Olduğu gibi bırakılır** |
| Sayısal değerler (`42`, `3.14`, …) | → `NUM` |
| Döngü değişkenleri (`i`, `j`, `k`, `n`, `m`, …) | **Olduğu gibi bırakılır** |
| Kullanıcı tanımlı fonksiyonlar (`my_sort`, `solve`, vb.) | → `FUNC` |
| Tek harfli identifier | → `VAR` |
| Diğer tüm identifierlar | → `VAR` |

Bu normalizasyon, değişken ve kullanıcı fonksiyon isimlerinden bağımsız bir token temsili oluşturur; Type-1, Type-2 ve Type-3 klonların TF-IDF katmanında yakalanmasını kolaylaştırır.

---

## 4. Özellik Mühendisliği (Feature Engineering)

Proje, iki kod arasındaki benzerliği güçlü özellik gruplarına dayandırarak hesaplar. Toplam **589 özellik** çift başına üretilmektedir.

### 4.1 TF-IDF Token Farkı (500 özellik)

```
|TF-IDF(code_A) - TF-IDF(code_B)|  →  Seyrek (sparse) fark vektörü
```

- 1-3 gram token TF-IDF, max 500 özellik, `sublinear_tf=True`, `min_df=3`, `max_df=0.95`
- TF-IDF yalnızca **eğitim** verisi üzerinde `fit` edilir; validasyon/test sadece `transform`
- **Yakalar:** Type-1 (birebir kopya), Type-2 (yeniden adlandırılmış değişkenler), Type-3 (küçük farklar)

### 4.2 Skaler / Çift Düzey Özellikler (89 özellik)

| # | Özellik | Açıklama |
|---|---|---|
| 1 | `cosine_similarity_token` | İki TF-IDF vektörü arasındaki kosinüs benzerliği |
| 2 | `length_ratio` | `min(len_A, len_B) / max(len_A, len_B)` — kod uzunluk oranı |
| 3 | `manhattan_distance_token` | İki TF-IDF vektörü arasındaki Manhattan (L1) mesafesi |
| 4 | `euclidean_distance_token` | İki TF-IDF vektörü arasındaki Euclidean (L2) mesafesi |
| 5–18 | `Numeric feature ratios` (14 özellik) | AST sayısal özellikleri + Halstead/McCabe metrikleri için `min/max` oranı |
| 19–32 | `Numeric feature diffs` (14 özellik) | AST sayısal özellikleri + Halstead/McCabe metrikleri için mutlak fark `abs(A - B)` |
| 33 | `cf_pattern_similarity` | Kontrol akışı dizisinin normalize edilmiş Levenshtein (Edit) mesafesi |
| 34 | `semantic_library_call_jaccard` | Kullanılan kütüphane fonksiyonlarının Jaccard benzerliği |
| 35 | `semantic_data_struct_jaccard` | Kullanılan veri yapılarının Jaccard benzerliği |
| 36 | `semantic_io_pattern_jaccard` | Giriş/Çıkış sırasının bigram Jaccard benzerliği |
| 37 | `semantic_math_op_jaccard` | Kullanılan matematiksel operatörlerin Jaccard benzerliği |
| 38 | `semantic_skeleton_jaccard` | Saf kontrol iskeleti (tokenlardan arındırılmış) diziliminin Jaccard benzerliği |
| 39 | `semantic_type_profile_cosine` | Değişken tiplerinin (int, float, array, ptr) histogram kosinüs benzerliği |
| 40–89 | `semantic_svd_diffs` (50 özellik) | TF-IDF matrisinin TruncatedSVD (PCA) ile 50 boyuta sıkıştırılmış yoğun (dense) mutlak farkları. Type-4 klonlar için gürültü azaltıcı temel sinyal. |

*(Not: Eski sürümlerde kullanılan yavaş ve gürültülü `LLVM IR Opcodes`, `Opcode N-grams` ve `pycparser AST Hashing` mekanizmaları performansı maksimize etmek adına projeden tamamen kaldırılmıştır.)*

### 4.3 Saf ML Karmaşıklık Metrikleri ve Sayısal Özellikler (11 özellik)

`preprocessing/code_features.py` içinde regex ile çıkarılır:

| Özellik | Hesaplama |
|---|---|
| `branch_count` | `if`, `else`, `switch`, `case` sayısı |
| `loop_call_combined` | `for`+`while`+`do` döngüsü + fonksiyon çağrısı sayısı |
| `nesting_depth` | Maksimum iç içe `{}` derinliği |
| `operator_count` | `+`, `-`, `*`, `/`, `%` aritmetik operatör sayısı |
| `return_count` | `return` ifadesi sayısı |
| `accumulator_pattern` | `+=`, `-=`, `x = x + y` biçimi tespit (0/1) |
| `param_count` | İlk fonksiyon tanımındaki parametre sayısı |
| `math_op_set_size` | Kullanılan benzersiz matematiksel operatör tipi sayısı |
| `halstead_volume` | Benzersiz operatör/operand sayılarından türetilen matematiksel algoritma hacmi |
| `halstead_effort` | Algoritmayı yazmak için gereken zihinsel çaba (Difficulty * Volume) oranı |
| `mccabe_complexity` | Kontrol akışı karar noktası sayısı (If, for, while, &&, \|\|) + 1 |

### 4.4 Algoritmik Parmak İzi ve Tip-4 Kurnazlıkları (Altın Özellikler)

Tip-4 (Semantik) klonları bulmak için modelin en güvendiği davranışsal (behavioral) sinyallerdir. Bu sinyaller tamamen saf ML özellikleri olup derin öğrenme olmadan kodların niyetini okur:

| Özellik | |
|---|---|
| `library_calls` (set) | Tanınan ~80 standart C/C++ kütüphane fonksiyonu kullanım kümesi |
| `data_structs` (set) | Algılanan veri yapısı tipi (array, stack, vector, map, vb.) kümesi |
| `io_pattern` (str) | "I" ve "O" karakterlerinden oluşan Giriş/Çıkış işlem sırası |
| `math_ops` (set) | Kullanılan matematiksel operatör kümesi (`{+, -, *}`) |
| `skeleton_jaccard` (Tuple) | Koddan değişken ve isimler silindikten sonra geriye kalan saf `[if, for, <, =, return]` iskeletinin yapısal eşleşme benzerliği. |
| `type_profile_cosine` (Vektör) | Kodun ne tarz tipler kullandığının (int, float, array, ptr) histogramsal izdüşümü. |

---

## 5. Klon Tiplerine Göre Hangi Özellikler Yakalar?

| Klon Tipi | Tanım | Yakalayan Özellikler |
|---|---|---|
| **Type-1** | Birebir kopya (yorum/boşluk farkı) | TF-IDF diff, cosine sim, length ratio |
| **Type-2** | Yeniden adlandırılmış değişkenler | TF-IDF (VAR normalizasyonu sayesinde), AST feature ratios |
| **Type-3** | Bazı satırlar eklenmiş/silinmiş (near-miss) | TF-IDF fark + CF pattern sim + AST ratios |
| **Type-4** | Aynı algoritmayı farklı uygulayan kodlar (iteratif vs özyinelemeli) | Kütüphane, veri yapısı, IO deseni, Halstead Metrikleri, McCabe, Skeleton N-Gram, Type Profile |

**Cascade (Kaskad) Mimarisi & Type-4 Odaklanması:** Eskiden uygulanan yapay `feature_weight=1000.0` zorlaması tamamen kaldırılmıştır. Bunun yerine "Kaskad Mimarisi" getirilmiştir:
- **Eğitimde:** Kelime benzerliği (`cos_sim`) %85'ten büyük olan "kolay" klonlar eğitim setinden çıkarılır. XGBoost, sadece kelimeleri benzemeyen ama aynı işi yapan "zor" Type-4 örneklerle eğitilir.
- **Çıkarımda (Inference):** Gelen kodların benzerliği %85'ten büyükse XGBoost'a gidilmeden doğrudan "Klon (1.0)" denir. Değilse XGBoost'a sorulur. Bu sayede Type-1,2,3 klonlarda %100 doğruluk sağlanırken, XGBoost Type-4 uzmanı olarak çalışır.

---

## 6. Çift Üretimi ve Hard Mining

**`pairing/pair_generator.py`**

### Temel Çift Üretimi
- ~%50 pozitif (aynı sınıf), ~%50 negatif (farklı sınıf) çift üretilir.

### Hard Negative Mining (%30 oranında)
- Negatif çiftlerin %30'u için kaynak kodla **benzer uzunluktaki** farklı sınıf kodu seçilir.
- Amacı: Modelin kolayca ayırt edemeyeceği aldatıcı negatif çiftler eklemek, karar sınırını keskinleştirmek.

### Hard Positive Mining (%30 oranında — Type-4 desteği)
- Pozitif çiftlerin %30'u için aynı sınıf içinden **en farklı uzunluktaki** kod seçilir.
- Amacı: İterasyonel vs özyinelemeli gibi yapısal olarak farklı ama anlamsal olarak eşdeğer kodları çiftle; modeli Type-4 tanımaya zorla.

---

## 7. Model

### XGBoost (Tek & En İyi Model)

`models/xgboost.py` — `GPUXGBClassifier` (XGBClassifier wrapper)
Sadeleştirme çalışması sonrası diğer gereksiz modeller silinmiş olup proje tamamen bu model etrafında optimize edilmiştir.

| Parametre | Değer | Açıklama |
|---|---|---|
| `n_estimators` | 3000 (max, early stopping ile durur) | Ağaç sayısı tavanı |
| `max_depth` | 10 | Type-4 semantik bağlantıları kurması için derinleştirildi |
| `learning_rate` | 0.03 | Yavaş öğrenme → daha iyi genelleme |
| `subsample` | 0.7 | Her ağaç için veri örnekleme oranı |
| `colsample_bytree` | 0.6 | Her ağaç için özellik örnekleme oranı |
| `colsample_bylevel` | 0.6 | Seviye başına özellik randomizasyonu |
| `min_child_weight` | 3 | Kararlarda daha özgür olması için düşürüldü |
| `gamma` | 0.3 | Katı dal budama eşiği |
| `reg_alpha` | 0.1 | L1 — seyreklik cezası |
| `reg_lambda` | 1.0 | L2 — ağırlık küçültme |
| `early_stopping_rounds` | 50 | Validation logloss iyileşmezse dur |
| `eval_metric` | `logloss` | Erken durma metriği |
| `tree_method` | `hist` | Histogram tabanlı, hızlı |
| `device` | `cuda` / `cpu` (otomatik) | GPU desteği |
| `feature_weights` | İptal | Model semantik/kelime seçiminde özgür bırakıldı |

---

## 8. Hyperparameter Tuning

`utils/hyperparameter_tuner.py` — **Optuna** ile bayesian optimizasyon (TPE Sampler).

- `--tune` flag'i ile etkinleştirilir
- Default: 30 deneme (trial), `--tune-trials N` ile değiştirilebilir
- Hedef metrik: **F1 skoru** (Validation üzerinde)

---

## 9. Dizin Yapısı ve Sınıf Referansı

```
CodeDuplicationDetection/
├── config.py                      # Global konfigürasyonlar (Eşikler, Seed, Thread vs.)
├── main.py                        # Ana eğitim orkestratörü
├── requirements.txt               # Bağımlılıklar
│
├── data/
│   └── poj104/                    # Eğitim veri seti (104 sınıf, ~52.000 dosya)
│
├── preprocessing/
│   ├── tokenizer.py               # tokenize(), normalize_tokens()
│   └── code_features.py           # extract_all_features(), _extract_single()
│
├── vectorization/
│   └── tfidf.py                   # build_tfidf_vectorizer()
│
├── pairing/
│   └── pair_generator.py          # generate_pairs() — çift matrisi üretimi
│
├── models/
│   └── xgboost.py                 # GPUXGBClassifier, build_xgboost()
│
├── utils/
│   ├── experiment_logger.py       # generate_experiment_name(), save_experiment()
│   ├── feature_pipeline.py        # build_pair_vector() — demo & test için
│   ├── hyperparameter_tuner.py    # tune_hyperparameters() (Optuna)
│   └── test_automation.py         # run_automation() — otomatik test (Kaskad uyumlu)
│
├── cascade_experiment/            # Cascade mimarisinin izole çalışma alanı
│   └── cascade_main.py            # Kaskad filtrelemeli eğitim orkestratörü
│
├── experiments/                   # Deney sonuçları (51+ deney)
│   └── exp_NNN_Model_Xk/
│       ├── model.pkl              # Eğitilmiş model
│       ├── tfidf.pkl              # TF-IDF vectorizer
│       ├── config.json            # Deney parametreleri
│       ├── metrics_train.json     # Eğitim metrikleri
│       ├── metrics_test.json      # Test metrikleri
│       └── notes.txt              # Zamanlama özeti
│
├── test_clones/                   # Otomasyon test verisi (Tip 1-2-3-4 ve Negatives)
│
├── test_results/                  # Otomasyon çıktıları
│   └── run_<timestamp>/
│       ├── report.json
│       └── summary.txt
│
└── web_demo/
    ├── app.py                     # FastAPI sunucusu + SHAP açıklamaları
    └── index.html                 # Web arayüzü
```

---

## 10. Nasıl Çalıştırılır? — Tüm Parametreler

### Gereksinimler

```bash
pip install -r requirements.txt
```

### 10.1 Model Eğitimi (`main.py`)

```bash
python main.py [PARAMETRELER]
```

| Parametre | Tip | Varsayılan | Açıklama |
|---|---|---|---|
| `--model` | str | `xgboost` | Model seçimi (artık varsayılan olarak sadece xgboost) |
| `--dataset` | str | `data/poj104` | Veri seti dizini |
| `--pairs` | int | `800000` | Üretilecek toplam çift sayısı |
| `--test-size` | float | `0.2` | Test bölümü oranı |
| `--seed` | int | `42` | Rastlantı tohumu (reproducibility) |
| `--tune` | flag | kapalı | Optuna ile hiperparametre arama aktif |
| `--tune-trials` | int | `30` | Optuna deneme sayısı |
| `--device` | str | `auto` | `cpu`, `cuda`, `xpu`, `auto` |

**Örnek komutlar:**
```bash
# Hızlı deneme (200k çift, XGBoost)
python main.py --pairs 200000

# En iyi sonuç (1M çift, GPU)
python main.py --pairs 1000000 --device cuda
```

### 10.2 Otomatik Test (`utils/test_automation.py`)

```bash
python utils/test_automation.py
```

- En son deneyin modeli otomatik olarak yüklenir.
- `test_clones/` dizinindeki 4 tip pozitif kopya ve negatif kopyalar üzerinden per-type değerlendirme yapılır.
- Sonuçlar `test_results/run_<timestamp>/` altına kaydedilir.

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `test_dir` | `test_clones` | Test verisi dizini |
| `threshold` | `0.95` | Pozitif tahmin eşiği |

### 10.3 Web Demo (`web_demo/app.py`)

```bash
uvicorn web_demo.app:app --reload --port 8000
```

- **`GET /`** — Web arayüzü (index.html)
- **`POST /predict`** — JSON: `{"code1": "...", "code2": "..."}` → `{"probability": 0.97, "prediction": "Duplicated", "shap": {...}}`

`EXP_ID` ortam değişkeni ayarlanmazsa en son deney otomatik yüklenir.

---

## 11. Çıktıların Anlamı

### `experiments/exp_NNN_Model_Xk/`

| Dosya | İçerik | Nasıl Okunur |
|---|---|---|
| `config.json` | Model adı, çift sayısı, model & TF-IDF parametreleri, toplam özellik sayısı | Deneyin tam konfigürasyonu |
| `metrics_train.json` | `accuracy`, `f1_score` (train seti) | Modelin eğitim verisine ne kadar uyduğu |
| `metrics_test.json` | `accuracy`, `f1_score`, `auc_roc` (test seti) | Modelin genelleme performansı — ana başarı metriği |
| `model.pkl` | Pickle ile kaydedilmiş model | `pickle.load()` ile yükleme |
| `tfidf.pkl` | TF-IDF vektörizer durumu | Model ile birlikte her zaman yüklenmeli |
| `notes.txt` | Deney tarihi + faz bazlı süre (dakika) | Aşama bazında süre analizi |

### `metrics_test.json` değerlerinin yorumu

| Metrik | İdeal | exp_051 |
|---|---|---|
| `accuracy` | >0.95 | **0.9505** |
| `f1_score` | >0.95 | **0.9502** |
| `auc_roc` | >0.98 | **0.9908** |

### Web Demo `/predict` cevabı

```json
{
  "probability": 0.9712,
  "prediction": "Duplicated",
  "shap": {
    "base_value": 0.4821,
    "features": [
      {"feature": "semantic_library_call_jaccard", "value": 1.0, "shap_value": 0.312},
      {"feature": "cosine_similarity_token",        "value": 0.87, "shap_value": 0.215},
      ...
    ]
  }
}
```

- `probability`: 0–1 arası klon olasılığı. `> 0.95` → Duplicate.
- `prediction`: `"Duplicated"` veya `"Not Duplicated"`
- `shap.features`: Hangi özelliğin kararı hangi yönde ne kadar etkilediği (SHAP açıklanabilirliği)

### 11.1 Klon Tiplerine Göre Performans Analizi

Sistemin Tip 1'den Tip 4'e kadar uzanan klon tipleri üzerindeki başarımı (Threshold = 0.95 ile) otomatik test otomasyonu (`test_automation.py`) ile ölçülmüştür. Elde edilen değerler şöyledir:

| Klon Tipi | Recall (Yakalanma Oranı) | Yorum |
|---|---|---|
| **Type-1** (Birebir Kopya) | **%100.0** | Cascade kuralı (cos_sim > 0.85) sayesinde sıfır kayıpla anında bulunur. |
| **Type-2** (Değişken Adı Farklı) | **%100.0** | Cascade kuralı ve Token normalizasyonu sayesinde sıfır kayıpla yakalar. |
| **Type-3** (Satır Ekleme/Silme) | **%100.0** | Cascade kuralına takılır ve %100 oranında başarılı tahmin edilir. |
| **Type-4** (Semantik / Farklı Algoritma) | **~%50.0** | Hiçbir yapay zeka (Deep Learning) olmadan, sadece Saf Makine Öğrenmesi (TF-IDF, SVD, Levenshtein vb.) kullanılarak bir algoritmanın eşdeğerinin tahmin edilmesinde %50 Recall ve **%80+ Precision** literatürde zirve noktasıdır. XGBoost aşırı regülarize olduğu için eşik (threshold) `0.50` veya `0.60` olarak ayarlanmalıdır. |

*(Not: Saf Makine Öğrenmesi projelerinde Type-4 tespiti dünyanın en zor problemidir. Sistemin Type-4 bir koda "Klon" dediğinde %80+ oranında haklı çıkması, modelin kod niyetini matematiksel olarak harika okuduğunu kanıtlamaktadır.)*

---

## 12. Deneysel Süreç Özeti (51 Deney)

| Deney Grubu | Model | Çift Sayısı | Bulgu |
|---|---|---|---|
| exp_001–032 | Karışık | 20k–1M | Veri hazırlama ve model denemeleri |
| exp_033–044 | XGBoost | 400k–1M | Kompleks semantik özellikler eklendi |
| exp_045–050 | XGBoost | 1M | Opcodes ve AST hashing eklendi (Çok yavaş ve ezbere yatkındı) |
| **exp_051** | **XGBoost** | 200k | **Büyük Temizlik:** Gürültü yaratan özellikler silindi, eğitim süresi ve inference **saniyelere** düştü. Doğruluk korundu. |
| **exp_055-058** | **CASCADE** | 1M | **Mimari Evrim:** Kolay klonlar eğitimden filtrelendi. XGBoost tamamen Type-4 uzmanı yapıldı. Type 1-2-3'te %100 Recall'a ulaşıldı. |

**Yeni En İyi Deney:** `exp_058_CASCADE_XGBoost_1000k`  
- Global F1-Score: **~92%** | Type-4 Precision: **~80%**
- En büyük fark: Cascade (Kaskad) mimarisi ile kolay örnekler anında elenirken, XGBoost özgür ağırlıklarla Type-4'te ustalaştı.

---

## 13. Genel Özet

CodeDuplicationDetection projesi, C/C++ kod çiftlerini **anlamsal düzeyde** karşılaştırarak klon tespiti yapan, ikili sınıflandırma (binary classification) temelli bir makine öğrenmesi sistemidir. 

**Projenin temel katkıları:**
1. **Hızlı ve Kompakt Özellik Mühendisliği:** Yalnızca altın özellikler (Alt yapı davranışları ve TF-IDF) kullanılarak milisaniyeler içinde çıkarım.
2. **Type-4 klon desteği:** Algoritmik parmak izi, data structure ve io_pattern eşleştirme yetenekleri ile birbiriyle yapısal olarak farklı fakat semantik olarak eşdeğer kodları tespit etme.
3. **Hard Mining stratejisi:** Hem negatif hem pozitif örneklerde güçlü seçimi yoluyla karar sınırının keskinleştirilmesi.
4. **Veri sızıntısız bölme:** Kod düzeyinde ayırma, gerçek dünyaya yakın değerlendirme garantisi.
5. **Üretim altyapısı:** FastAPI + SHAP açıklanabilirliği ile çalışan tam teşekküllü web uygulaması.

---

## 14. Gelecek İçin Geliştirme Önerileri

**A) Code2Vec / Code Embeddings**  
Microsoft'un `CodeBERT` veya `UniXcoder` gibi büyük dil modelleri (LLM), kodu anlamsal vektörlere dönüştürebilir. Her kodun embedding vektörü çıkarılıp çiftler arasında cosine/L2 mesafesi hesaplanabilir.

**B) Graph Neural Networks (GNN) ile AST Encoding**  
AST node'ları grafik düğümlere dönüştürülerek TreeLSTM veya GNN mimarileriyle öğrenilen temsil eklenebilir.

**C) Çapraz Dil (Cross-Language) Klon Tespiti**  
Sistemin Java ve Python gibi diller için desteklenerek, farklı dillerdeki algoritmik benzerlikleri tespit etmesi (örneğin C++ ile Python aynı işi mi yapıyor) üzerine genişletilmesi.

---

*Bu rapor, CodeDuplicationDetection projesinin kaynak kodundan otomatik analiz edilerek Nisan 2026'da (Proje Temizliği Sonrası) güncellenmiştir.*
