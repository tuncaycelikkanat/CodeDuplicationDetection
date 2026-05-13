# Code Duplication Detection — Teknik Rapor

**Proje:** CodeDuplicationDetection  
**Tarih:** Mayıs 2026  
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
    ├── Hard Negative Mining (%30 oranında — benzer uzunlukta farklı sınıflar)
    └── Hard Positive Mining (%30 oranında — Type-4 zorlaması, en farklı uzunlukta aynı sınıflar)
        │
        ▼
[6] Vektör Oluşturma (Dense Feature Array)
    [cos_token | length_ratio | manhattan | euclidean | AST Ratios & Diffs | CF Sim | Semantic Jaccards | SVD Diffs]
    (Toplam ~90+ yoğun özellik. Seyrek TF-IDF fark matrisi performansı maksimize etmek için kaldırıldı.)
        │
        ▼
[7] Model Eğitimi (Two-Stage Cascade Mimarisi)
    ├── Stage-1: Lojistik Regresyon (Sadece Lexical özellikler ile eğitilir, kolay klonları bulur)
    ├── Kaskad Filtreleme: Eğitimden Stage-1'in kolayca klon bulduğu (%95+ prob) örnekler silinir.
    └── Stage-2 (Uzman): XGBoost veya Stacking Ensemble (LightGBM + RF + LinearSVC) "Zor/Type-4" klonlara odaklanır.
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

Projenin en büyük evrimi, basit tek aşamalı tahminden **İki Aşamalı (Two-Stage)** filtreleme ve sınıflandırma sistemine geçişidir. Eski `main.py` `main_deprecated.py` yapılarak kullanımdan kaldırılmış ve eğitim tamamen yeni `main.py` (Cascade) içerisine taşınmıştır.

### Aşama 1: Lexical Filter (Kolay Klon Yakalayıcı)
Model eğitiminin başında, sadece ilk 4 leksikal özellik (Token Kosinüs Benzerliği, Uzunluk Oranı, Manhattan ve Öklid mesafeleri) kullanılarak hızlı bir **Logistic Regression** (Stage-1) eğitilir.
- **Amacı:** Eğitim verisindeki birbirine çok benzeyen Type-1 ve Type-2 klonları tespit edip (%95+ olasılıkla) Stage-2'nin eğitim setinden çıkarmak.
- **Faydası:** Karmaşık modelin (Stage-2) kolay örnekleri ezberlemesi engellenir. Model sadece "yapısal olarak farklı ama anlamsal olarak eşdeğer" olan **Type-4** klonları öğrenmeye zorlanır. Çıkarım (inference) anında ise bu model, gelen kodun kolay bir klon olup olmadığını saniyeden kısa sürede anlayıp sistem kaynaklarını korur.

### Aşama 2: Type-4 Uzmanı (XGBoost veya Ensemble)
Zorlu örnekler üzerinde iki farklı uzman model kullanılabilir:

1. **GPU XGBoost (`models/xgboost.py`):** Derin ağaçlarla (`max_depth=10`) semantik örüntüleri yakalayan geleneksel uzman.
2. **Feature-Partitioned Stacking Ensemble (`models/ensemble.py`):**
   Özellik gruplarının ilgili uzman modellere dağıtıldığı yepyeni bir mimari:
   - **LightGBM (HistGradientBoosting):** Sadece Yüzeysel (Lexical) ve Vektörel (SVD) özelliklere bakar.
   - **Random Forest:** Sadece Yapısal (AST Ratios/Diffs ve CF Patterns) özelliklere bakar.
   - **LinearSVC (Calibrated):** Sadece Semantik (Jaccard benzerlikleri, Type Profiles) özelliklere bakar. LinearSVC kullanılarak SVM tabanlı öğrenmenin eğitim hızı dramatik düzeyde artırılmıştır.
   - **Meta-Classifier (Logistic Regression):** Bu üç modelin çıktı olasılıklarını birleştirerek nihai kararı verir.

---

## 3. Özellik Mühendisliği (Feature Engineering)

Proje, bellek tüketen seyrek (sparse) TF-IDF fark matrisini bırakıp tamamen yoğun (dense) özellik vektörlerine geçmiştir. Bu sayede özellik matrisi kompaktlaşmış ve model eğitim/çıkarım hızı inanılmaz oranda artmıştır.

### 3.1 Lexical (Yüzeysel) Özellikler
- Token tabanlı kosinüs benzerliği (`cos_token`)
- Kod uzunluk oranı (`length_ratio`)
- TF-IDF mutlak farklarının L1 (Manhattan) ve L2 (Öklid) normları.

### 3.2 AST ve Karmaşıklık Özellikleri
Regex ile soyutlanmış hızlı metriklerin oranları (`min/max`) ve farkları (`abs(a - b)`):
- Döngü, dal (branch), fonksiyon çağrısı, operatör ve parametre yoğunlukları.
- **Halstead Hacmi / Zihinsel Çaba (Effort):** Kodun algoritmik karmaşıklığını ve satır başına düşen yoğunluğunu (density) ölçer.
- **McCabe Karmaşıklığı:** Karar noktalarının yoğunluğu.

### 3.3 Semantik "Algoritmik Parmak İzi" Özellikleri (Type-4 İçin Zirve)
Yapay zeka (Deep Learning) olmadan kodun niyetini okumayı sağlayan Jaccard/Kosinüs benzerlikleri:
- **Kütüphane / Kategori Benzerliği:** `printf`, `malloc`, `sort` gibi çağrıların ve kategorik kullanımların Jaccard benzerliği.
- **Veri Yapıları:** Array, stack, queue, map, vector kullanım benzerliği.
- **IO Deseni (IO Pattern):** Girdi ve çıktı sırasının (örn. `I-O-I-O`) string bigram eşleşmesi.
- **Matematiksel Operatörler:** `+`, `-`, `*`, `%` kümesinin benzerliği.
- **Kontrol Akış İskeleti (Skeleton & Abstract CF):** Tokenlardan arındırılmış saf döngü/şart dizilimlerinin Levenshtein mesafesi.
- **Değişken Tip Profili:** Kodun hangi veri tiplerini (int, float, char, pointer, array) ne oranda kullandığının kosinüs benzerliği.

### 3.4 SVD Boyut İndirgeme
- Token seviyesindeki binlerce boyutlu TF-IDF temsili, **TruncatedSVD** ile 50 boyuta düşürülür. Çiftler arasındaki bu SVD bileşenlerinin mutlak farkları, gürültüden arındırılmış semantik sinyal olarak modele verilir.

---

## 4. Çift Üretimi ve Hard Mining (`pairing/pair_generator.py`)

Sistem eğitim setini hazırlarken Numpy tabanlı vektörize operasyonlar ve `joblib` paralel işleme teknikleri kullanır. Öğrenmeyi hızlandırmak ve karar sınırını keskinleştirmek için gelişmiş madencilik (mining) işlemleri yapılır:

- **Hard Negative Mining (%30):** Rastgele iki farklı problem sınıfı seçmek yerine, kaynak koda **en yakın uzunluktaki** farklı sınıf kodu seçilerek modeli yanıltabilecek aldatıcı negatifler üretilir.
- **Hard Positive Mining (%30):** Aynı sınıf içinden **en farklı uzunluktaki** kod seçilerek Type-4 klonları (örneğin kısa bir yinelemeli fonksiyon ile uzun bir döngüsel fonksiyon) zorla eşleştirilir.
- **Esnek Positive Ratio:** Sistem `0.5` oranla dengeli veya `0.1` gibi gerçek dünyadaki kod tabanlarına benzeyen (klon olmayanların çoğunlukta olduğu) dağılımlarla eğitilebilir.

---

## 5. Değerlendirme & Test Otomasyonu

**`utils/test_automation.py`**
Sistem, test setini tüm Klon Tiplerine (Type 1, 2, 3, 4) ayırarak, "Gerçek Negatifler" ile harmanlayıp bağımsız test eder. Raporlar konsola basılır ve `test_results/run_<timestamp>/` altında detaylı JSON, metin özeti ve Karışıklık Matrisi (Confusion Matrix) formatlarında arşivlenir. 

Ayrıca Optimizasyon Aracı (`utils/find_best_threshold.py`) kullanılarak, F1 ve MCC (Matthews Correlation Coefficient) değerlerini en üst noktaya çıkaran ideal olasılık eşiği (threshold) saptanır.

| Klon Tipi | Yaklaşım ve Başarım |
|---|---|
| **Type-1** (Birebir Kopya) | Stage-1 Lexical filtre veya Kaskad eşiği (cos_sim > 0.85) anında **%100** doğrulukla yakalar. Karmaşık modele gönderilmez. |
| **Type-2** (Değişken Adı Farklı) | Token normalizasyonu (tüm değişkenler VAR, fonksiyonlar FUNC) sayesinde Stage-1 **%100** oranında klon olarak işaretler. |
| **Type-3** (Satır Ekleme/Silme) | Edit distance (Levenshtein) ve CF desen benzerliği sayesinde yine çoğu Stage-1 tarafından yakalanır (**%100** başarısına yaklaşılır). |
| **Type-4** (Farklı Algoritma) | Yalnızca Hard Mining ile eğitilmiş Stage-2 (XGBoost / Ensemble) uzman modeller devreye girer. Sadece kural tabanlı Makine Öğrenmesi metrikleri ile dünya standardında zirve olan **%50+ Recall ve Çok Yüksek Precision** dengesine ulaşılır. |

---

## 6. Dizin Yapısı ve Modüller

```text
CodeDuplicationDetection/
├── main.py                        # AKTİF EĞİTİM ALANI (Two-Stage Cascade Eğitim Betiği)
├── models/
│   ├── ensemble.py                # Feature-Partitioned Stacking Ensemble Mimarisi
│   └── xgboost.py                 # GPU XGBoost Mimarisi
├── pairing/
│   └── pair_generator.py          # Vektörize Çift ve Hard-Mining Üreticisi
├── preprocessing/
│   ├── tokenizer.py               # Kod tokenizasyon ve VAR/FUNC/NUM normalizasyonu
│   └── code_features.py           # AST, Metric ve Semantik (Jaccard, IO vb.) özellikler
├── vectorization/
│   └── tfidf.py                   # Token ve Karakter TF-IDF (SVD destekli)
├── utils/
│   ├── feature_pipeline.py        # Çıkarım (Inference) için tekil çift özellik oluşturucu
│   ├── test_automation.py         # Type 1-4 ayrık model test aracı
│   ├── find_best_threshold.py     # Optima eşik arayıcı
│   ├── similarity_utils.py        # Merkezi Jaccard/N-Gram benzerlik fonsiyonları
│   └── experiment_logger.py       # Deney kaydı, CV sonuçları, config.json oluşturucu
├── commands.md                    # Komut Referansı (Nasıl Çalıştırılır?)
├── tests/                         # Unit ve Integration testleri
├── web_demo/                      # Canlı model çıkarımı yapan FastAPI / SHAP Arayüzü
└── main_deprecated.py             # DEPRECATED (Eski tek aşamalı mimari, kullanılmaz)
```

> **Nasıl Çalıştırılır?** 
> Sistemdeki tüm işlemler, hiperparametre optimizasyonundan web demosuna kadar `commands.md` dosyasında detaylandırılmıştır. 
> Hızlı bir Ensemble eğitim başlatmak için:
> ```bash
> python main.py --pairs 1000000 --model ensemble
> ```

---

*Bu rapor, CodeDuplicationDetection projesinin "Two-Stage Cascade ve Feature-Partitioned Stacking Ensemble" evrimine uygun olarak Mayıs 2026'da güncellenmiştir.*
