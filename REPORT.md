# 🚀 Code Duplication Detection — Kapsamlı Teknik Rapor ve Mimari Özeti

**Proje:** CodeDuplicationDetection  
**Sürüm:** 2.0 (İleri Seviye Optimizasyon Sürümü)  
**Kapsam:** C/C++ Kaynak Kodları (Type 1, 2, 3 ve 4 Klon Tespiti)  
**Temel Mimari:** Two-Stage Cascade Sistemi + Feature-Partitioned Ensemble / GPU XGBoost  
**Derin Öğrenme:** CodeBERT tabanlı Mean-Pooled SSL Embeddings + Dinamik PCA İndirgeme  

---

## 1. Proje Özeti ve Amacı

Bu proje, C/C++ kaynak kodları arasında var olan benzerlikleri inceleyerek **kopya/klon (code clone)** tespiti yapan gelişmiş bir Makine Öğrenmesi (ML) ve Doğal Dil İşleme (NLP) ardışık düzenidir (pipeline). İki farklı kaynak kod dosyasını (A ve B) girdi olarak alır ve aynı mantıksal problemi çözüp çözmediklerini analiz ederek "1 (Klon)" veya "0 (Farklı)" kararı üretir.

Sistem, literatürde tanımlanmış **4 kod klon tipini** de kapsayacak şekilde uçtan uca özelleştirilmiştir:
1. **Type-1 (Birebir Kopya):** Sadece yorum satırları (comment) ve boşlukların (whitespace) değiştirildiği klonlar.
2. **Type-2 (Yeniden Adlandırılmış Kopya):** Değişken, fonksiyon, makro ve sınıf isimlerinin değiştirildiği, temel tiplerin farklılaştırıldığı klonlar.
3. **Type-3 (Modifiye Kopya):** Küçük kod bloklarının silindiği, yeni satırların eklendiği veya ifadelerin değiştirildiği klonlar.
4. **Type-4 (Semantik/Anlamsal Kopya):** Aynı problemi çözen ancak *tamamen farklı algoritmik yapıya ve sözdizimine* sahip klonlar. Geleneksel metin tabanlı veya soyut sözdizimi ağacı (AST) tabanlı sistemlerin başarısız olduğu, sistemimizin asıl gücünü gösterdiği alandır.

---

## 2. Pipeline: Klon Tipleri Nasıl Tespit Edilir?

Projedeki veri akışı (pipeline) her bir klon tipini yakalamak için farklı uzmanlıklara sahip 4 ana katmandan oluşur. Girdi olarak alınan kod çifti sırasıyla bu katmanlardan geçerek bir özellik vektörüne (Feature Vector) dönüşür.

### A. Preprocessing ve Normalizasyon Katmanı (Type-1 ve Type-2)
*   **Akıllı Yorum Temizliği (Comment Stripping):** Koda dahil edilmiş `//` ve `/* */` içerikleri C++ dil kurallarına uygun olarak tamamen temizlenir. Bu adım **Type-1** klonları neredeyse "sıfır fark" (exact match) haline getirir.
*   **Tokenizer ve Değişken İndeksleme:** Koddaki döngü değişkenleri (i, j, k), geçici değişkenler (temp, swap) ve sayısal değerler (0xFF, 42UL, 3.14) standardize edilir. İsim değiştirmeleri `VAR1`, `VAR2` veya `FUNC1` şeklinde anonimleştirilir. Bu sayede isimleri tamamen değiştirilmiş kodlar **(Type-2)** sistem için aynı kelime dizilimi (sequence) olarak algılanır.

### B. Leksikal ve AST (Soyut Sözdizimi Ağacı) Katmanı (Type-3)
*   Token diziliminden hesaplanan **TF-IDF Kosinüs Benzerliği** ve satır oranları gibi Leksikal Özellikler, **Type-3** klonlarındaki küçük satır eklemelerini (insertion/deletion) tolere eder.
*   **Tree-sitter C++ Parser:** Kodun AST'sini çıkararak döngü, dallanma, işlem operatörleri, I/O paternleri ve McCabe/Halstead karmaşıklık metriklerini çıkarır. Kodun bir kısmı değiştirilse bile yapısal karmaşıklık oranı (ratio) ve if-else yoğunlukları **Type-3** klonları ele verir.
*   *Önemli Not:* `code_features.py` içerisinde kullanılan `MultiSet (Counter)` mantığı ile özelliklerin sadece varlığı değil, **sıklığı (frekansı)** da Jaccard benzerliği hesabına katılır. (Örn: 5 defa printf kullanan kod ile 1 defa kullanan kod artık ayrıştırılır).

### C. Semantik Özellik Katmanı (Type-4 Başlangıcı)
Aynı algoritma (örn: sıralama), biri `for` döngüsü ve dizilerle, diğeri özyineli (recursive) `if` dallanmaları ve pointer'larla yazılabilir. Bu durumda Leksikal ve AST benzerliği düşer.
*   **Kütüphane ve Tip Profili:** İki kod string (std::string vs char*), hafıza yönetimi (malloc vs new) veya IO (cin/cout vs printf/scanf) kullanımlarında farklılaşsa bile, üst soyutlamadaki kütüphane kategorileri (ALGO, MATH, IO) karşılaştırılarak algoritmik hedeflerin aynı olup olmadığı anlaşılır. `void, bool, auto` profillemeleri yapılır.

### D. Deep Learning SSL (CodeBERT) Katmanı (Type-4 Uzmanı)
En inatçı **Type-4** klonlarını çözmek için Microsoft CodeBERT (Self-Supervised Learning) modeli kullanılır.
*   **Mean Pooling ve Chunking:** Kodlar 510 token'lık parçalara (chunks) ayrılır (uzun kod kayıplarını/truncation önlemek için). Çıkan tüm gizli (hidden) katmanların ortalaması alınarak (Mean Pooling) 768 boyutlu vektör oluşturulur.
*   **PCA Fark (Abs Diff):** PCA ile 64 boyuta indirgenen gömülerin mutlak farkı (`|pca_A - pca_B|`) özellik vektörüne eklenir. XGBoost/Ensemble modeli bu 64 farklı anlamsal eksendeki farklılıkları Type-4 kararı için ustaca kullanır.
*   *Güvenlik:* Veri Sızıntısını (Data Leakage) önlemek için PCA sadece cross-validation eğitim setine (Train) *fit* edilir. Disk cache'lemesi (MD5 hash) sayesinde OOM riski olmadan ultra-hızlı çalışır.

---

## 3. Two-Stage Cascade (İki Aşamalı Kaskad) Mimarisi

Kapsamlı özellik vektörleri hesaplansa da, Type-1 ve Type-2 klonlar için PCA veya Semantik analiz yapmak zaman kaybıdır. Ayrıca, karmaşık modeller bu kolay örnekleri "ezberleyebilir" ve zor klonlarda (Type-4) zayıf kalabilir. Bu yüzden sistem **iki aşamalı bir kalkan (Cascade)** kullanır:

1.  **Stage-1 (Kolay Klon / Negatif Filtresi):**
    *   Hızlı ve hafif `HistGradientBoostingClassifier` modeli (Sadece ilk 32 yüzeysel/yapısal özelliği kullanarak çalışır).
    *   Olasılık çıktıları matematiksel olarak düzeltilmiştir (`CalibratedClassifierCV`).
    *   Eşik (`CASCADE_STAGE1_THRESHOLD = 0.85`): Eğer sistem bir kodun açıkça klon (Type-1/2) olduğundan `%85`'ten fazla eminse, karar hemen verilir. Aşırı bariz negatifler de (farklı olduğu kesin olanlar) filtrelenir.
2.  **Stage-2 (Zor Vaka Uzmanı):**
    *   Eğitim aşamasında: Stage-1 tarafından başarıyla elenen tüm kolay klonlar ve negatifler veri setinden çıkartılır. Geriye kalan şiddetli dengesizlik (imbalance) `scale_pos_weight` ile kompanse edilir.
    *   **XGBoost** veya **Feature-Partitioned Stacking Ensemble** modeli, enerjisinin %100'ünü geriye kalan zor **Type-3 ve Type-4** klonları sınıflandırmaya harcar. Ensemble modeli, Leksikal özellikler için LightGBM, Yapısal özellikler için Random Forest, Semantik özellikler için Calibrated LinearSVC alt modellerini çalıştırıp Lojistik Regresyon meta-modeli ile (C=0.1) birleştirir.

---

## 4. Hard Mining: Gerçek Hayatı Simüle Etmek

Makine öğrenmesinde "rastgele" çift (pair) seçmek, sadece kolay klonları üretir. Projenin çift oluşturma modülü (`pair_generator.py`) akıllı madencilik (Hard Mining) kullanır:
*   **Hard Negative Mining (%30):** Rastgele iki farklı problem seçmek yerine, *kod uzunlukları birbirine en yakın olan* iki farklı problemi eşleştirir. Model "boyutları aynı diye klon sanma" yanılgısından kurtulur.
*   **Hard Positive Mining (%30 - Type-4 Zorlaması):** Aynı problemi çözen iki kodu seçerken rastgele değil, **TF-IDF uzayında Kosinüs Benzerliği EN DÜŞÜK (birbirine en az benzeyen)** iki kodu seçer. Bu sistemin modele kasıtlı olarak en acımasız Type-4 örnekleri vermesini sağlar.
*   **Filtreleme:** `np.unique` mekanizması kullanılarak üretilen on binlerce veri seti içerisindeki tekrarlı çiftler (duplicates) ve simetrik hatalar otomatik olarak temizlenir.

---

## 5. Komut Referansı (Nasıl Çalıştırılır?)

**Ortam Hazırlığı:** Proje Python 3.12 gerektirir. `uv` veya `pip` kullanabilirsiniz.
```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-gpu.txt  # SSL Model (CodeBERT) ve PyTorch için
```

**Standart Model Eğitimi (XGBoost):**
```bash
python main.py --pairs 100000 --model xgboost --use-ssl
```

**Cross-Validation ile Ensemble Eğitimi (Veri Sızıntısı Korumalı):**
```bash
python main.py --pairs 50000 --model ensemble --use-ssl --cv --cv-folds 5
```

**Hiperparametre Optimizasyonu (Optuna):**
```bash
python main.py --tune --tune-trials 50 --model xgboost
```

**Test Otomasyonu (Type 1-4 Raporu Alma):**
```bash
python utils/test_automation.py --exp-id 1 --threshold 0.85
```

**Web Uygulaması (Açıklanabilir AI / SHAP):**
```bash
cd web_demo && uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Tarayıcınızdan `http://localhost:8000` adresine girerek anlık analiz yapabilir, modelin kararında (SHAP) hangi kod özelliklerinin ağırlık taşıdığını görebilirsiniz.
