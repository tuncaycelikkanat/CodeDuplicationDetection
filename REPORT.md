# 🚀 Code Duplication Detection — Kapsamlı Teknik Rapor ve Kullanım Kılavuzu

**Proje:** CodeDuplicationDetection  
**Sürüm/Tarih:** Mayıs 2026 (Güncel Sürüm)  
**Veri Seti:** POJ-104 (C/C++ Kaynak Kodu, 104 Farklı Problem Sınıfı, ~52.000 Dosya)  
**Temel Mimari:** Two-Stage Cascade Mimarisi (HistGBM + XGBoost/Feature-Partitioned Stacking Ensemble)  
**Derin Öğrenme Entegrasyonu:** CodeBERT SSL (Self-Supervised Learning) Gömüleri + PCA İndirgeme  

---

## İçindekiler
1. [Proje Özeti ve Amacı](#1-proje-özeti-ve-amacı)
2. [Mimari Tasarım: İki Aşamalı (Two-Stage) Cascade Sistemi](#2-mimari-tasarım-iki-aşamalı-two-stage-cascade-sistemi)
3. [Özellik Mühendisliği (Feature Engineering) Detayları](#3-özellik-mühendisliği-feature-engineering-detayları)
4. [Özel Modeller ve Hiperparametreler](#4-özel-modeller-ve-hiperparametreler)
5. [Çift Üretimi (Pairing) ve Hard Mining Algoritması](#5-çift-üretimi-pairing-ve-hard-mining-algoritması)
6. [Hızlı Başlangıç ve Komut Referansı (Nasıl Çalıştırılır?)](#6-hızlı-başlangıç-ve-komut-referansı-nasıl-çalıştırılır)
7. [Test Otomasyonu ve Değerlendirme](#7-test-otomasyonu-ve-değerlendirme)
8. [Açıklanabilir Yapay Zeka: Web Arayüzü (Web Demo)](#8-açıklanabilir-yapay-zeka-web-arayüzü-web-demo)
9. [Geliştirici Notları ve Optimizasyonlar](#9-geliştirici-notları-ve-optimizasyonlar)

---

## 1. Proje Özeti ve Amacı

Bu proje, C/C++ kaynak kodları arasında anlamsal (semantik) ve sözdizimsel (sentaktik) benzerlikleri tespit eden gelişmiş bir makine öğrenmesi ve Doğal Dil İşleme (NLP) sistemidir. Sistem, verilen iki kod parçasının aynı problemi çözüp çözmediğini değerlendirerek "1 (Klon/Kopya)" veya "0 (Farklı)" şeklinde ikili sınıflandırma (binary classification) yapar.

Proje, literatürdeki **4 farklı kod klonu tipini** yüksek doğrulukla tespit edecek şekilde tasarlanmıştır:
- **Type-1:** Birebir aynı kodlar. (Sadece yorum satırları ve boşluklar farklı). Tespit edilmesi en kolay klon tipidir.
- **Type-2:** Yeniden adlandırılmış kodlar. (Değişken, fonksiyon ve sınıf isimleri değiştirilmiş, veri tipleri farklılaştırılmış). Token normalizasyonu sayesinde kolayca yakalanır.
- **Type-3:** Yakın kopyalar. (Küçük satır ekleme, silme veya değiştirme işlemleri yapılmış). Hem leksikal hem yapısal metrikler gerekir.
- **Type-4:** Semantik klonlar. (Tamamen farklı algoritmik yapı ve sözdizimi, ancak aynı işlevsellik). *Sistemin asıl başarı gösterdiği alandır.* Geleneksel sistemlerin tıkandığı noktada CodeBERT ve derin algoritma analizleri devreye girer.

---

## 2. Mimari Tasarım: İki Aşamalı (Two-Stage) Cascade Sistemi

Sistemin kalbinde, performansı ve doğruluğu maksimize etmek için geliştirilmiş **Two-Stage Cascade (İki Aşamalı Kaskad) Mimarisi** bulunmaktadır. Bu mimari, eğitim ve çıkarım (inference) sürelerini dramatik biçimde hızlandırırken Type-4 başarısını artırır.

### Neden İki Aşama?
Tek bir büyük model (XGBoost) kullandığımızda, model zamanının büyük kısmını zaten bariz olan Type-1 ve Type-2 klonları ezberlemeye ayırıyordu. Zorlu (Type-4) semantik klonları ise gözden kaçırıyordu. Kaskad mimarisi sayesinde kolay klonlar ön kapıda filtrelenerek içeri alınmaz.

### Aşama 1 (Stage-1): Lexical (Yüzeysel) ve Yapısal Filtre
- **Model:** `HistGradientBoostingClassifier` (Hızlı ve sığ bir karar ağacı).
- **Girdi:** İlk 32 özellik (Token kosinüs benzerliği, satır/uzunluk oranları, AST metrikleri - `config.STAGE1_FEATURE_COUNT = 32`).
- **İşlev:** Eğitim verisindeki ve test/çıkarım sırasındaki **kolay klonları (Type-1, Type-2)** `CASCADE_STAGE1_THRESHOLD` (örn. 0.95) eşiğiyle anında tespit eder.
- **Faydası:** Karmaşık ana modelin bu kolay örnekleri ezberlemesi engellenir. Ana modelin (Stage-2) eğitim veri seti küçülür, eğitim hızlanır ve ağaçların yaprakları tamamen "zorlu vakalara" (Hard Examples) ayrılır.

### Aşama 2 (Stage-2): Semantik Uzman (XGBoost veya Ensemble)
- **Model Seçenekleri:** Derin GPU Destekli **XGBoost** veya özel olarak bileşenlere ayrılmış **Feature-Partitioned Stacking Ensemble**.
- **Girdi:** Lexical, AST, Abstract Control Flow, SVD (TF-IDF tabanlı LSA) ve CodeBERT SSL (PCA) farklılıklarından oluşan yoğun özellik vektörünün *tamamı*.
- **İşlev:** Stage-1'den "klon değil" veya "emin değilim" cevabı alan tüm kod çiftleri bu uzmana gönderilir. Derin bir semantik analiz yapılır.

---

## 3. Özellik Mühendisliği (Feature Engineering) Detayları

Sistem bellek tüketen seyrek (sparse) TF-IDF fark matrisini tamamen bırakmış, tamamen yoğun (dense) özellik vektörlerine (`np.float32`) geçmiştir.

Toplam Özellik Sayısı: **~91** (Sadece TF-IDF/SVD kullanılıyorsa) veya **~155** (CodeBERT SSL devredeyken).

### 3.1. Lexical ve Uzunluk Özellikleri [İndeks: 0-3]
En temel sözdizimsel analizlerdir. Type-1 ve Type-2 klonları için harikadır.
- `cos_token`: İki kodun TF-IDF token vektörlerinin kosinüs benzerliği. (Cascade filtresinin en çok başvurduğu metrik).
- `length_ratio`: `min(len1, len2) / max(len1, len2)`. İki kod arasındaki uzunluk asimetrisi.
- `manhattan_token` & `euclidean_token`: TF-IDF vektörlerinin mutlak L1 ve L2 fark normları.

### 3.2. AST ve Karmaşıklık Özellikleri [İndeks: 4-31]
Kodun soyut sentaks ağacından çıkarılan sayaçların oranları ve mutlak farkları.
- **Döngü ve Dallanma:** `branch_count`, `loop_call_combined`
- **Operatör ve Parametreler:** `operator_count`, `param_count`
- **Karmaşıklık (Complexity):**
  - **Halstead Hacmi (Volume):** Algoritmanın kapladığı alan.
  - **Halstead Zihinsel Çaba (Effort):** Algoritmanın anlaşılma zorluğu.
  - **McCabe Siklomatik Karmaşıklığı:** Karar düğümlerinin (if, while) sayısı.
- **Yoğunluk Metrikleri:** Uzunluğa karşı dirençli özellikler (örn. `branch_density = branches / lines`). İki kodun uzunluğu farklı olsa bile "if-else yoğunluğu" aynıysa Type-4 adayıdır.

### 3.3. Semantik "Algoritmik Parmak İzi" Özellikleri [İndeks: 32-40]
Regex ve Tree-sitter ile kodun "niyetini" okuyan özellikler kümesi. Type-4 için kritiktir.
- **Kütüphane Jaccard Benzerliği:** `printf`, `malloc`, `sort` gibi çağrıların kesişim oranı.
- **Kategori Benzerliği:** IO, MATH, STRING, MEMORY, ALGO gibi kütüphane kategorilerinin Jaccard benzerliği.
- **Veri Yapıları:** Array, Linked List, Stack, Queue, Map/Set kullanımlarının örtüşmesi.
- **I/O Deseni:** Koddaki `printf/scanf` sırasının (`I-O-I-O`) string bigram eşleşmesi.
- **Matematiksel Operatörler:** `+`, `-`, `*`, `%` kümesinin Jaccard benzerliği.
- **Kontrol Akış İskeleti (Skeleton):** Saf döngü/şart dizilimlerinin (`F-W-I-R`) Levenshtein edit mesafesi.
- **Tip Profili (`type_profile_cos`):** Kodda kullanılan veri tiplerinin (int, float, pointer, struct, array) oranlarının kosinüs benzerliği.

### 3.4. SVD Boyut İndirgeme (LSA) [İndeks: 41-90]
Token tabanlı N-Gram TF-IDF modeli (1-3 gram, maks 500 kelime) üzerine **TruncatedSVD** (Latent Semantic Analysis) uygulanarak boyut 50'ye (`SVD_N_COMPONENTS = 50`) sıkıştırılır.
İki kodun SVD bileşenlerinin mutlak farkı (`|emb_A - emb_B|`) özellik vektörüne 50 boyutlu bir dilim olarak eklenir.

### 3.5. Derin Öğrenme: CodeBERT SSL Embeddings ve PCA [İndeks: 91-154]
*Bu özellik `--use-ssl` parametresi ile aktifleşir.*
- **Self-Supervised Learning (SSL):** Microsoft'un geliştirdiği devasa ön-eğitimli C/C++ modelinden (`microsoft/codebert-base`) kodun 768 boyutlu semantik gömüleri (embeddings) çıkarılır.
- **PCA İndirgeme:** 768 boyut, 1 milyon çift (Pair) için RAM'e sığmaz (2.4 GB). Bu nedenle **PCA** ile semantik öz korunarak 64 boyuta (`SSL_PCA_COMPONENTS = 64`) indirgenir (~200 MB).
- **Mutlak Fark:** İki kodun CodeBERT gömülerinin mutlak farkı (`|pca_A - pca_B|`) özellik vektörüne eklenir. Sistem sadece 1 adet "benzerlik skoru" vermek yerine 64 farklı semantik boyutta "ne kadar farklı olduklarını" ağaçlara öğretir. Type-4 klon başarısında eşsiz bir artış sağlar.

---

## 4. Özel Modeller ve Hiperparametreler

`models/` dizininde iki ana mimari tanımlıdır. İkisi de Stage-2 uzmanı olarak hizmet eder.

### 4.1. Feature-Partitioned Stacking Ensemble (`models/ensemble.py`)
Milyonlarca satır özelliği tek bir modele vermek yerine, özellik grupları "uzmanlarına" dağıtılmıştır (Partitioning):
1. **LightGBM (HistGBM):** Leksikal (0-3) ve yüksek boyutlu SVD/SSL fark bloğunu (41+) alır. Eksik verilerle başa çıkabilen çok hızlı bir yapısı vardır.
2. **Random Forest:** AST ve Kontrol Akışı (4-32) bloğunu alır. Sayısal sayaçların non-lineer etkileşimlerini harika yakalar.
3. **LinearSVC (Calibrated):** Semantik parmak izi (33-40) bloğunu alır. Kategorik (Jaccard) verileri düzlemsel olarak çok hızlı ve yüksek netlikle ayırır (`Platt Scaling` ile olasılık çıktısı verir).
4. **Meta-Classifier:** Lojistik Regresyon, bu üç uzman modelin çıktısını girdi olarak alarak nihai kararı harmanlar.

### 4.2. GPUXGBClassifier (`models/xgboost.py`)
Klasik ve aşırı optimize edilmiş XGBoost yaklaşımıdır. `DMatrix` seviyesinde GPU'ya entegre edilmiştir. 
Parametreleri Type-4 yakalamaya özel dizayn edilmiştir:
- `max_depth = 10` (Derin ağaçlar semantik soyutlamaları yakalar)
- `learning_rate = 0.03` (Yavaş öğrenme generalizasyonu artırır)
- `min_child_weight = 3`, `gamma = 0.3` (Sert budama, ağaçların kolay klonları ezberlemesini engeller)

---

## 5. Çift Üretimi (Pairing) ve Hard Mining Algoritması

Makine öğrenmesinde "çöp giren, çöp çıkar". Modeli eğitmek için `pairing/pair_generator.py` içindeki vektörize edilmiş akıllı çift üretim motoru kullanılır.

1. **Vektörize Numpy Üretimi:** `for` döngüleri yerine `np.random.RandomState` ile milyonlarca indeks eşleşmesi anında oluşturulur.
2. **O(1) Etiket Araması:** Eskiden 104 sınıf içerisinde O(n) ile aranan etiketler, `label_to_idx` dict map ile O(1) maliyete indirilmiştir. Çift üretim hızı 100x artmıştır.
3. **Hard Negative Mining (%30):** Rastgele negatif (farklı sınıf) üretmek yerine, havuzun %30'u *uzunlukları birbirine en çok benzeyen farklı problem sınıflarından* seçilir. Model sadece uzunluğa bakarak "klon" dememeyi öğrenir.
4. **Hard Positive Mining (%30):** Rastgele pozitif (aynı sınıf) üretmek yerine, havuzun %30'u *uzunlukları birbirinden en çok farklı olan aynı problem sınıflarından* seçilir. Bu, modelin farklı yapıdaki kodları (Type-4) birbirine bağlamayı öğrenmesini sağlar.

---

## 6. Hızlı Başlangıç ve Komut Referansı (Nasıl Çalıştırılır?)

Tüm komutlar sanal ortam (venv) aktifken proje kök dizininden (`/home/tuncay/PycharmProjects/CodeDuplicationDetection`) çalıştırılmalıdır.

```bash
source .venv/bin/activate
```

### 6.1. Standart Eğitimler

En çok kullanacağınız komutlar şunlardır:

```bash
# Cascade XGBoost Eğitimi (1 Milyon Çift, GPU destekli, otomatik threshold)
python main.py --pairs 1000000 --device cuda

# Cascade Ensemble Eğitimi (Daha yüksek Type-4 başarısı)
python main.py --pairs 1000000 --model ensemble

# CodeBERT SSL ile En Gelişmiş Eğitim (Sadece güçlü donanımlar için)
python main.py --pairs 500000 --model ensemble --use-ssl --device cuda
```

### 6.2. Gerçekçi Sınıf Dengesi (Positive Ratio)
Varsayılan olarak model %50 klon, %50 klon olmayan veri ile eğitilir (Laboratuvar ortamı). Ancak gerçek dünyada klonlar çok daha azdır.
```bash
# Sadece %5 klon (Gerçekçi dengesizlik)
python main.py --pairs 1000000 --positive-ratio 0.05
```

### 6.3. Hiperparametre Ayarı (Tuning)
XGBoost için Optuna kullanarak en iyi parametreleri arayabilirsiniz.
```bash
python main.py --pairs 1000000 --tune --tune-trials 50
```

### 6.4. Cross-Validation
Sistemin veriyi ezberleyip ezberlemediğini (Data Leakage) anlamak için 5-Fold Stratified CV uygulayın. Kodlar fold'lara bölünür, çiftler ondan sonra üretilir.
```bash
python main.py --cv --cv-folds 5 --pairs 500000
```

---

## 7. Test Otomasyonu ve Değerlendirme

Sistemi eğitmek kadar doğru test etmek de önemlidir. `utils/` altındaki araçlar bunu otomatikleştirir.

### Adım 1: Test Verisi Oluşturun
Otomasyon için `test_clones/` klasörüne Type-1, 2, 3 ve 4 klonlarından oluşan saf test verileri oluşturmalısınız.
```bash
python utils/generate_test_clones.py --overwrite
```

### Adım 2: Otomatik Klon Tipi Testi Uygulayın
Eğitilen en son deneyi (klasördeki en yeni `exp_NNN`) otomatik test eder. Size Type-1'den Type-4'e kadar detaylı F1, Precision, Recall tabloları sunar.
```bash
python utils/test_automation.py

# Belirli bir deneyi ve eşiği (threshold) test etmek için
python utils/test_automation.py --exp-id 56 --threshold 0.90
```

### Adım 3: Deneyleri Karşılaştırın
Geçmişte eğittiğiniz birden fazla modeli yan yana kıyaslayın:
```bash
# Belirli deneyleri F1 skoruna göre yan yana tablo halinde listele
python utils/compare_experiments.py --exp-ids 54 55 56
```

---

## 8. Açıklanabilir Yapay Zeka: Web Arayüzü (Web Demo)

`web_demo/app.py` üzerinde **FastAPI** tabanlı gelişmiş bir REST API bulunmaktadır. Kod klonlarını tarayıcı arayüzünden test etmenizi ve sonuçları SHAP ile analiz etmenizi sağlar.

```bash
# En son deney modelini ayağa kaldır
uvicorn web_demo.app:app --reload

# Belirli bir deney numarasını (Örn. 56) ayağa kaldır
EXP_ID=56 uvicorn web_demo.app:app --reload --port 8080
```
Tarayıcınızdan `http://localhost:8000` adresine giderek arayüzü kullanabilirsiniz.

### API Uç Noktaları (Endpoints):
- `POST /predict`: İki C/C++ kodunu JSON olarak alır. Stage-1 ve Stage-2 sonucunu verir. Eğer Stage-2 (XGBoost) çalıştıysa **SHAP (SHapley Additive exPlanations)** değerlerini hesaplar. Hangi özelliğin (örn. SVD diff, IO pattern, cf_sim) modele ne kadar "klon" dedirttiğini veya "klon değil" dedirttiğini açıklar.
- `POST /predict_batch`: Maksimum 500 çifti kabul eden toplu çıkarım (Batch Inference) arayüzü.

---

## 9. Geliştirici Notları ve Optimizasyonlar

### 9.1. Intel ve İşlemci Optimizasyonları
Eğer sisteminiz CPU üzerinde çalışıyorsa (veya `device=cpu` verilmişse), `main.py` başlangıcında `sklearnex` yama kütüphanesi otomatik devreye girer. Scikit-learn (PCA, SVD, HistGBM) fonksiyonları Intel MKL kullanarak multi-threading ile donanıma gömülü hızlarda çalışır. Çıkarım sürelerini ~3x hızlandırır.

### 9.2. SSL Gömü Önbelleği (Disk Cache)
CodeBERT modelleri devasa olduğundan 1 milyon çift için gömü (embedding) çıkarmak saatler sürebilir.
Bu nedenle ilk çalıştırmada çıkarılan vektörler `ssl_cache.npy` içerisine kaydedilir (Pickle yerine çok daha hızlı olan Memory-Mapped NPY). 
Aynı komutu bir daha çalıştırdığınızda anında SSD'den belleğe (O(1) maliyetle) okunur.
```bash
python main.py --pairs 500000 --use-ssl --ssl-cache ssl_cache.npy
```

### 9.3. Memory Management (RAM Patlamalarını Önleme)
Projeyi 16GB - 32GB RAM'li sistemlerde çalıştırılabilir kılmak için `gc.collect()` (Garbage Collector) aktif olarak kullanılmıştır. `main.py` ve `pair_generator.py` içerisinde büyük TF-IDF matrisleri, SVD parçaları ve indeksleme array'leri kullanıldıktan *hemen* sonra `del` komutuyla bellekten atılır. 

### 9.4. Tüm Unit Testlerin Koşulması (CI/CD Uyumluluğu)
Sistemde 40+ üzeri unit test ve entegrasyon testi bulunmaktadır. Kod üzerinde değişiklik yaptıktan sonra her şeyin düzgün çalıştığından emin olmak için testleri koşunuz:
```bash
python -m pytest tests/ -v
```

---
*CodeDuplicationDetection - Mayıs 2026 Mimarisi.*
*Yapay Zeka destekli, ölçeklenebilir ve endüstri standardında kurumsal kod analizi.*
