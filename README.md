# CodeDuplicationDetection

C/C++ kaynak kodları için **4 farklı klon tipini** (Type 1–4) yüksek doğrulukla tespit eden makine öğrenmesi tabanlı, derin öğrenme (CodeBERT) destekli kod analiz sistemi.

Bu proje, kod benzerliklerini sadece kelime (lexical) düzeyinde değil, Soyut Sözdizimi Ağacı (AST) ve Anlamsal (Semantic) düzeyde inceleyerek en zorlu Type-4 klonlarını dahi başarıyla tespit edecek Two-Stage (İki Aşamalı) Cascade mimarisine sahiptir.

## Proje Yapısı

```text
CodeDuplicationDetection/
├── config.py                    # Merkezi sabitler (CASCADE, STAGE1, SSL_PCA, ...)
├── main.py                      # AKTiF EGITIM (Two-Stage Cascade Egitim Betigi)
├── requirements.txt             # Temel bagimliliklar
├── requirements-gpu.txt         # GPU / CodeBERT SSL bagimlilikları
│
├── preprocessing/
│   ├── tokenizer.py               # C/C++ akıllı tokenizer (Yorum temizleme + Normalizasyon)
│   ├── code_features.py           # AST / CF / Semantik özellik çıkarımı (MultiSet Frekans destekli)
│   └── tree_sitter_parser.py      # Tree-sitter C++ parser
│
├── pairing/
│   └── pair_generator.py          # Kod cifti olusturma (Vektörize Hard Positive/Negative Mining)
│
├── vectorization/
│   ├── tfidf.py                   # Token TF-IDF ve SVD modelleyici
│   └── ssl_encoder.py             # CodeBERT Mean-Pooled embedding çıkarımı (Chunking + Dinamik PCA)
│
├── models/
│   ├── ensemble.py                # Feature-Partitioned Stacking Ensemble
│   └── xgboost.py                 # XGBoost model sarmalayici (GPU destekli + aucpr)
│
├── utils/
│   ├── feature_pipeline.py        # Tekli cift icin özellik çıkarımı (demo/test)
│   ├── test_automation.py         # Otomatik klon tipi başarım testi
│   └── ...                        # Deney kaydediciler ve tuner araçları
│
├── web_demo/
│   └── app.py                     # FastAPI web servisi (Açıklanabilir SHAP destekli)
│
└── data/poj104/                   # Veri seti (Problem sınıflarına ayrılmış)
```

## Mimari Özeti

Proje, gereksiz işlem yükünden kaçınmak ve model doğruluğunu artırmak için **Two-Stage Cascade** yaklaşımını kullanır:
1. **Stage-1 (HistGBM - Kalibre Edilmiş):** Hafif özelliklerle (Lexical/AST) çalışıp, bariz Type-1 ve Type-2 klonlarını (ve bariz negatifleri) %85 güven eşiğiyle hızla filtreden geçirir.
2. **Stage-2 (XGBoost veya Ensemble):** PCA ile indirgenmiş 64 boyutlu CodeBERT anlamsal farkları (SSL_PCA), SVD LSA matrisleri ve yapısal özellikleri harmanlayıp "zorlu vakalara" odaklanır.
   
Detaylı teknik bilgi ve işleyiş hakkında bilgi almak için **[REPORT.md](REPORT.md)** dosyasına göz atabilirsiniz.

## Kurulum

Sistem Python 3.12 ile çalışacak şekilde yapılandırılmıştır. (Gereksinimler için `uv` veya `pip` kullanabilirsiniz).

```bash
# Sanal ortam oluştur
python3.12 -m venv .venv
source .venv/bin/activate

# CPU bağımlılıklarını kur
pip install -r requirements.txt

# GPU desteği ve Derin Öğrenme modülleri için
pip install -r requirements-gpu.txt
```

## Örnek Kullanım

XGBoost ile CodeBERT özelliklerini kullanarak baştan sona veri seti eğitimi yapmak:
```bash
python main.py --pairs 100000 --model xgboost --use-ssl
```

Elde edilen model ile test klon klasörlerinde başarım ölçümü:
```bash
python utils/test_automation.py --threshold 0.85
```

Görsel arayüzü başlatmak (Modelin nasıl karar verdiğini görmek için SHAP açıklamaları içerir):
```bash
cd web_demo && uvicorn app:app --host 0.0.0.0 --port 8000
```
Tarayıcınızdan http://localhost:8000 adresini ziyaret edebilirsiniz.
