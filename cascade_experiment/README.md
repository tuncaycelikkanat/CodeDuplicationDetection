# Cascade Architecture Experiment

Bu klasör, projenin ana mimarisine dokunmadan tamamen yeni bir makine öğrenmesi testinin yapıldığı yerdir.

## Neden Cascade?
Geleneksel eğitimde XGBoost hem kolay kopya kodları (Type 1, Type 2) hem de tamamen farklı yazılmış ama aynı mantığı kuran (Type 4) kodları ayırt etmeye çalışır. "Kolay" olanları ayırt etmek için TF-IDF gibi kelime-kök modelleri yetiyorken, bu kolay kodları modele göstermek XGBoost'un tembelleşmesine yol açar.

**Çözüm:**
1. Veriyi `generate_pairs` ile ürettikten hemen sonra, kelime benzerliği (`cosine_similarity`) > 0.85 olan "Kolay Klonları" tespit et ve **eğitim setinden çıkart**.
2. XGBoost'u sadece ve sadece kelimeleri birbirinden tamamen farklı olan kodlar (Zor / Type-4 kodlar) üzerinde eğit.
3. Çıkarım (Test / Inference) esnasında iki aşamalı çalış:
   - Kodları kıyasla. Eğer kelime benzerliği > 0.85 ise XGBoost'a hiç sorma, direkt "KLON (1)" de.
   - Eğer benzerlik < 0.85 ise XGBoost'a sor.

## Çalıştırma
Bu yeni eğitim modülünü test etmek için:
```bash
python cascade_experiment/cascade_main.py --pairs 2000
```

Eğer bu yaklaşım `main.py`'den daha başarılı sonuçlar verirse, ileride bu mantığı `main.py` ve `app.py` içerisine entegre edebiliriz. Şimdilik burası bizim laboratuvarımızdır.
