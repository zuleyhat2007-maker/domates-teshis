# Domates Yaprağı Hastalık Teşhis Sistemi

Domates yaprağı fotoğraflarından hastalık tespiti yapan derin öğrenme projesi ve web arayüzü.
Üç model (özel CNN, ResNet50, VGG16) eğitilip karşılaştırılmış, en başarılı model web sitesine bağlanmıştır.

**Sınıflar:** Bakteriyel Leke · Sağlıklı · Septoria Yaprak Lekesi · Hedef Leke
**Veri seti:** [Tomato Leaf Disease (Kaggle, naveedgull)](https://www.kaggle.com/datasets/naveedgull/tomato-leaf-disease). Görüntüler %70 eğitim, %15 doğrulama ve %15 test olarak ayrılmıştır (seed 42).

---

## Sonuçlar (test seti, 830 görüntü)

| Model | Accuracy | Precision | Recall | F1 (ağırlıklı) | F1 (makro) |
|---|---|---|---|---|---|
| **VGG16** | **%93,98** | **%94,46** | **%93,98** | **%94,07** | **%93,60** |
| ResNet50 | %92,53 | %93,64 | %92,53 | %92,59 | %92,04 |
| Özel CNN | %70,24 | %83,97 | %70,24 | %69,14 | %67,20 |

**Sınıf bazında F1 (VGG16):** Bakteriyel Leke %97,6 · Sağlıklı %94,0 · Septoria %94,2 · Hedef Leke %88,6

ImageNet ile ön eğitilmiş modeller (transfer öğrenme), sıfırdan eğitilen CNN'e göre belirgin şekilde daha başarılıdır.
Grafikler, karışıklık matrisleri ve tüm detaylar web sitesinin **Performans Paneli** sayfasında ve `rapor/tubitak_rapor.html` dosyasında yer alır.

## Dağılım dışı görüntü testi ve domates filtresi

Sınıflandırma modelleri yalnızca dört sınıf bilir ve alakasız bir görüntüyü reddedemez.
Bunu göstermek için modellere bir kod ekranı fotoğrafı (`rapor/sacma.jpg`) verilmiştir. CNN ve ResNet50 bu görüntüyü yaklaşık %100 güvenle "Septoria", VGG16 ise yaklaşık %100 güvenle "Sağlıklı" olarak sınıflandırmıştır (`rapor/sacma_sonuc.csv`).

Bu sorunu çözmek için web sitesine sınıflandırmadan önce çalışan bir **domates filtresi** eklenmiştir:
1. Görüntüden ImageNet ile eğitilmiş MobileNetV2 kullanılarak bir özellik vektörü çıkarılır.
2. Bu vektör, eğitim setinden seçilen 1200 gerçek domates yaprağıyla karşılaştırılır (en yakın 5 komşunun ortalama kosinüs benzerliği).
3. Benzerlik eşiğin altındaysa görüntü **reddedilir** ve sınıflandırmaya hiç gönderilmez.

| Görüntü | Benzerlik | Sonuç (eşik 0,55) |
|---|---|---|
| Veri setindeki gerçek yapraklar (test) | en düşük 0,66 · ortalama 0,83 | %100 kabul |
| Kod ekranı fotoğrafı (sacma.jpg) | 0,32 | Reddedildi |

**Sınırlılık:** Veri setindeki fotoğraflar sade zeminde çekilmiş tek yapraklardan oluşur. Tarlada, karışık arka planla çekilen fotoğraflar daha düşük benzerlik alabilir. Eşik, Performans Paneli'nden ayarlanabilir.

---

## Klasör yapısı

```
DomatesTeshis/
├── app.py              Web sitesi (Flask)
├── model_utils.py      Ön işleme, model yükleme, domates filtresi
├── hazirla.py          Metrikleri ve filtreyi yeniden hesaplar (isteğe bağlı)
├── requirements.txt
├── models/             Eğitilmiş modeller (cnn, resnet, vgg .keras)
├── data/               Hazır test metrikleri ve filtre verisi
├── templates/, static/ Site sayfaları
├── notebook/           Eğitim notebook'u (.ipynb)
└── rapor/              Notebook raporu (HTML), saçma foto testi
```

## Çalıştırma

Python 3.11 önerilir.

```bash
cd DomatesTeshis
pip install -r requirements.txt
python app.py
```

Tarayıcıda **http://127.0.0.1:5050** adresini açın.

- **Teşhis sayfası:** Galeriden seçerek, dosya yükleyerek ya da kamerayla fotoğraf çekerek analiz yapılır. Domates yaprağı olmayan görüntüler reddedilir.
- **Performans Paneli (`/panel`):** Model metrikleri, accuracy karşılaştırma grafiği, karışıklık matrisi, sitede yapılan analizlerin canlı takibi, aktif model seçimi ve filtre eşiği ayarı.

Metrikler ve filtre verisi `data/` klasöründe hazır olduğu için `hazirla.py`'yi çalıştırmak gerekmez. Yeniden hesaplamak isterseniz `python hazirla.py` komutunu çalıştırın. Bu komut veri setini Kaggle'dan indirir ve birkaç dakika sürer.

*Sonuçlar bilgilendirme amaçlıdır; kesin teşhis için bir ziraat mühendisine danışılmalıdır.*
