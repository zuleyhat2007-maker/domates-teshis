# -*- coding: utf-8 -*-
"""Ortak yardımcılar: görüntü yükleme, model ön işleme, domates filtresi."""
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

METRICS_PATH = DATA_DIR / "metrics.json"
GATE_PATH = DATA_DIR / "gate.npz"
SETTINGS_PATH = DATA_DIR / "settings.json"
DB_PATH = DATA_DIR / "kayitlar.sqlite"

# flow_from_directory alfabetik sıraladığı için modellerin çıktı sırası bu
CLASSES = ["Bacterial Spot", "Healthy", "Septoria", "Target Spot"]
CLASS_TR = {
    "Bacterial Spot": "Bakteriyel Leke",
    "Healthy": "Sağlıklı",
    "Septoria": "Septoria Yaprak Lekesi",
    "Target Spot": "Hedef Leke",
}
CLASS_INFO = {
    "Bacterial Spot": "Xanthomonas bakterilerinin neden olduğu hastalık. Yapraklarda küçük, koyu, "
                      "su emmiş gibi görünen lekeler oluşur; ileride lekelerin ortası kuruyup dökülebilir. "
                      "Hastalıklı yaprakları uzaklaştırın, yapraktan (üstten) sulamadan kaçının.",
    "Healthy": "Yaprakta belirgin bir hastalık belirtisi görülmedi. Düzenli kontrol ve dengeli "
               "sulamayla bitkiyi izlemeye devam edin.",
    "Septoria": "Septoria lycopersici mantarının neden olduğu hastalık. Genellikle alt yapraklardan "
                "başlayan, gri-beyaz merkezli ve koyu kenarlı küçük yuvarlak lekelerle görülür. "
                "Hastalıklı yaprakları toplayın, bitkiler arasında hava akışını artırın.",
    "Target Spot": "Corynespora cassiicola mantarının neden olduğu hastalık. Yapraklarda iç içe halkalar "
                   "hâlinde, hedef tahtasına benzeyen kahverengi lekeler oluşur. Nemi azaltın, "
                   "bitki artıklarını temizleyin.",
}

MODEL_FILES = {"cnn": "cnn_model.keras", "resnet": "resnet_model.keras", "vgg": "vgg_model.keras"}
MODEL_LABELS = {"cnn": "Özel CNN", "resnet": "ResNet50", "vgg": "VGG16"}

GATE_SIZE = (224, 224)
GATE_K = 5


def open_image(src):
    """Dosya yolu ya da dosya nesnesinden RGB PIL görüntüsü (telefon fotoğraflarındaki dönmeyi düzeltir)."""
    img = Image.open(src)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def to_batch(images, size_hw):
    h, w = size_hw
    return np.stack([np.asarray(im.resize((w, h), Image.BILINEAR), dtype=np.float32) for im in images])


def preprocess(key, batch):
    """Her model eğitimde hangi ön işlemeyi gördüyse aynısı."""
    if key == "cnn":
        return batch / 255.0
    if key == "resnet":
        from tensorflow.keras.applications.resnet50 import preprocess_input
        return preprocess_input(batch.copy())
    if key == "vgg":
        from tensorflow.keras.applications.vgg16 import preprocess_input
        return preprocess_input(batch.copy())
    raise ValueError(key)


def load_models():
    import tensorflow as tf
    models = {}
    for key, fname in MODEL_FILES.items():
        path = MODEL_DIR / fname
        if path.exists():
            print(f"Model yükleniyor: {fname}")
            models[key] = tf.keras.models.load_model(path, compile=False)
        else:
            print(f"UYARI: {path} bulunamadı, {MODEL_LABELS[key]} atlanıyor")
    if not models:
        raise SystemExit(f"Hiç model bulunamadı. .keras dosyaları {MODEL_DIR} içinde olmalı.")
    return models


def predict_probs(key, model, images):
    h, w = model.input_shape[1:3]
    x = preprocess(key, to_batch(images, (h, w)))
    return model.predict(x, verbose=0)


class TomatoGate:
    """Domates yaprağı filtresi.

    ImageNet ile eğitilmiş MobileNetV2'den görsel özellik vektörü çıkarır ve yüklenen
    görüntünün eğitim setindeki gerçek domates yapraklarına ne kadar benzediğini ölçer
    (en yakın 5 komşunun ortalama kosinüs benzerliği). Benzerlik eşiğin altındaysa
    görüntü sınıflandırmaya gönderilmeden reddedilir.
    """

    def __init__(self):
        from tensorflow.keras.applications import MobileNetV2
        weights = os.environ.get("GATE_WEIGHTS", "imagenet")
        self.net = MobileNetV2(include_top=False, pooling="avg", input_shape=(*GATE_SIZE, 3),
                               weights=None if weights == "none" else weights)

    def embed(self, images):
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x = preprocess_input(to_batch(images, GATE_SIZE))
        e = self.net.predict(x, verbose=0)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    @staticmethod
    def score(emb, ref, k=GATE_K):
        sims = emb @ ref.T
        k = min(k, ref.shape[0])
        return np.sort(sims, axis=1)[:, -k:].mean(axis=1)
