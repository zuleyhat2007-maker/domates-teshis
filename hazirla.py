# -*- coding: utf-8 -*-
"""
Site için hazırlık (bir kez çalıştırılır, birkaç dakika sürer):
  1) Notebook'taki train/val/test ayrımını aynen yeniden oluşturur (aynı seed, aynı oranlar)
  2) Üç modeli test setinde değerlendirir -> data/metrics.json
  3) Domates filtresini gerçek yaprak görüntüleriyle kalibre eder -> data/gate.npz
"""
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from model_utils import (CLASSES, DATA_DIR, GATE_K, GATE_PATH, METRICS_PATH, MODEL_DIR,
                         MODEL_LABELS, SETTINGS_PATH, TomatoGate, load_models, open_image,
                         predict_probs)

# ---- Notebook ile birebir aynı ayarlar ----
SPLIT_BASE = Path(os.environ.get("SPLIT_BASE", "/tmp/tomato_split"))
TRAIN_RATIO, VAL_RATIO, RANDOM_SEED = 0.70, 0.15, 42
CLASS_MAP = {
    "Tomato___healthy": "Healthy",
    "Tomato___Septoria_leaf_spot": "Septoria",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Bacterial_spot": "Bacterial Spot",
}
EXTS = (".jpg", ".jpeg", ".png")
BATCH = 32
GATE_REF_PER_CLASS = 300
DEFAULT_REJECT_PERCENTILE = 2.0


def train_path():
    if os.environ.get("DATASET_TRAIN"):
        return Path(os.environ["DATASET_TRAIN"])
    import kagglehub
    return Path(kagglehub.dataset_download("naveedgull/tomato-leaf-disease")) / "Tomato Leaf Disease" / "train"


def split_ok():
    return all((SPLIT_BASE / s / c).is_dir() and any((SPLIT_BASE / s / c).iterdir())
               for s in ("train", "val", "test") for c in CLASSES)


def ensure_split():
    if split_ok():
        print(f"Mevcut ayrım kullanılıyor: {SPLIT_BASE}")
        return
    print("Notebook'taki ayrım yeniden oluşturuluyor...")
    src_root = train_path()
    random.seed(RANDOM_SEED)
    if SPLIT_BASE.exists():
        shutil.rmtree(SPLIT_BASE)
    for raw_cls, clean in CLASS_MAP.items():
        src = src_root / raw_cls
        imgs = [f for f in os.listdir(src) if f.lower().endswith(EXTS)]
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val = int(n * TRAIN_RATIO), int(n * VAL_RATIO)
        parts = {"train": imgs[:n_train], "val": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}
        for split, files in parts.items():
            dest = SPLIT_BASE / split / clean
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(src / f, dest / f)


def list_split(split):
    items = []
    for ci, c in enumerate(CLASSES):
        folder = SPLIT_BASE / split / c
        items += [(folder / f, ci) for f in sorted(os.listdir(folder)) if f.lower().endswith(EXTS)]
    return items


def batches(items, size=BATCH):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def evaluate(models, test_items):
    y_true = np.array([ci for _, ci in test_items])
    probs = {k: [] for k in models}
    times = {k: 0.0 for k in models}
    for i, chunk in enumerate(batches(test_items)):
        imgs = [open_image(p) for p, _ in chunk]
        for k, m in models.items():
            t = time.perf_counter()
            probs[k].append(predict_probs(k, m, imgs))
            times[k] += time.perf_counter() - t
        print(f"\r  Test görüntüleri: {min((i + 1) * BATCH, len(test_items))}/{len(test_items)}", end="")
    print()
    out = {}
    for k in models:
        p = np.concatenate(probs[k])
        y_pred = p.argmax(1)
        rep = classification_report(y_true, y_pred, labels=range(len(CLASSES)), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
        out[k] = {
            "label": MODEL_LABELS[k],
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mean_confidence": float(p.max(1).mean()),
            "ms_per_image": 1000 * times[k] / len(test_items),
            "per_class": {c: {"precision": rep[c]["precision"], "recall": rep[c]["recall"],
                              "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])} for c in CLASSES},
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
        }
        print(f"  {MODEL_LABELS[k]:9s} accuracy={out[k]['accuracy']:.4f}  f1={out[k]['f1']:.4f}")
    return out


def embed_items(gate, items):
    embs = []
    for chunk in batches(items):
        embs.append(gate.embed([open_image(p) for p, _ in chunk]))
    return np.concatenate(embs)


def calibrate_gate(gate):
    rng = random.Random(0)
    ref_items = []
    for ci in range(len(CLASSES)):
        cls_items = [it for it in list_split("train") if it[1] == ci]
        ref_items += rng.sample(cls_items, min(GATE_REF_PER_CLASS, len(cls_items)))
    print(f"  Referans: {len(ref_items)} eğitim görüntüsü")
    ref = embed_items(gate, ref_items)
    val_scores = np.sort(TomatoGate.score(embed_items(gate, list_split("val")), ref))
    test_scores = TomatoGate.score(embed_items(gate, list_split("test")), ref)
    thr = float(np.percentile(val_scores, DEFAULT_REJECT_PERCENTILE))
    info = {
        "k": GATE_K,
        "reference_images": len(ref_items),
        "default_percentile": DEFAULT_REJECT_PERCENTILE,
        "default_threshold": thr,
        "test_accept_rate": float((test_scores >= thr).mean()),
        "val_score_range": [float(val_scores.min()), float(val_scores.max())],
    }
    sacma = MODEL_DIR.parent / "rapor" / "sacma.jpg"
    if sacma.exists():
        s = float(TomatoGate.score(gate.embed([open_image(sacma)]), ref)[0])
        info["sacma_score"] = s
        info["sacma_rejected"] = s < thr
        print(f"  sacma.jpg benzerlik={s:.3f}  eşik={thr:.3f}  ->  {'REDDEDİLDİ' if s < thr else 'KABUL EDİLDİ'}")
    print(f"  Gerçek domates test görüntülerinin kabul oranı: %{100 * info['test_accept_rate']:.1f}")
    np.savez_compressed(GATE_PATH, ref=ref.astype(np.float32), val_scores=val_scores, test_scores=test_scores)
    return info


def main():
    ensure_split()
    counts = {s: {c: len(os.listdir(SPLIT_BASE / s / c)) for c in CLASSES} for s in ("train", "val", "test")}
    print("Görüntü sayıları:", json.dumps(counts, ensure_ascii=False))

    print("\n[1/2] Modeller test setinde değerlendiriliyor")
    models = load_models()
    model_metrics = evaluate(models, list_split("test"))

    print("\n[2/2] Domates filtresi kalibre ediliyor")
    gate_info = calibrate_gate(TomatoGate())

    metrics = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "classes": CLASSES,
        "dataset": counts,
        "models": model_metrics,
        "gate": gate_info,
    }
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    best = max(model_metrics, key=lambda k: model_metrics[k]["accuracy"])
    SETTINGS_PATH.write_text(json.dumps({"active_model": best, "threshold": 0.55}), encoding="utf-8")
    print(f"\nHazır. En iyi model: {MODEL_LABELS[best]}. Şimdi siteyi başlat:  python app.py")


if __name__ == "__main__":
    main()
