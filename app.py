# -*- coding: utf-8 -*-
"""Domates Yaprağı Hastalık Teşhisi - Flask uygulaması.

Çalıştırma:  python app.py   ->  http://127.0.0.1:5050
"""
import hmac
import json
import os
import sqlite3
import threading
import time
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request

from model_utils import (BASE_DIR, CLASS_INFO, CLASS_TR, CLASSES, DB_PATH, GATE_PATH, METRICS_PATH,
                         MODEL_LABELS, SETTINGS_PATH, TomatoGate, load_models, open_image, predict_probs)

if not METRICS_PATH.exists() or not GATE_PATH.exists():
    raise SystemExit("data/metrics.json veya data/gate.npz bulunamadı. Yeniden oluşturmak için:  python hazirla.py")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
GATE_DATA = np.load(GATE_PATH)
REF, VAL_SCORES, TEST_SCORES = GATE_DATA["ref"], GATE_DATA["val_scores"], GATE_DATA["test_scores"]

MODELS = load_models()
GATE = TomatoGate()
LOCK = threading.Lock()
# Herkese açık yayında panel ayarlarını korumak için (Hugging Face'te "Secret" olarak tanımlanır)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


def admin_ok():
    return not ADMIN_PASSWORD or hmac.compare_digest(request.headers.get("X-Admin-Key", ""), ADMIN_PASSWORD)
THUMB_DIR = BASE_DIR / "static" / "uploads"
THUMB_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Ayarlar ----------------
DEFAULT_THRESHOLD = 0.55


def load_settings():
    s = {"active_model": next(iter(MODELS)), "threshold": DEFAULT_THRESHOLD}
    if SETTINGS_PATH.exists():
        s.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    s.pop("reject_percentile", None)
    if s["active_model"] not in MODELS:
        s["active_model"] = next(iter(MODELS))
    return s


SETTINGS = load_settings()


def threshold():
    return float(SETTINGS["threshold"])


def settings_payload():
    thr = threshold()
    return {
        "active_model": SETTINGS["active_model"],
        "threshold": thr,
        "sacma_score": METRICS["gate"].get("sacma_score"),
        "admin_required": bool(ADMIN_PASSWORD),
        "test_accept_rate": float((TEST_SCORES >= thr).mean()),
        "models": [{"key": k, "label": MODEL_LABELS[k],
                    "accuracy": METRICS["models"].get(k, {}).get("accuracy")} for k in MODELS],
    }


# ---------------- Kayıtlar (SQLite) ----------------
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


with db() as con:
    con.execute("""CREATE TABLE IF NOT EXISTS kayit (
        id TEXT PRIMARY KEY, ts REAL, accepted INTEGER, gate_score REAL, threshold REAL,
        model TEXT, pred TEXT, confidence REAL, thumb TEXT)""")


def save_thumb(img, rid):
    t = img.copy()
    t.thumbnail((160, 160))
    t.save(THUMB_DIR / f"{rid}.jpg", "JPEG", quality=80)
    return f"/static/uploads/{rid}.jpg"


def log(rec):
    with LOCK, db() as con:
        con.execute("INSERT INTO kayit VALUES (:id,:ts,:accepted,:gate_score,:threshold,:model,:pred,:confidence,:thumb)", rec)


# ---------------- Sayfalar ----------------
@app.route("/")
def index():
    return render_template("index.html", classes=[CLASS_TR[c] for c in CLASSES], page="index")


@app.route("/panel")
def panel():
    return render_template("panel.html", page="panel")


# ---------------- API ----------------
@app.post("/api/predict")
def api_predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="Görüntü gönderilmedi."), 400
    try:
        img = open_image(f.stream)
    except Exception:
        return jsonify(error="Dosya okunamadı. Lütfen JPG, PNG veya WEBP bir fotoğraf yükleyin."), 400

    rid = uuid.uuid4().hex[:12]
    thr = threshold()
    with LOCK:
        gscore = float(TomatoGate.score(GATE.embed([img]), REF)[0])
    accepted = gscore >= thr
    rec = {"id": rid, "ts": time.time(), "accepted": int(accepted), "gate_score": gscore,
           "threshold": thr, "model": SETTINGS["active_model"], "pred": None, "confidence": None,
           "thumb": save_thumb(img, rid)}

    if not accepted:
        log(rec)
        return jsonify(accepted=False, gate_score=gscore, threshold=thr,
                       message="Bu görüntü bir domates yaprağına benzemiyor. Sistem yalnızca domates "
                               "yaprağı fotoğraflarını analiz eder, bu yüzden görüntü reddedildi.")

    results = {}
    with LOCK:
        for k, m in MODELS.items():
            t = time.perf_counter()
            p = predict_probs(k, m, [img])[0]
            results[k] = {"probs": p, "ms": 1000 * (time.perf_counter() - t)}

    active = SETTINGS["active_model"]
    main = results[active]["probs"]
    ci = int(main.argmax())
    cls = CLASSES[ci]
    rec.update(pred=cls, confidence=float(main[ci]))
    log(rec)

    others = []
    for k, r in results.items():
        j = int(r["probs"].argmax())
        others.append({"key": k, "label": MODEL_LABELS[k], "pred": CLASSES[j], "pred_tr": CLASS_TR[CLASSES[j]],
                       "confidence": float(r["probs"][j]), "ms": r["ms"], "active": k == active})
    return jsonify(
        accepted=True, gate_score=gscore, threshold=thr,
        model=active, model_label=MODEL_LABELS[active],
        pred=cls, pred_tr=CLASS_TR[cls], healthy=cls == "Healthy", confidence=float(main[ci]),
        probs=[{"cls": c, "cls_tr": CLASS_TR[c], "p": float(main[i])} for i, c in enumerate(CLASSES)],
        info=CLASS_INFO[cls],
        agreement=sum(o["pred"] == cls for o in others), n_models=len(others), others=others,
    )


@app.get("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.get("/api/stats")
def api_stats():
    now = time.time()
    with db() as con:
        rows = con.execute("SELECT * FROM kayit ORDER BY ts DESC").fetchall()
    acc = [r for r in rows if r["accepted"]]
    per_class = {c: 0 for c in CLASSES}
    for r in acc:
        per_class[r["pred"]] = per_class.get(r["pred"], 0) + 1
    # son 30 dakika, dakika başına analiz sayısı
    timeline = [{"t": int(now // 60 - i) * 60, "kabul": 0, "ret": 0} for i in range(29, -1, -1)]
    start = timeline[0]["t"]
    for r in rows:
        idx = int(r["ts"] // 60 * 60 - start) // 60
        if 0 <= idx < 30:
            timeline[idx]["kabul" if r["accepted"] else "ret"] += 1
    recent = [{
        "ts": r["ts"], "accepted": bool(r["accepted"]), "pred_tr": CLASS_TR.get(r["pred"], "-") if r["pred"] else None,
        "confidence": r["confidence"], "gate_score": r["gate_score"], "model": MODEL_LABELS.get(r["model"], r["model"]),
        "thumb": r["thumb"]} for r in rows[:12]]
    return jsonify(
        server_time=now,
        total=len(rows), accepted=len(acc), rejected=len(rows) - len(acc),
        mean_confidence=float(np.mean([r["confidence"] for r in acc])) if acc else None,
        per_class=[{"cls": c, "cls_tr": CLASS_TR[c], "n": per_class[c]} for c in CLASSES],
        timeline=timeline, recent=recent, settings=settings_payload(),
    )


@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        if not admin_ok():
            return jsonify(error="Yönetici şifresi yanlış."), 401
        body = request.get_json(force=True, silent=True) or {}
        if body.get("active_model") in MODELS:
            SETTINGS["active_model"] = body["active_model"]
        if "threshold" in body:
            try:
                SETTINGS["threshold"] = round(float(min(0.95, max(0.2, float(body["threshold"])))), 3)
            except (TypeError, ValueError):
                return jsonify(error="Geçersiz eşik değeri."), 400
        SETTINGS_PATH.write_text(json.dumps(SETTINGS), encoding="utf-8")
    return jsonify(settings_payload())


@app.post("/api/reset")
def api_reset():
    if not admin_ok():
        return jsonify(error="Yönetici şifresi yanlış."), 401
    with LOCK, db() as con:
        con.execute("DELETE FROM kayit")
    for p in THUMB_DIR.glob("*.jpg"):
        p.unlink(missing_ok=True)
    return jsonify(ok=True)


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Dosya çok büyük (en fazla 25 MB)."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\nSite hazır:  http://127.0.0.1:{port}   (Performans paneli: /panel)\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
