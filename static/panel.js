(() => {
  const $ = (id) => document.getElementById(id);
  const pct = (v, d = 1) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const MODEL_COLOR = { cnn: "#c98a22", resnet: "#2f7d4f", vgg: "#3b6ea8" };
  const CLASS_TR = { "Bacterial Spot": "Bakteriyel Leke", Healthy: "Sağlıklı", Septoria: "Septoria", "Target Spot": "Hedef Leke" };
  const CLASS_COLOR = ["#c8452d", "#2f7d4f", "#c98a22", "#3b6ea8"];

  Chart.defaults.font.family = '"Plus Jakarta Sans", system-ui, sans-serif';
  Chart.defaults.color = "#5d6a61";
  Chart.defaults.borderColor = "#ecebe4";
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.maintainAspectRatio = false;

  // Çubukların üstüne değer yazan küçük eklenti
  const valueLabels = {
    id: "valueLabels",
    afterDatasetsDraw(chart, _a, opts) {
      if (!opts?.on) return;
      const { ctx } = chart;
      ctx.save();
      ctx.font = '700 12px "Plus Jakarta Sans", system-ui';
      ctx.fillStyle = "#18241c";
      ctx.textAlign = "center";
      chart.data.datasets.forEach((ds, i) => {
        chart.getDatasetMeta(i).data.forEach((bar, j) => {
          ctx.fillText((ds.data[j] * 100).toFixed(1) + "%", bar.x, bar.y - 6);
        });
      });
      ctx.restore();
    },
  };
  const pctAxis = (min = 0) => ({ min, max: 1, ticks: { callback: (v) => Math.round(v * 100) + "%" } });
  const lowBound = (vals) => Math.max(0, Math.floor((Math.min(...vals) - 0.08) * 10) / 10);

  let METRICS = null, SETTINGS = null, cmModel = null;
  const charts = {};

  // ================= MODEL PERFORMANSI =================
  function renderMetrics() {
    const M = METRICS.models, keys = Object.keys(M), labels = keys.map((k) => M[k].label);
    const nTest = Object.values(METRICS.dataset.test).reduce((a, b) => a + b, 0);
    const nTrain = Object.values(METRICS.dataset.train).reduce((a, b) => a + b, 0);
    const nVal = Object.values(METRICS.dataset.val).reduce((a, b) => a + b, 0);
    $("testInfo").textContent = `Test seti: ${nTest} görüntü · Eğitim: ${nTrain} · Doğrulama: ${nVal} · Ölçüm: ${METRICS.created}`;

    const accs = keys.map((k) => M[k].accuracy);
    charts.acc = new Chart($("chAcc"), {
      type: "bar",
      data: { labels, datasets: [{ label: "Accuracy", data: accs, backgroundColor: keys.map((k) => MODEL_COLOR[k]), borderRadius: 8, maxBarThickness: 70 }] },
      options: { plugins: { legend: { display: false }, valueLabels: { on: true }, tooltip: { callbacks: { label: (c) => " " + pct(c.raw, 2) } } },
        scales: { y: pctAxis(lowBound(accs)), x: { grid: { display: false } } }, layout: { padding: { top: 22 } } },
      plugins: [valueLabels],
    });

    const metricNames = [["accuracy", "Accuracy"], ["precision", "Precision"], ["recall", "Recall"], ["f1", "F1"]];
    const allVals = keys.flatMap((k) => metricNames.map(([m]) => M[k][m]));
    charts.all = new Chart($("chAll"), {
      type: "bar",
      data: { labels: metricNames.map((m) => m[1]),
        datasets: keys.map((k) => ({ label: M[k].label, data: metricNames.map(([m]) => M[k][m]), backgroundColor: MODEL_COLOR[k], borderRadius: 6 })) },
      options: { plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${pct(c.raw, 2)}` } } },
        scales: { y: pctAxis(lowBound(allVals)), x: { grid: { display: false } } } },
    });

    const classVals = keys.flatMap((k) => METRICS.classes.map((c) => M[k].per_class[c].f1));
    charts.cls = new Chart($("chClass"), {
      type: "bar",
      data: { labels: METRICS.classes.map((c) => CLASS_TR[c] || c),
        datasets: keys.map((k) => ({ label: M[k].label, data: METRICS.classes.map((c) => M[k].per_class[c].f1), backgroundColor: MODEL_COLOR[k], borderRadius: 6 })) },
      options: { plugins: { tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${pct(c.raw, 2)}` } } },
        scales: { y: pctAxis(lowBound(classVals)), x: { grid: { display: false } } } },
    });

    // karışıklık matrisi sekmeleri
    $("cmTabs").innerHTML = keys.map((k) => `<button data-k="${k}">${esc(M[k].label)}</button>`).join("");
    $("cmTabs").onclick = (e) => { const k = e.target.dataset.k; if (k) { cmModel = k; renderCM(); } };

    const best = keys.reduce((a, b) => (M[b].accuracy > M[a].accuracy ? b : a));
    const col = (m, d = 2) => { const top = Math.max(...keys.map((k) => M[k][m])); return keys.map((k) => `<td class="num ${M[k][m] === top ? "best" : ""}">${pct(M[k][m], d)}</td>`); };
    const rows = [["Accuracy", col("accuracy")], ["Precision (ağırlıklı)", col("precision")], ["Recall (ağırlıklı)", col("recall")],
      ["F1 (ağırlıklı)", col("f1")], ["F1 (makro)", col("f1_macro")], ["Ortalama güven", col("mean_confidence")]];
    $("modelTable").innerHTML = `<thead><tr><th>Metrik</th>${keys.map((k) => `<th>${esc(M[k].label)}${k === best ? " ★" : ""}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(([n, cells]) => `<tr><td>${n}</td>${cells.join("")}</tr>`).join("")}
      <tr><td>Görüntü başına süre</td>${keys.map((k) => `<td class="num">${M[k].ms_per_image.toFixed(1)} ms</td>`).join("")}</tr></tbody>`;

  }

  function renderKpis() {
    const k = SETTINGS.active_model, m = METRICS.models[k];
    if (!m) return;
    $("kAcc").textContent = pct(m.accuracy); $("kPrec").textContent = pct(m.precision);
    $("kRec").textContent = pct(m.recall); $("kF1").textContent = pct(m.f1);
    $("kAccModel").textContent = `aktif model: ${m.label}`;
    if (!cmModel) { cmModel = k; renderCM(); }
  }

  function renderCM() {
    const m = METRICS.models[cmModel], cls = METRICS.classes, cm = m.confusion_matrix;
    [...$("cmTabs").children].forEach((b) => b.classList.toggle("on", b.dataset.k === cmModel));
    const max = Math.max(...cm.flat());
    let html = `<div class="h"></div>` + cls.map((c) => `<div class="h">${esc(CLASS_TR[c] || c)}</div>`).join("");
    cm.forEach((row, i) => {
      html += `<div class="h rh">${esc(CLASS_TR[cls[i]] || cls[i])}</div>`;
      row.forEach((v, j) => {
        const a = max ? v / max : 0;
        const bg = i === j ? `rgba(47,125,79,${0.12 + 0.75 * a})` : v ? `rgba(200,69,45,${0.12 + 0.6 * a})` : "#f4f3ee";
        const fg = a > 0.55 ? "#fff" : "#18241c";
        html += `<div style="background:${bg};color:${fg};font-weight:${i === j ? 800 : 500}">${v}</div>`;
      });
    });
    $("cm").style.gridTemplateColumns = `minmax(90px,auto) repeat(${cls.length}, minmax(60px,1fr))`;
    $("cm").innerHTML = html;
  }

  // ================= CANLI İZLEME =================
  function renderStats(s) {
    $("sTotal").textContent = s.total;
    $("sAcc").textContent = s.accepted;
    $("sRej").textContent = s.rejected;
    $("sRejRate").textContent = s.total ? `toplamın %${((100 * s.rejected) / s.total).toFixed(0)}'i` : "";
    $("sConf").textContent = pct(s.mean_confidence);

    const tl = s.timeline, tlLabels = tl.map((p) => new Date(p.t * 1000).toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" }));
    if (!charts.tl) {
      charts.tl = new Chart($("chTimeline"), {
        type: "bar",
        data: { labels: tlLabels, datasets: [
          { label: "Kabul", data: tl.map((p) => p.kabul), backgroundColor: "#2f7d4f", borderRadius: 4, stack: "a" },
          { label: "Ret", data: tl.map((p) => p.ret), backgroundColor: "#c8452d", borderRadius: 4, stack: "a" }] },
        options: { animation: false, scales: { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { grid: { display: false }, ticks: { maxTicksLimit: 8 } } } },
      });
    } else {
      charts.tl.data.labels = tlLabels;
      charts.tl.data.datasets[0].data = tl.map((p) => p.kabul);
      charts.tl.data.datasets[1].data = tl.map((p) => p.ret);
      charts.tl.update("none");
    }

    const distData = [...s.per_class.map((c) => c.n), s.rejected];
    const distLabels = [...s.per_class.map((c) => c.cls_tr), "Reddedildi"];
    if (!charts.dist) {
      charts.dist = new Chart($("chDist"), {
        type: "doughnut",
        data: { labels: distLabels, datasets: [{ data: distData, backgroundColor: [...CLASS_COLOR, "#b9bdb5"], borderWidth: 2, borderColor: "#fff" }] },
        options: { cutout: "62%", plugins: { legend: { position: "right" } } },
      });
    } else {
      charts.dist.data.datasets[0].data = distData;
      charts.dist.update();
    }

    $("recentEmpty").hidden = s.recent.length > 0;
    $("recentTable").innerHTML = !s.recent.length ? "" :
      `<thead><tr><th></th><th>Zaman</th><th>Durum</th><th>Teşhis</th><th>Güven</th><th>Benzerlik</th><th>Model</th></tr></thead><tbody>` +
      s.recent.map((r) => `<tr>
        <td><img src="${esc(r.thumb)}" alt=""></td>
        <td class="num">${new Date(r.ts * 1000).toLocaleTimeString("tr-TR")}</td>
        <td>${r.accepted ? '<span class="tag ok">Kabul</span>' : '<span class="tag bad">Reddedildi</span>'}</td>
        <td>${r.accepted ? `<span class="tag ${r.pred_tr === "Sağlıklı" ? "ok" : "sick"}">${esc(r.pred_tr)}</span>` : "–"}</td>
        <td class="num">${r.accepted ? pct(r.confidence) : "–"}</td>
        <td class="num">${r.gate_score.toFixed(3)}</td>
        <td>${esc(r.model)}</td></tr>`).join("") + "</tbody>";
  }

  // ================= KONTROL =================
  let dirty = false;
  function renderSettings(st, force = false) {
    SETTINGS = st;
    renderKpis();
    $("adminRow").hidden = !st.admin_required;
    if (dirty && !force) return; // kullanıcı değişiklik yaparken üzerine yazma
    $("modelPick").innerHTML = st.models.map((m) => `
      <label class="model-opt ${m.key === st.active_model ? "on" : ""}">
        <span style="display:flex;align-items:center;gap:10px;color:inherit;font-weight:600">
          <input type="radio" name="am" value="${m.key}" ${m.key === st.active_model ? "checked" : ""}>${esc(m.label)}</span>
        <span>Accuracy ${pct(m.accuracy)}</span></label>`).join("");
    $("rejRange").value = st.threshold;
    $("rejVal").textContent = st.threshold.toFixed(2);
    $("gAccRate").textContent = pct(st.test_accept_rate);
    renderSacma(st.threshold);
  }
  $("modelPick").addEventListener("change", () => {
    dirty = true;
    [...$("modelPick").children].forEach((l) => l.classList.toggle("on", l.querySelector("input").checked));
  });
  $("rejRange").addEventListener("input", (e) => {
    dirty = true;
    const t = parseFloat(e.target.value);
    $("rejVal").textContent = t.toFixed(2);
    renderSacma(t);
  });
  function renderSacma(t) {
    const s = METRICS?.gate?.sacma_score;
    $("gSacma").innerHTML = s == null ? "–" : (s < t ? `<span class="tag ok">Reddedildi ✓</span>` : `<span class="tag bad">Kabul edilir!</span>`) + ` <span class="muted small">(${s.toFixed(2)})</span>`;
  }

  $("btnSave").onclick = async () => {
    const am = document.querySelector('input[name="am"]:checked')?.value;
    const res = await fetch("/api/settings", { method: "POST", headers: { "Content-Type": "application/json", "X-Admin-Key": $("adminKey").value },
      body: JSON.stringify({ active_model: am, threshold: parseFloat($("rejRange").value) }) });
    const st = await res.json();
    if (!res.ok) { $("saveMsg").textContent = st.error; return; }
    dirty = false;
    renderSettings(st, true);
    $("saveMsg").textContent = `Kaydedildi · ${new Date().toLocaleTimeString("tr-TR")}`;
  };
  $("btnReset").onclick = async () => {
    if (!window.confirm("Tüm analiz kayıtları silinsin mi?")) return;
    const res = await fetch("/api/reset", { method: "POST", headers: { "X-Admin-Key": $("adminKey").value } });
    $("saveMsg").textContent = res.ok ? "Kayıtlar sıfırlandı." : (await res.json()).error;
    poll();
  };

  // ================= Döngü =================
  async function poll() {
    try {
      const s = await (await fetch("/api/stats", { cache: "no-store" })).json();
      renderSettings(s.settings);
      renderStats(s);
      document.querySelector(".live").classList.remove("off");
      $("lastUpdate").textContent = "son güncelleme " + new Date().toLocaleTimeString("tr-TR");
    } catch {
      document.querySelector(".live").classList.add("off");
      $("lastUpdate").textContent = "sunucuya ulaşılamıyor";
    }
  }

  (async () => {
    METRICS = await (await fetch("/api/metrics")).json();
    renderMetrics();
    await poll();
    setInterval(poll, 3000);
  })();
})();
