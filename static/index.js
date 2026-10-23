(() => {
  const $ = (id) => document.getElementById(id);
  const dz = $("dropzone"), preview = $("preview"), dzEmpty = $("dzEmpty");
  const btnAnalyze = $("btnAnalyze"), fileName = $("fileName");
  const rEmpty = $("resultEmpty"), rLoading = $("resultLoading"), rBody = $("resultBody");
  let current = null;
  let stream = null;

  const pct = (v) => (v * 100).toFixed(1) + "%";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function setFile(file) {
    if (!file) return;
    if (file.type && !file.type.startsWith("image/")) {
      showError("Lütfen bir görüntü dosyası seç (JPG, PNG, WEBP).");
      return;
    }
    current = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    dzEmpty.hidden = true;
    fileName.textContent = file.name ? `Seçilen: ${file.name}` : "Kameradan çekilen fotoğraf";
    btnAnalyze.disabled = false;
    rBody.hidden = true;
    rEmpty.hidden = false;
  }

  // ---- seçim yöntemleri ----
  $("btnGallery").onclick = () => $("galleryInput").click();
  $("btnFile").onclick = () => $("fileInput").click();
  ["galleryInput", "fileInput", "cameraInput"].forEach((id) => {
    $(id).addEventListener("change", (e) => { setFile(e.target.files[0]); e.target.value = ""; });
  });
  dz.onclick = () => { if (!current) $("galleryInput").click(); };
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault(); dz.classList.remove("drag");
    if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
  });
  document.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) setFile(item.getAsFile());
  });

  // ---- kamera ----
  const isTouch = window.matchMedia("(pointer: coarse)").matches;
  const dlg = $("camDialog"), video = $("camVideo"), camErr = $("camError");

  $("btnCamera").onclick = async () => {
    // Telefonda doğrudan cihazın kamera uygulaması açılır
    if (isTouch || !navigator.mediaDevices?.getUserMedia) { $("cameraInput").click(); return; }
    camErr.hidden = true; video.hidden = false; $("camShot").disabled = false;
    dlg.showModal();
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream;
    } catch (err) {
      video.hidden = true; $("camShot").disabled = true;
      camErr.textContent = "Kameraya erişilemedi. Tarayıcıdan kamera izni verip tekrar dene ya da Galeriden Seç'i kullan.";
      camErr.hidden = false;
    }
  };
  function closeCam() {
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    if (dlg.open) dlg.close();
  }
  $("camClose").onclick = closeCam;
  dlg.addEventListener("cancel", closeCam);
  $("camShot").onclick = () => {
    const c = $("camCanvas");
    c.width = video.videoWidth; c.height = video.videoHeight;
    c.getContext("2d").drawImage(video, 0, 0);
    c.toBlob((blob) => {
      setFile(new File([blob], "kamera.jpg", { type: "image/jpeg" }));
      closeCam();
    }, "image/jpeg", 0.92);
  };

  // ---- analiz ----
  btnAnalyze.onclick = async () => {
    if (!current) return;
    btnAnalyze.disabled = true;
    rEmpty.hidden = true; rBody.hidden = true; rLoading.hidden = false;
    const fd = new FormData();
    fd.append("image", current, current.name || "foto.jpg");
    try {
      const res = await fetch("/api/predict", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Sunucu hatası");
      render(data);
      if (window.innerWidth < 860) $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError(err.message || "Bağlantı hatası. Sunucu çalışıyor mu?");
    } finally {
      rLoading.hidden = true;
      btnAnalyze.disabled = false;
    }
  };

  function showError(msg) {
    rEmpty.hidden = true; rLoading.hidden = true;
    rBody.innerHTML = `<div class="verdict rej"><div class="ic">!</div><div><small>Hata</small><p>${esc(msg)}</p></div></div>`;
    rBody.hidden = false;
  }

  function gateMeter(d) {
    const clamp = (v) => Math.max(0, Math.min(100, v * 100));
    const w = clamp(d.gate_score), t = clamp(d.threshold);
    return `<div class="meter ${d.accepted ? "ok" : ""}">Domates yaprağına benzerlik: <b>${d.gate_score.toFixed(3)}</b> (eşik ${d.threshold.toFixed(3)})
      <div class="bar"><i style="width:${w}%"></i><span class="thr" style="left:${t}%"></span></div></div>`;
  }

  function render(d) {
    if (!d.accepted) {
      rBody.innerHTML = `
        <div class="verdict rej"><div class="ic">✕</div><div>
          <small>Görüntü reddedildi</small><h3>Domates yaprağı algılanmadı</h3>
          <p>${esc(d.message)}</p></div></div>
        ${gateMeter(d)}
        <ul class="rej-tips">
          <li>Fotoğrafta tek bir domates yaprağı olduğundan emin ol.</li>
          <li>Yaprağı yakından ve net çek; arka planda başka nesne olmasın.</li>
          <li>Doğal ışıkta, gölgesiz çekim daha iyi sonuç verir.</li>
        </ul>`;
      rBody.hidden = false;
      return;
    }
    const cls = d.healthy ? "ok" : "sick";
    const bars = [...d.probs].sort((a, b) => b.p - a.p).map((p, i) => `
      <div class="bar-row ${i === 0 ? "top" : ""}"><span>${esc(p.cls_tr)}</span>
        <div class="bar"><i style="width:${(p.p * 100).toFixed(1)}%"></i></div><b>${pct(p.p)}</b></div>`).join("");
    const others = d.others.map((o) => `
      <div class="other ${o.active ? "active" : ""}"><strong>${esc(o.label)}${o.active ? " · aktif" : ""}</strong>
        <span>${esc(o.pred_tr)} · ${pct(o.confidence)}</span></div>`).join("");
    const full = d.agreement === d.n_models;
    rBody.innerHTML = `
      <div class="verdict ${cls}"><div class="ic">${d.healthy ? "✓" : "!"}</div><div>
        <small>${esc(d.model_label)} tahmini</small><h3>${esc(d.pred_tr)}</h3>
        <p>Güven: <b>${pct(d.confidence)}</b>${d.confidence < 0.6 ? " · düşük güven, farklı bir açıdan tekrar çekmeyi dene" : ""}</p></div></div>
      <div class="conf">${bars}</div>
      <div class="info"><h4>${d.healthy ? "Durum" : "Hastalık hakkında"}</h4><p>${esc(d.info)}</p></div>
      <p class="agree ${full ? "full" : "part"}">${d.agreement}/${d.n_models} model aynı sonuçta${full ? " ✓" : ", sonucu dikkatli değerlendir"}</p>
      <div class="others">${others}</div>
      ${gateMeter(d)}`;
    rBody.hidden = false;
  }
})();
