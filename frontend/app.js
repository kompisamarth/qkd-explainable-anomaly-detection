/* QKD SOC dashboard - state, routing and view rendering. */
(function () {
  "use strict";

  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  const ICONS = {
    check: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    warn: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    alert: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    info: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
    lock: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    unlock: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 9.9-1"/></svg>',
    play: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>',
    pause: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>',
  };

  const S = {
    meta: null, session: null, scn: null, src: "trial",
    mode: "blind", revealed: false,
    idx: 0, playing: true, timer: null, stage: null,
    mdi: null, bench: null, chan: null, data: null, atlas: 0, drawnOnce: {},
  };

  // Every chart carries its source: label (simulated / real / analytic), files and seeds.
  function prov(id, p, extra) {
    const node = document.getElementById(id);
    if (!node || !p) return;
    const real = /real/i.test(p.label || ""), analytic = /analytic/i.test(p.label || "");
    // Live sessions have no result file: name the engine and model that produced them instead
    const files = (p.files || [p.file]).filter(Boolean).concat(p.files || p.file ? [] : [p.engine, p.model].filter(Boolean));
    const missing = (p.missing || []).length ? ` <b style="color:var(--crimson)">missing: ${esc(p.missing.join(", "))}</b>` : "";
    node.innerHTML = `<span class="prov-badge ${real ? "real" : analytic ? "analytic" : "sim"}">${esc(p.label || "Simulated data")}</span>
      <span>source: ${files.map(f => `<code>${esc(f)}</code>`).join(", ") || "&mdash;"}</span>
      <span>${esc(p.seeds != null ? p.seeds : (p.seed != null ? "seed " + p.seed : ""))}</span>${extra ? `<span>${extra}</span>` : ""}${missing}`;
  }
  const ci = r => r && r.ci95_low != null ? ` (95% CI ${pct(r.ci95_low, 1)}&ndash;${pct(r.ci95_high, 1)})` : "";

  function fatal(msg) {
    let bar = document.getElementById("fatalBar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "fatalBar";
      bar.style.cssText =
        "position:sticky;top:0;z-index:999;margin:0 0 14px;padding:13px 18px;" +
        "border-radius:10px;background:#fef2f2;border:1px solid #fecaca;color:#9f1239;" +
        "font-size:.88rem;line-height:1.5";
      const w = document.querySelector(".wrap") || document.body;
      w.insertBefore(bar, w.firstChild);
    }
    bar.innerHTML = "<b>Dashboard cannot run.</b> " + msg;
  }

  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    return r.json();
  }

  // ---------------------------------------------------------------- tabs
  function showView(name) {
    $$(".tab").forEach(t => t.classList.toggle("active", t.dataset.v === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.id === "v-" + name));
    if (name === "mdi" && !S.mdi) loadMdi();
    if (name === "bench" && !S.bench) loadBench();
    if (name === "channel" && !S.chan) loadChannel();
    if (name === "data" && !S.data) loadData();
    if (name === "explain") renderExplainCtl();
  }
  $$(".tab").forEach(t => t.addEventListener("click", () => showView(t.dataset.v)));

  // ---------------------------------------------------------------- boot
  async function boot() {
    try {
      const h = await api("/healthz");
    if (!h.model) fatal(h.detail || "The model checkpoint is missing.");
      $("#healthDot").classList.toggle("off", !h.model);
      $("#healthTxt").textContent = h.model ? "MODEL ONLINE" : "MODEL MISSING";
    } catch {
      $("#healthDot").classList.add("off");
      $("#healthTxt").textContent = "BACKEND OFFLINE";
      return;
    }
    S.meta = await api("/api/meta");

    renderOverview();
    renderScenarios();
    setupCustom();
    renderAtlas();
    S.stage = Photons.create($("#photons"));
    setupControls();
    
    // Start with Blind Mode by default
    selectScenario(0);
    applyMode();
    runBlind();
  }

  // ---------------------------------------------------------------- controls & modes
  // Applies blind/guided visibility. This must also run at boot: the app starts
  // in blind mode, and while this logic lived only inside the mode-toggle click
  // handler the first load left the guided panels on screen - including the
  // "Ground truth regime" readout, which gave the answer away before Reveal.
  function applyMode() {
    const isBlind = (S.mode === "blind");
    $("#blindSide").style.display = isBlind ? "" : "none";
    $("#guidedSide").style.display = S.mode === "guided" ? "" : "none";
    $("#customSide").style.display = S.mode === "custom" ? "" : "none";
    $("#lookBanner").style.display = isBlind ? "none" : "";
    $("#scnPhysicsPanel").style.display = isBlind ? "none" : "";
    $("#revealPanel").style.display = isBlind ? "" : "none";
    if (isBlind) {
      $("#scnTitle").textContent = "Autonomous Quantum Channel Monitor";
      $("#scnHead").textContent =
        "Real-time pulse tracking and 63-dimensional causal feature extraction.";
    }
  }

  function setupControls() {
    $$("#modeSeg button").forEach(b => {
      b.addEventListener("click", () => {
        $$("#modeSeg button").forEach(x => x.classList.toggle("on", x === b));
        S.mode = b.dataset.mode;
        applyMode();
        if (S.mode === "blind") {
          runBlind();
        } else if (S.mode === "custom") {
          runCustom();
        } else {
          selectScenario(0);
          run();
        }
      });
    });

    $("#btnRandSeed").addEventListener("click", () => {
      $("#inBlindSeed").value = Math.floor(Math.random() * 1000000);
    });

    $("#btnRunBlind").addEventListener("click", runBlind);

    $("#btnReveal").addEventListener("click", () => {
      S.revealed = true;
      $("#revealBody").style.display = "";
      $("#btnReveal").innerHTML = `${ICONS.unlock} Ground Truth Verified`;
      drawMonitor(false);
    });

    $("#inBlindBlocks").addEventListener("input", e => {
      $("#lblBlindBlocks").textContent = e.target.value + " blk";
      syncRange();
    });

    $$("#srcSeg button").forEach(b => b.addEventListener("click", () => {
      $$("#srcSeg button").forEach(x => x.classList.toggle("on", x === b));
      S.src = b.dataset.src;
      $("#liveOpts").style.display = S.src === "live" ? "" : "none";
    }));

    $("#btnRun").addEventListener("click", run);
  }

  // ---------------------------------------------------------------- monitor
  function renderScenarios() {
    $("#scnList").innerHTML = S.meta.scenarios.map((s, i) => `
      <button class="scn${i === 0 ? " on" : ""}" data-i="${i}">
        <span class="n">0${i + 1}</span>
        <div class="t">${esc(s.title)}</div>
        <div class="d">${esc(s.tag)}</div>
      </button>`).join("");
    $$("#scnList .scn").forEach(b =>
      b.addEventListener("click", () => { selectScenario(+b.dataset.i); run(); }));
  }

  function selectScenario(i) {
    S.scn = S.meta.scenarios[i];
    $$("#scnList .scn").forEach((b, j) => b.classList.toggle("on", j === i));
    $("#scnTitle").textContent = S.scn.title;
    $("#scnHead").textContent = S.scn.headline;
    $("#scnLook").textContent = S.scn.look;
    $("#inChannel").value = S.scn.channel;
    $("#inBlocks").value = S.scn.blocks;
    $("#inOnset").value = S.scn.onset;
    syncRange();
  }

  function syncRange() {
    $("#lblBlocks").textContent = $("#inBlocks").value + " blk";
    $("#lblOnset").textContent = $("#inOnset").value;
    $$("input[type=range]").forEach(r => {
      const p = ((r.value - r.min) / (r.max - r.min)) * 100;
      r.style.setProperty("--pct", p + "%");
    });
  }
  $$("input[type=range]").forEach(r => r.addEventListener("input", syncRange));
  $("#inBlocks").addEventListener("input", () => {
    $("#inOnset").max = Math.max(55, +$("#inBlocks").value - 20); syncRange();
  });

  function setupCustom() {
    const c = S.meta.controls || {};
    $("#cAttack").innerHTML = S.meta.class_names.map((n, i) => `<option value="${i}"${i === 1 ? " selected" : ""}>${esc(n)}</option>`).join("");
    const lo = c.fiber_distance_km || [5, 80], hlo = c.heldout_distance_km || [80, 110];
    const syncC = () => {
      const fib = $("#cChannel").value === "fiber";
      $("#cDistWrap").style.display = fib ? "" : "none";
      const d = +$("#cDist").value;
      $("#lblCDist").textContent = d + " km";
      $("#cDistNote").innerHTML = (d < lo[0] || d > lo[1])
        ? `<span style="color:var(--amber-text)">Outside the training range ${lo[0]}&ndash;${lo[1]} km${d >= hlo[0] && d <= hlo[1] ? " (held-out range)" : ""}: expect more false alarms.</span>`
        : `Training range ${lo[0]}&ndash;${lo[1]} km.`;
      const cls = +$("#cAttack").value;
      const r = (c.attack_intensity || {})[cls];
      $("#cIntWrap").style.display = cls === 0 ? "none" : "";
      if (r) {
        const v = +$("#cInt").value;
        $("#lblCInt").textContent = v.toFixed(2);
        $("#cIntNote").innerHTML = (v < r[0] || v > r[1])
          ? `<span style="color:var(--amber-text)">Outside the training range ${r[0]}&ndash;${r[1]}.</span>` : `Training range ${r[0]}&ndash;${r[1]}.`;
      }
      $("#lblCOnset").textContent = $("#cOnset").value;
    };
    $("#cAttack").addEventListener("change", () => {
      const r = (c.attack_intensity || {})[+$("#cAttack").value];
      if (r) $("#cInt").value = ((r[0] + r[1]) / 2).toFixed(2);
      syncC();
    });
    ["cChannel", "cDist", "cInt", "cOnset"].forEach(id => $("#" + id).addEventListener("input", syncC));
    $("#btnRunCustom").addEventListener("click", runCustom);
    syncC();
  }

  async function runCustom() {
    const btn = $("#btnRunCustom");
    btn.disabled = true; btn.innerHTML = `${ICONS.play} Simulating...`;
    stopPlay();
    S.revealed = true;
    try {
      const cls = +$("#cAttack").value;
      S.session = await api("/api/simulate/custom", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          protocol: $("#cProtocol").value, channel: $("#cChannel").value, distance_km: +$("#cDist").value,
          cls, intensity: cls === 0 ? null : +$("#cInt").value, detector: $("#cDetector").value,
          onset: +$("#cOnset").value, seed: +$("#cSeed").value,
        }),
      });
      $("#scnTitle").textContent = "Custom simulation: " + S.meta.class_full[cls];
      const ph = S.session.physics || {};
      $("#scnHead").textContent = `${S.session.provenance.protocol === "decoy_bb84" ? "Decoy-state BB84" : "BB84"}, ` +
        `${ph.channel_type === "free_space" ? "free space" : (ph.distance_km || 0).toFixed(0) + " km fibre"}, ` +
        `${(ph.loss_db || 0).toFixed(1)} dB loss, ${ph.detector_type}, e_d ${pct(ph.e_d0 || 0, 2)}.`;
      S.drawnOnce = {}; S.idx = 0;
      renderMonitorStatic();
      startPlay();
    } catch (e) {
      $("#verdictA").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false; btn.innerHTML = `${ICONS.play} Simulate`;
    }
  }

  async function runBlind() {
    const btn = $("#btnRunBlind");
    btn.disabled = true;
    btn.innerHTML = `${ICONS.play} Analyzing Link...`;
    stopPlay();
    S.revealed = false;
    $("#revealBody").style.display = "none";
    $("#btnReveal").innerHTML = `${ICONS.lock} Reveal Ground Truth`;
    try {
      S.session = await api("/api/simulate/blind", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          channel: $("#inBlindChannel").value,
          blocks: +$("#inBlindBlocks").value,
          seed: +$("#inBlindSeed").value,
        }),
      });
      S.drawnOnce = {};
      S.idx = 0;
      renderMonitorStatic();
      startPlay();
    } catch (e) {
      fatal(esc(e.message));
      $("#verdictA").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${ICONS.play} Analyze Blind Link`;
    }
  }

  async function run() {
    const btn = $("#btnRun");
    btn.disabled = true;
    btn.innerHTML = `${ICONS.play} Simulating...`;
    stopPlay();
    S.revealed = true;
    try {
      S.session = S.src === "trial"
        ? await api(`/api/trial/${S.scn.cls}`)
        : await api("/api/simulate", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              cls: S.scn.cls, channel: $("#inChannel").value,
              blocks: +$("#inBlocks").value, onset: +$("#inOnset").value,
              seed: +$("#inSeed").value,
            }),
          });
      S.drawnOnce = {};
      S.idx = 0;
      renderMonitorStatic();
      startPlay();
    } catch (e) {
      fatal(esc(e.message));
      $("#verdictA").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false;
      btn.innerHTML = `${ICONS.play} Run Scenario`;
    }
  }

  function renderMonitorStatic() {
    const s = S.session, m = s.summary;
    const scrub = $("#scrub");
    scrub.max = s.blocks.length - 1; scrub.value = 0;

    const kpi = (k, v, sub, kind) =>
      `<div class="stat ${kind || ""}"><div class="bar"></div><div class="k">${k}</div>
       <div class="v">${v}</div><div class="s">${sub}</div></div>`;

    $("#kpis").innerHTML =
      kpi("Mean QBER After Onset", pct(m.mean_qber_post, 2), "Theoretical threshold: 11.00%",
          m.mean_qber_post > 0.11 ? "bad" : "") +
      kpi("Transmission Efficiency", pct(m.final_efficiency, 1), "Yield monitoring (PNS deficit)", "info") +
      kpi("Basis Asymmetry |Qz - Qx|", pct(m.final_asymmetry, 2), "Dual-basis differential", "warn") +
      (m.delta_t == null
        ? kpi("Detection Latency", "Nominal", "Zero attack flags raised", s.class_id === 0 ? "good" : "bad")
        : kpi("Detection Latency ΔT", m.delta_t + " blk", "First alarm at block " + m.first_alarm, "bad"));

    $("#verdictA").innerHTML = m.threshold_fires
      ? `<div class="banner bad"><span class="em">${ICONS.alert}</span><div><b>Classical 11% Static Threshold: ABORT TRIGGERED.</b>
         QBER exceeded 11.0% limit. Sufficient signal for textbook scalar rule.</div></div>`
      : `<div class="banner warn"><span class="em">${ICONS.warn}</span><div><b>Classical 11% Static Threshold: SILENT (BYPASSED).</b>
         QBER remained below 11.0%. A conventional BB84 system would continue sharing compromised key material.</div></div>`;

    let vb;
    if (s.class_id === 0) {
      const fp = m.false_positive_rate;
      vb = `<div class="banner ${fp < .1 ? "ok" : "bad"}"><span class="em">${fp < .1 ? ICONS.check : ICONS.warn}</span>
        <div><b>Temporal AI Model: LINK SECURE.</b> Nominal channel confirmed. Flagged only ${pct(fp, 1)} of blocks under baseline turbulence.</div></div>`;
    } else if (m.delta_t != null) {
      vb = `<div class="banner ok"><span class="em">${ICONS.check}</span><div><b>Temporal AI Model: BREACH DETECTED.</b>
        Flagged within ${m.delta_t} blocks. Maintained consistent attribution across ${pct(m.post_accuracy, 0)} of post-onset blocks.</div></div>`;
    } else {
      vb = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div><b>Temporal AI Model: DETECTION DELAYED.</b>
        Adversary power near noise floor; initial sliding window did not accumulate sufficient anomaly energy.</div></div>`;
    }
    if (s.p_eve > 0 && s.p_eve < 0.12 && s.class_id !== 0) {
      vb += `<div class="banner info" style="margin-top:9px"><span class="em">${ICONS.info}</span><div>
        Weak interception intensity drawn for this session (p<sub>eve</sub> = ${s.p_eve}). Physical signals near the detector dark count boundary are rigorously reported.</div></div>`;
    }
    $("#verdictB").innerHTML = vb;

    const ph = S.meta.class_physics[s.class_id];
    
    // Scenario physics panel for Guided mode
    $("#scnPhysics").innerHTML = `
      <div class="kv"><span>Drawn attack strength (p_eve)</span><span>${s.p_eve}</span></div>
      <div class="kv"><span>Ground truth regime</span><span>${esc(s.class_full)}</span></div>
      <p style="margin:13px 0 0"><b>Physical mechanism:</b> <span class="muted">${esc(ph.what)}</span></p>
      <p style="margin:9px 0 0"><b>Observable signature:</b> <span class="muted">${esc(ph.watch)}</span></p>
      <p style="margin:9px 0 0"><b>Detection challenge:</b> <span class="muted">${esc(ph.hard)}</span></p>
      <p class="muted" style="margin:13px 0 0;font-size:0.78rem">Each session draws baseline QBER, channel transmittance, dark-count rate and detector basis mismatch from overlapping empirical distributions.</p>`;

    // Blind mode Ground Truth Scorecard
    const majPred = s.pred[s.pred.length - 1];
    const isCorrect = (s.class_id === majPred);
    $("#revealBody").innerHTML = `
      <div class="grid g3" style="gap:12px;margin-bottom:14px;">
        <div class="stat"><div class="bar"></div>
          <div class="k">Secret Ground Truth</div>
          <div class="v" style="font-size:1.25rem;color:var(--accent);">${esc(s.class_full)}</div>
          <div class="s">Onset: block ${s.onset} &nbsp;|&nbsp; p<sub>eve</sub> = ${s.p_eve}</div>
        </div>
        <div class="stat ${isCorrect ? "good" : "bad"}"><div class="bar"></div>
          <div class="k">AI Classifier Verdict</div>
          <div class="v" style="font-size:1.25rem;">${isCorrect ? "TRUE POSITIVE" : "MISCLASSIFIED"}</div>
          <div class="s">${isCorrect ? `Correctly attributed in ${m.delta_t ?? 0} blocks` : `Classified as ${S.meta.class_names[majPred]}`}</div>
        </div>
        <div class="stat ${m.threshold_fires ? "warn" : "bad"}"><div class="bar"></div>
          <div class="k">Classical 11% Rule</div>
          <div class="v" style="font-size:1.25rem;">${m.threshold_fires ? "ABORT TRIGGERED" : "SILENT (BYPASSED)"}</div>
          <div class="s">${m.threshold_fires ? "Exceeded scalar limit" : "Failed to catch stealth breach"}</div>
        </div>
      </div>
      <div class="kv"><span>Physical mechanism</span><span>${esc(ph.what)}</span></div>
      <div class="kv"><span>Observable physical signature</span><span>${esc(ph.watch)}</span></div>
      <div class="kv"><span>Detection challenge</span><span>${esc(ph.hard)}</span></div>
    `;

    prov("provMon", s.provenance, s.provenance.out_of_training_range ? "outside the training distance range" : "");
    drawMonitor(true);
  }

  function drawMonitor(first) {
    const s = S.session, n = S.idx + 1;
    const xs = s.blocks.slice(0, n);
    const cut = a => a.slice(0, n);
    const onsetI = s.blocks.indexOf(s.onset);
    const alarmI = s.summary.first_alarm != null ? s.blocks.indexOf(s.summary.first_alarm) : null;
    const anim = !!first;

    // Show onset marker only if revealed or in guided mode
    const showOnset = (S.revealed || S.mode === "guided") && onsetI <= S.idx;

    const series = [{ name: "window mean (W=50)", color: "#0d9488", data: cut(s.qber), width: 2.6 }];
    if (s.qber_block) series.unshift({ name: "block QBER", color: "#64748b", data: cut(s.qber_block), width: 1 });
    Charts.line($("#chQber"), {
      xs, animate: anim, pulseLast: true, yMax: 0.14,
      fmtY: v => (v * 100).toFixed(0) + "%",
      band: s.ci_low ? { lo: cut(s.ci_low), hi: cut(s.ci_high), color: "#64748b" } : null,
      series,
      threshold: { y: 0.11, color: "#dc2626", label: "11.0% abort" },
      markers: [
        { i: showOnset ? onsetI : null, color: "#d97706", label: "onset" },
        { i: alarmI != null && alarmI <= S.idx ? alarmI : null, color: "#059669", label: "alarm" },
      ],
    });

    Charts.line($("#chBasis"), {
      xs, animate: anim, pulseLast: true,
      fmtY: v => (v * 100).toFixed(0) + "%",
      series: [
        { name: "Q_Z rectilinear", color: "#0284c7", data: cut(s.qber_z) },
        { name: "Q_X diagonal", color: "#d97706", data: cut(s.qber_x) },
      ],
      markers: [{ i: showOnset ? onsetI : null, color: "#d97706", label: "onset" }],
    });

    Charts.line($("#chEff"), {
      xs, animate: anim, pulseLast: true,
      fmtY: v => (v * 100).toFixed(v < 0.01 ? 3 : 1) + "%",
      series: [{ name: "transmission efficiency", color: "#0d9488", data: cut(s.efficiency), fill: true }],
      markers: [{ i: showOnset ? onsetI : null, color: "#d97706", label: "onset" }],
    });

    const p = s.probs[S.idx] || [];
    if (!$("#probBars").children.length) {
      $("#probBars").innerHTML = S.meta.class_names.map((nm, k) => `
        <div class="pb"><div class="row"><span class="nm">${esc(nm)}</span>
        <span class="vl" data-v="${k}">0%</span></div>
        <div class="track"><div class="fill" data-f="${k}"
          style="background:${S.meta.class_colors[k]}"></div></div></div>`).join("");
    }
    p.forEach((v, k) => {
      $(`#probBars .fill[data-f="${k}"]`).style.width = (v * 100).toFixed(1) + "%";
      $(`#probBars .vl[data-v="${k}"]`).textContent = pct(v, 1);
    });

    // Threat HUD real-time update
    const pred = s.pred[S.idx];
    const conf = p[pred] || 0.5;
    const th = $("#threatHud");
    if (pred === 0 || conf < 0.50) {
      th.className = "threat-hud secure";
      $("#threatStatus").textContent = "LINK SECURE — DISTILLATION ACTIVE";
      $("#threatSub").textContent = `Confidence ${pct(conf, 1)} · Nominal channel dynamics`;
    } else if (conf < 0.70) {
      th.className = "threat-hud warn";
      $("#threatStatus").textContent = `LOW-CONFIDENCE ANOMALY DETECTED`;
      $("#threatSub").textContent = `Confidence ${pct(conf, 1)} · Elevated feature variance`;
    } else {
      th.className = "threat-hud breach";
      $("#threatStatus").textContent = `BREACH DETECTED — ${S.meta.class_names[pred].toUpperCase()} IDENTIFIED`;
      $("#threatSub").textContent = `Confidence ${pct(conf, 1)} · Autonomous key abort triggered`;
    }

    $("#scrubLbl").textContent = `block ${s.blocks[S.idx]} / ${s.blocks[s.blocks.length - 1]}`;
    $("#scrub").value = S.idx;
    syncRange();

    // The canvas draws Eve straight from ground truth, so in blind mode it has
    // to stay sealed until Reveal - otherwise the status line announces the
    // attack and its exact interception rate before the analyst has judged.
    const reveal = S.revealed || S.mode === "guided";
    const active = s.blocks[S.idx] >= s.onset && s.class_id !== 0 && reveal;
    S.stage.set({
      attackActive: active, attackClass: reveal ? s.class_id : 0,
      intercept: active ? Math.max(s.p_eve, .15) : 0,
      qber: s.qber[S.idx], efficiency: s.efficiency[S.idx],
      blind: !reveal,
    });
  }

  function startPlay() {
    S.playing = true;
    $("#btnPlay").innerHTML = `${ICONS.pause} Pause`;
    S.stage.resume();
    clearInterval(S.timer);
    S.timer = setInterval(() => {
      if (!S.playing) return;
      S.idx++;
      if (S.idx >= S.session.blocks.length) { S.idx = S.session.blocks.length - 1; stopPlay(); return; }
      drawMonitor(false);
    }, 110);
  }
  function stopPlay() {
    S.playing = false; clearInterval(S.timer);
    const b = $("#btnPlay");
    if (b) b.innerHTML = `${ICONS.play} Resume`;
    if (S.stage) S.stage.pause();
  }
  $("#btnPlay").addEventListener("click", () => {
    if (S.playing) stopPlay();
    else { if (S.idx >= S.session.blocks.length - 1) S.idx = 0; startPlay(); }
  });
  $("#scrub").addEventListener("input", e => {
    stopPlay(); S.idx = +e.target.value; drawMonitor(false);
  });

  // ---------------------------------------------------------------- explain
  function renderExplainCtl() {
    if (!S.session) {
      $("#explainCtl").innerHTML = `<div class="banner info"><span class="em">${ICONS.info}</span><div>Run a transmission session in <b>Live Monitor</b> first.</div></div>`;
      return;
    }
    const s = S.session;
    if (!$("#exBlock")) {
      $("#explainCtl").innerHTML = `<div class="control" style="max-width:620px;margin:0">
        <label><span>Transmission Block to Audit</span><b id="exLbl">${s.blocks[S.idx]}</b></label>
        <input type="range" id="exBlock" min="${s.blocks[0]}" max="${s.blocks[s.blocks.length - 1]}"
               value="${s.blocks[S.idx]}"></div>`;
      $("#exBlock").addEventListener("input", e => {
        $("#exLbl").textContent = e.target.value; syncRange();
        clearTimeout(S._exT);
        S._exT = setTimeout(() => loadExplain(+e.target.value), 180);
      });
    } else {
      $("#exBlock").min = s.blocks[0];
      $("#exBlock").max = s.blocks[s.blocks.length - 1];
      $("#exBlock").value = s.blocks[S.idx];
      $("#exLbl").textContent = s.blocks[S.idx];
    }
    syncRange();
    loadExplain(s.blocks[S.idx]);
  }

  async function loadExplain(block) {
    // Dragging the slider fires several requests; only the latest may render
    const req = S.explainReq = (S.explainReq || 0) + 1;
    $("#chShap").innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
    let d;
    try { d = await api(`/api/explain?session_id=${S.session.session_id}&block=${block}`); }
    catch (e) {
      if (req !== S.explainReq) return;
      $("#chShap").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>${esc(e.message)}</div></div>`;
      return;
    }
    if (req !== S.explainReq) return;
    try { renderExplain(d); }
    catch (e) {
      console.error(e);
      $("#chShap").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>Could not render this block's attribution: ${esc(e.message)}</div></div>`;
    }
  }

  function renderExplain(d) {
    const truth = S.session.class_id;
    const ok = d.pred === truth;
    // In blind mode the ground truth stays sealed here too until Reveal
    const truthVisible = S.mode !== "blind" || S.revealed;

    $("#explainMeta").innerHTML = `
      <div class="stat ${d.pred === 0 ? "good" : "bad"}"><div class="bar"></div>
        <div class="k">Model Verdict</div><div class="v" style="font-size:1.2rem">${esc(d.pred_name)}</div>
        <div class="s">Confidence ${pct(d.probs[d.pred], 1)}</div></div>` + (truthVisible ? `
      <div class="stat info"><div class="bar"></div><div class="k">Ground Truth</div>
        <div class="v" style="font-size:1.2rem">${esc(S.meta.class_names[truth])}</div>
        <div class="s">Simulated physical regime</div></div>
      <div class="stat ${ok ? "good" : "warn"}"><div class="bar"></div><div class="k">Audit Outcome</div>
        <div class="v" style="font-size:1.2rem">${ok ? "Attribution Validated" : "Misclassified"}</div>
        <div class="s">Block ${d.block}</div></div>` : `
      <div class="stat"><div class="bar"></div><div class="k">Ground Truth</div>
        <div class="v" style="font-size:1.2rem">${ICONS.lock} Sealed</div>
        <div class="s">Reveal it in Live Monitor</div></div>`);

    Charts.hbars($("#chShap"), {
      diverging: true, labelW: 176,
      items: d.top_features.map(f => ({ label: f.feature, value: f.shap })),
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(3),
    });

    // A server started before these fields existed still serves this file; say so
    // instead of failing, and keep the rest of the tab working.
    const stale = `<div class="banner warn"><span class="em">${ICONS.warn}</span><div>The running server predates this panel. Restart it (<code>python run_dashboard.py</code>) to load it.</div></div>`;
    if (Array.isArray(d.thresholds)) renderThresholds(d.thresholds, truthVisible && truth !== 0);
    else $("#shapThresholds").innerHTML = stale;
    prov("provShap", d.provenance || { label: "Simulated data" }, "block " + d.block);
    S.explain = d;
    if (Array.isArray(d.features)) renderAllFeatures();
    else $("#shapAll").innerHTML = stale;

    const gs = Object.entries(d.groups).sort((a, b) => b[1] - a[1]);
    $("#shapGroups").innerHTML = `<div class="grid g3" style="gap:10px">` + gs.map(([g, v]) => `
      <div class="stat"><div class="bar"></div><div class="k">${esc(g.replace(/_/g, " "))}</div>
      <div class="v" style="font-size:1.35rem">${pct(v, 0)}</div></div>`).join("") + `</div>`;

    const t = d.top_features.slice(0, 2).map(f => `<code>${esc(f.feature)}</code>`);
    const reasons = [
      `${t[0]} and ${t[1]} are consistent with a benign channel: QBER and click rate within the drift and loss variation every session shows.`,
      `${t[0]} and ${t[1]} show the intercept-resend signature: a step rise in short-window QBER (about a quarter of the intercepted fraction) while the window mean stays below 11%.`,
      `${t[0]} and ${t[1]} drive a PNS verdict. A gain-matched PNS attack leaves QBER and click rate unchanged, so without decoy states this attribution is weak.`,
      `${t[0]} and ${t[1]} point to detector blinding: QBER fell below the optical misalignment floor, which honest single-photon detection cannot do.`,
      `${t[0]} and ${t[1]} drive a time-shift verdict: the click rate dropped while QBER stayed near its baseline. Benign loss steps can look the same, so treat this as a lead, not proof.`,
      `${t[0]} and ${t[1]} drive a Trojan-horse verdict. Bob's telemetry carries almost no Trojan signature in this model, so this attribution is weak; Alice's watchdog photodiode is the real test.`,
      `${t[0]} and ${t[1]} drive a beam-splitting verdict. A loss-compensated tap leaves Bob's statistics unchanged, so this attribution is weak by construction.`,
    ];
    $("#shapWords").innerHTML = `<p style="margin:0;line-height:1.55;">${reasons[d.pred]}</p>` + (ok || !truthVisible ? "" :
      `<div class="banner warn" style="margin-top:12px"><span class="em">${ICONS.warn}</span><div>This specific transmission block exhibited transient misclassification. Borderline attribution occurs primarily immediately following onset before the sliding window fully fills with attacked pulses.</div></div>`);
  }

  // Raw feature values span ~1e-6 .. 1e4, so pick precision by magnitude
  function fnum(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (a >= 1) return v.toFixed(3);
    if (a >= 1e-3) return v.toFixed(4);
    return v.toExponential(2);
  }

  function renderThresholds(list, isAttack) {
    const fmt = (v, unit) => unit === "prob" ? pct(v, 1) : pct(v, 2);
    const fired = list.filter(t => t.exceeded).length;
    const rows = list.map(t => {
      // Gauge spans 0 .. 1.6x the threshold (or the value, if larger)
      const span = Math.max(t.threshold * 1.6, t.value * 1.05, 1e-9);
      const vw = Math.min(100, t.value / span * 100), tw = t.threshold / span * 100;
      const status = t.exceeded
        ? `<span class="th-chip alarm">${ICONS.alert} Alarm: exceeds threshold</span>`
        : isAttack
          ? `<span class="th-chip bypass">${ICONS.warn} Within threshold: attack bypasses</span>`
          : `<span class="th-chip ok">${ICONS.check} Within threshold</span>`;
      return `<div class="th-row ${t.exceeded ? "is-alarm" : ""}">
        <div class="th-head"><b>${esc(t.name)}</b>${status}</div>
        <div class="th-gauge" title="${esc(t.metric)}">
          <div class="th-fill" style="width:${vw.toFixed(1)}%"></div>
          <div class="th-line" style="left:${tw.toFixed(1)}%"></div>
        </div>
        <div class="th-meta"><span>${esc(t.metric)}: <b>${fmt(t.value, t.unit)}</b> vs limit <b>${fmt(t.threshold, t.unit)}</b>
          (${t.margin >= 0 ? "+" : ""}${t.unit === "prob" ? (t.margin * 100).toFixed(1) + " pts" : (t.margin * 100).toFixed(2) + " pts"})</span>
          <span class="muted">${esc(t.note)}</span></div>
      </div>`;
    }).join("");
    const lead = fired === 0
      ? `<div class="banner ${isAttack ? "warn" : "info"}"><span class="em">${isAttack ? ICONS.warn : ICONS.info}</span><div><b>No detector alarms on this block.</b> Every metric is within its threshold${isAttack ? ", so this attack goes unflagged here." : "."}</div></div>`
      : `<div class="banner ${fired === list.length ? "bad" : "warn"}"><span class="em">${ICONS.alert}</span><div><b>${fired} of ${list.length} detectors alarm on this block.</b> ${list.length - fired} stay within their threshold.</div></div>`;
    $("#shapThresholds").innerHTML = lead + `<div class="th-list">${rows}</div>`;
  }

  function renderAllFeatures() {
    const d = S.explain;
    if (!d) return;
    const gsel = $("#shapAllGroup");
    if (gsel.options.length === 1) {
      Object.keys(S.meta.feature_groups).forEach(g =>
        gsel.insertAdjacentHTML("beforeend", `<option value="${g}">${esc(g.replace(/_/g, " "))}</option>`));
      gsel.addEventListener("change", renderAllFeatures);
      $("#shapAllSort").addEventListener("change", renderAllFeatures);
    }
    const g = gsel.value, sort = $("#shapAllSort").value;
    let fs = d.features.filter(f => !g || f.group === g);
    if (sort === "pos") fs = fs.slice().sort((a, b) => b.shap - a.shap);
    else if (sort === "neg") fs = fs.slice().sort((a, b) => a.shap - b.shap);
    else if (sort === "group") fs = fs.slice().sort((a, b) => a.group.localeCompare(b.group) || a.rank - b.rank);
    const mx = Math.max(...d.features.map(f => Math.abs(f.shap)), 1e-9);
    const rows = fs.map(f => {
      const w = Math.abs(f.shap) / mx * 50;
      const pos = f.shap >= 0;
      return `<tr>
        <td class="muted">#${f.rank}</td>
        <td><code>${esc(f.feature)}</code><div class="fx-ctx">${esc(f.context)}</div></td>
        <td class="muted fx-grp">${esc(f.group.replace(/_/g, " "))}</td>
        <td class="fx-num">${fnum(f.value)}</td>
        <td class="fx-bar-cell"><div class="fx-bar"><span class="${pos ? "pos" : "neg"}"
          style="${pos ? "left:50%" : "right:50%"};width:${w.toFixed(1)}%"></span></div></td>
        <td class="fx-num ${pos ? "bad" : "good"}">${pos ? "+" : ""}${f.shap.toFixed(4)}</td>
      </tr>`;
    }).join("");
    $("#shapAll").innerHTML = `<div class="fx-scroll"><table class="fx-table">
      <thead><tr><th>Rank</th><th>Feature &amp; meaning</th><th>Group</th><th>Value</th>
        <th>Away from &middot; toward verdict</th><th>SHAP</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
      <p class="cap" style="margin:10px 0 0">${fs.length} feature${fs.length === 1 ? "" : "s"} shown.
        Positive SHAP pushes toward <b>${esc(d.pred_name)}</b>; negative pushes away from it.</p>`;
  }

  // ---------------------------------------------------------------- mdi
  async function loadMdi() {
    S.mdi = await api("/api/mdi");
    $("#inCtrl").addEventListener("input", () => { syncRange(); renderMdiCards(); });
    syncRange(); renderMdiCards();

    const a = S.mdi.attack, kr = S.mdi.key_rate;
    prov("provMdiMon", S.mdi.provenance && S.mdi.provenance.attack);
    prov("provMdiRate", S.mdi.provenance && S.mdi.provenance.key_rate);
    if (a) {
      Charts.line($("#chMdiMon"), {
        xs: a.map(r => (r.p_control * 100).toFixed(0) + "%"), yMax: 0.58,
        fmtY: v => (v * 100).toFixed(0) + "%",
        series: [
          { name: "BB84 (Forged by Eve)", color: "#dc2626", data: a.map(r => r.bb84_check_error) },
          { name: "MDI-QKD X-Basis Phase Error", color: "#0d9488", data: a.map(r => r.mdi_check_error) },
        ],
        threshold: { y: 0.11, color: "#d97706", label: "11% abort limit" },
      });
    }
    if (kr) {
      Charts.logLine($("#chMdiRate"), {
        xs: kr.map(r => r.distance_km),
        series: [
          { name: "PLOB Bound", color: "#94a3b8", data: kr.map(r => r.plob_bound), dash: true },
          { name: "SNS-TF-QKD", color: "#7c3aed", data: kr.map(r => r.sns_tf_qkd) },
          { name: "MDI-QKD", color: "#0d9488", data: kr.map(r => r.mdi_qkd) },
          { name: "Decoy BB84", color: "#d97706", data: kr.map(r => r.decoy_bb84) },
        ],
      });
    }
  }

  function renderMdiCards() {
    const v = +$("#inCtrl").value / 100;
    $("#lblCtrl").textContent = (v * 100).toFixed(0) + "%";
    const rows = S.mdi.attack;
    const r = rows.reduce((b, x) => Math.abs(x.p_control - v) < Math.abs(b.p_control - v) ? x : b, rows[0]);

    const tile = (k, val, sub, kind) =>
      `<div class="stat ${kind}"><div class="bar"></div><div class="k">${k}</div>
       <div class="v" style="font-size:1.45rem">${val}</div><div class="s">${sub}</div></div>`;

    $("#mdiBB84").innerHTML =
      `<div class="grid" style="gap:10px">
        ${tile("Mutual Information Leaked", pct(r.bb84_eve_information, 0), "Directly extracted key bits", "bad")}
        ${tile("Observed Error Rate", pct(r.bb84_qber, 2), "Remains below 11% threshold", "bad")}
      </div>
      <div class="banner bad" style="margin-top:12px"><span class="em">${ICONS.alert}</span><div><b>Breach Undetected.</b>
        Eve controls the single-photon detector outputs directly; all local telemetry monitors are forged to match benign noise profiles.</div></div>`;

    $("#mdiMDI").innerHTML =
      `<div class="grid" style="gap:10px">
        ${tile("Z-Basis Sifted Error", pct(r.mdi_qber, 2), "Kept low by adversary pulses", "")}
        ${tile("X-Basis Bell Phase Error", pct(r.mdi_check_error, 1), "Cannot be forged by untrusted relay",
               r.mdi_aborts ? "bad" : "good")}
      </div>` + (r.mdi_aborts
        ? `<div class="banner ok" style="margin-top:12px"><span class="em">${ICONS.check}</span><div><b>Protocol Aborted.</b>
           Attacking the pulses collapses two-photon Bell interference. Alice and Bob identify phase disturbance by evaluating conjugate correlations.</div></div>`
        : `<div class="banner info" style="margin-top:12px"><span class="em">${ICONS.info}</span><div>Below abort threshold &mdash; minimal mutual information gained by adversary at this power level.</div></div>`);

    const first = rows.find(x => x.mdi_aborts);
    if (first && !$("#mdiCross")) {
      const p = document.createElement("p");
      p.id = "mdiCross"; p.className = "muted"; p.style.marginTop = "14px";
      p.innerHTML = `<b>Sensitivity Crossover:</b> MDI-QKD trips its phase check once Eve controls just
        <b style="color:var(--accent)">${pct(first.p_control, 0)}</b> of the relay. Conventional BB84 fails to trip at any adversary control level.`;
      $("#v-mdi").querySelector(".panel").appendChild(p);
    }
  }

  // ---------------------------------------------------------------- bench
  const MODEL_LABEL = { xgb: "XGBoost", rf: "Random Forest", svm: "SVM-RBF" };
  function table(rows, cols, fmt) {
    if (!rows || !rows.length) return '<p class="muted">Not generated yet.</p>';
    cols = cols || Object.keys(rows[0]);
    return `<table><thead><tr>${cols.map(c => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(r => `<tr>${cols.map(c => {
        let v = r[c];
        if (fmt && fmt[c]) v = fmt[c](v, r);
        else if (typeof v === "number") v = Math.abs(v) < 1 && v !== 0 ? v.toFixed(4) : v.toFixed(3);
        return `<td>${v == null ? "&mdash;" : v}</td>`;
      }).join("")}</tr>`).join("")}</tbody></table>`;
  }

  async function loadBench() {
    S.bench = await api("/api/benchmarks");
    const b = S.bench, P = b.provenance || {};
    const sm = {};
    (b.summary || []).forEach(r => { sm[r[Object.keys(r)[0]]] = r; });
    const tile = (k, v, sub, kind) =>
      `<div class="stat ${kind}"><div class="bar"></div><div class="k">${k}</div>
       <div class="v">${v}</div><div class="s">${sub}</div></div>`;
    const m = (k, d = 2) => sm[k] ? pct(sm[k].mean, d) : "&mdash;";
    $("#benchKpis").innerHTML =
      tile("7-class accuracy (XGBoost)", m("accuracy"), sm.accuracy ? `95% CI ${pct(sm.accuracy.ci95_low, 1)}&ndash;${pct(sm.accuracy.ci95_high, 1)}, ${sm.accuracy.n_seeds} seeds` : "", "info") +
      tile("Macro-F1", sm.macro_f1 ? sm.macro_f1.mean.toFixed(3) : "&mdash;", sm.macro_f1 ? `&plusmn; ${sm.macro_f1.std.toFixed(3)} sd` : "", "info") +
      tile("Missed attacks (FNR)", m("xgb_binary_fnr"), "fixed 11% threshold: " + m("fixed11_binary_fnr", 1), "warn") +
      tile("False alarms (FPR)", m("xgb_binary_fpr"), "fixed 11% threshold: " + m("fixed11_binary_fpr", 1), "warn");
    prov("provBenchKpi", P.summary);

    const cmp = b.comparison || [];
    const bar = (metric, el) => {
      const rows = cmp.filter(r => r.metric === metric);
      if (!rows.length) { $(el).innerHTML = '<p class="muted">Run scripts/10_multiseed_evaluation.py</p>'; return; }
      Charts.hbars($(el), {
        labelW: 210, max: 1,
        items: rows.map(r => ({ label: MODEL_LABEL[r.model] || r.model, value: r.mean, lo: r.ci95_low, hi: r.ci95_high,
          color: MODEL_LABEL[r.model] ? "#0d9488" : "#94a3b8" })),
        fmt: v => pct(v, 1),
      });
    };
    bar("binary_fnr", "#chCmpFnr"); bar("binary_fpr", "#chCmpFpr");
    prov("provCmp", P.comparison, "whiskers: 95% CI over seeds");
    const metrics = ["accuracy", "macro_f1", "binary_fnr", "binary_fpr", "binary_auroc", "binary_auprc", "ece_top_label"];
    const models = [...new Set(cmp.map(r => r.model))];
    $("#tblCmp").innerHTML = table(models.map(mo => {
      const row = { model: MODEL_LABEL[mo] || mo };
      metrics.forEach(k => { const r = cmp.find(x => x.model === mo && x.metric === k); row[k] = r ? r : null; });
      return row;
    }), ["model", ...metrics], Object.fromEntries(metrics.map(k => [k, v => v ? `${v.mean.toFixed(3)} <span class="muted">&plusmn;${v.std.toFixed(3)}</span>` : "&mdash;"])));

    const drawPC = () => {
      const mo = $("#pcModel").value;
      const rows = (b.per_class_multiseed || []).filter(r => r.model === mo);
      if (!rows.length) { $("#chPerClass").innerHTML = '<p class="muted">Not generated yet.</p>'; return; }
      Charts.hbars($("#chPerClass"), {
        labelW: 130, max: 1,
        items: rows.map(r => ({ label: S.meta.class_names[r.class_id], value: r.f1_mean,
          lo: Math.max(0, r.f1_mean - (r.f1_std || 0)), hi: Math.min(1, r.f1_mean + (r.f1_std || 0)),
          color: S.meta.class_colors[r.class_id] })),
        fmt: v => v.toFixed(2),
      });
    };
    $("#pcModel").onchange = drawPC; drawPC();
    prov("provPerClass", P.per_class_multiseed, "whiskers: &plusmn; 1 sd over seeds");

    if (b.confusion && b.confusion.values) {
      Charts.heatmap($("#chCm"), { labels: S.meta.class_names, values: b.confusion.values });
    } else $("#chCm").innerHTML = '<p class="muted">Not generated yet.</p>';
    prov("provCm", P.confusion);

    const dpc = b.per_class_multiseed_decoy || [];
    if (dpc.length && b.per_class_multiseed) {
      const items = [];
      S.meta.class_names.forEach((nm, c) => {
        const a1 = (b.per_class_multiseed || []).find(r => r.model === "xgb" && r.class_id === c);
        const a2 = dpc.find(r => r.model === "xgb" && r.class_id === c);
        if (a1) items.push({ label: nm + " (no decoy)", value: a1.f1_mean, lo: a1.f1_mean - a1.f1_std, hi: a1.f1_mean + a1.f1_std, color: "#94a3b8" });
        if (a2) items.push({ label: nm + " (decoy)", value: a2.f1_mean, lo: a2.f1_mean - a2.f1_std, hi: a2.f1_mean + a2.f1_std, color: S.meta.class_colors[c] });
      });
      Charts.hbars($("#chDecoy"), { labelW: 190, max: 1, items, fmt: v => v.toFixed(2) });
    } else $("#chDecoy").innerHTML = '<p class="muted">Run scripts/10_multiseed_evaluation.py --decoy</p>';
    prov("provDecoy", P.per_class_multiseed_decoy, "grey: without decoy features");

    const lat = (b.latency || []).filter(r => r.model === "xgb");
    $("#tblLatency").innerHTML = table(lat, ["class_name", "intensity_bin", "intensity_mean", "sessions", "detection_rate", "latency_mean", "latency_median"], {
      detection_rate: v => pct(v, 0), intensity_mean: v => v.toFixed(3), latency_mean: v => v.toFixed(1), latency_median: v => v.toFixed(0),
    }) + (lat.length ? `<p class="cap" style="margin-top:8px">Pre-onset false-alarm rate (per block): ${pct(lat[0].pre_attack_false_alarm_mean, 1)} &plusmn; ${pct(lat[0].pre_attack_false_alarm_std, 1)}</p>` : "");
    prov("provLatency", P.latency);

    const rel = b.reliability || [];
    if (rel.length) {
      const bins = [...new Set(rel.map(r => r.bin_low))].sort((x, y) => x - y);
      const ser = Object.keys(MODEL_LABEL).filter(mo => rel.some(r => r.model === mo)).map((mo, i) => ({
        name: MODEL_LABEL[mo], color: ["#0d9488", "#7c3aed", "#d97706"][i],
        data: bins.map(bl => { const g = rel.filter(r => r.model === mo && r.bin_low === bl); return g.length ? g.reduce((a, r) => a + r.observed_frequency, 0) / g.length : null; }),
      }));
      ser.push({ name: "perfect calibration", color: "#cbd5e1", data: bins.map(bl => bl + 0.05) });
      Charts.line($("#chRel"), { xs: bins.map(bl => (bl * 100).toFixed(0) + "%"), yMax: 1, fmtY: v => (v * 100).toFixed(0) + "%", series: ser });
    }
    prov("provRel", P.reliability, "x: predicted P(attack) bin; y: observed attack frequency");

    const held = (b.heldout || []).filter(r => /^(xgb|rf|svm)_(accuracy|macro_f1|binary_fnr|binary_fpr)$/.test(r[Object.keys(r)[0]]));
    $("#tblHeld").innerHTML = table(held.map(r => {
      const key = r[Object.keys(r)[0]];
      const [mo, ...rest] = key.split("_");
      return { model: MODEL_LABEL[mo], metric: rest.join("_"), mean: r.mean, ci: r };
    }), ["model", "metric", "mean", "ci"], { mean: v => v.toFixed(3), ci: r => `${r.ci95_low.toFixed(3)}&ndash;${r.ci95_high.toFixed(3)}` });
    prov("provHeld", P.heldout, "unseen attack intensities and 80&ndash;110 km fibre");

    if (b.multiseed) {
      Charts.dots($("#chSeeds"), { values: b.multiseed.map(r => r.accuracy), labels: b.multiseed.map(r => r.seed) });
    }
    prov("provSeeds", P.multiseed);
    if (b.ablation) {
      const rows = [...b.ablation].sort((x, y) => y.accuracy_drop_pct - x.accuracy_drop_pct);
      Charts.hbars($("#chAbl"), {
        labelW: 158,
        items: rows.map(r => ({ label: r.display_name, value: r.accuracy_drop_pct, color: r.accuracy_drop_pct > 3 ? "#dc2626" : "#0d9488" })),
        fmt: v => "-" + v.toFixed(2) + " pts",
      });
    }
    prov("provAbl", P.ablation, "single seed; no confidence interval");
  }

  // ---------------------------------------------------------------- channel
  async function loadChannel() {
    S.chan = await api("/api/channel");
    const c = S.chan.calibrated || [];
    const draw = (metric, el) => {
      const rows = c.filter(r => r.metric === metric && MODEL_LABEL[r.model]);
      if (!rows.length) { $(el).innerHTML = '<p class="muted">Run scripts/10_multiseed_evaluation.py</p>'; return; }
      Charts.hbars($(el), {
        labelW: 200, max: 1,
        items: rows.map(r => ({ label: `${MODEL_LABEL[r.model]} · ${r.channel === "fiber" ? "fibre" : "free space"}`,
          value: r.mean, lo: r.ci95_low, hi: r.ci95_high, color: r.channel === "fiber" ? "#0d9488" : "#d97706" })),
        fmt: v => pct(v, 1),
      });
    };
    draw("binary_fpr", "#chChFpr"); draw("macro_f1", "#chChF1");
    $("#tblChannel").innerHTML = table(c.map(r => ({ model: MODEL_LABEL[r.model] || r.model, channel: r.channel, metric: r.metric, mean: r.mean, ci: r })),
      ["model", "channel", "metric", "mean", "ci"], { mean: v => v.toFixed(3), ci: r => `${r.ci95_low.toFixed(3)}&ndash;${r.ci95_high.toFixed(3)}` });
    prov("provChannel", S.chan.provenance.calibrated, "whiskers: 95% CI over seeds");
    $("#tblChannelLegacy").innerHTML = table(S.chan.legacy, ["model", "fiber_f1", "fso_f1", "fiber_fpr", "fso_fpr", "f1_drop"]);
    prov("provChannelLegacy", S.chan.provenance.legacy, "no seeds repeated; no confidence interval");
  }

  // ---------------------------------------------------------------- data
  async function loadData() {
    S.data = await api("/api/data");
    const d = S.data, P = d.provenance || {};
    const card = (name, m) => m ? `<div class="stat"><div class="bar"></div><div class="k">${esc(name.replace(/_/g, " "))}</div>
      <div class="v" style="font-size:1.3rem">${(m.rows || 0).toLocaleString()} rows</div>
      <div class="s">${m.sessions} sessions · seed ${m.seed} · config ${esc(m.config_hash)}<br>commit ${esc(String(m.git_commit).slice(0, 10))} · ${esc(m.generated_utc)}</div></div>` : "";
    $("#dataCards").innerHTML = Object.entries(d.sidecars || {}).map(([k, m]) => card(k, m)).join("") || '<p class="muted">Run scripts/06_generate_multiattack_dataset.py</p>';
    prov("provSidecars", P.sidecars);

    const tr = (d.sidecars || {}).train_multiattack;
    if (tr && tr.block_class_counts) {
      Charts.hbars($("#chBalance"), { labelW: 130,
        items: Object.entries(tr.block_class_counts).map(([k, v]) => ({ label: d.class_names[+k], value: v, color: S.meta.class_colors[+k] })),
        fmt: v => Math.round(v).toLocaleString() });
    }
    prov("provBalance", P.sidecars, "train split; class 0 includes pre-onset windows of attacked sessions");
    if (d.sessions && d.sessions.distance_km) {
      const bins = d.sessions.distance_bins;
      Charts.line($("#chDist"), { xs: bins.slice(0, -1).map((b0, i) => `${b0}-${bins[i + 1]}`), fmtY: v => v.toFixed(0),
        series: Object.entries(d.sessions.distance_km).map(([sp, h], i) => ({ name: sp, color: ["#94a3b8", "#0d9488", "#d97706"][i % 3], data: h })) });
    }
    const leak = (d.dataset_leakage || {}).classes || {};
    $("#tblLeak").innerHTML = table(Object.entries(leak).filter(([, v]) => typeof v === "object").map(([k, v]) => ({
      class: d.class_names[+k], best_auc: v.best_auc, feature: `<code>${esc(v.feature)}</code>`,
      flag: v.flagged ? `<span style="color:var(--crimson-text);font-weight:600">flagged (&gt; 0.95)</span>` : "ok" })),
      ["class", "best_auc", "feature", "flag"], { best_auc: v => v.toFixed(3) });
    const ms = d.multiseed_leakage && d.multiseed_leakage.per_seed;
    prov("provLeak", P.dataset_leakage, ms ? `multi-seed max AUC: ${Object.values(ms).map(a => a.max_auc.toFixed(3)).join(", ")}` : "");

    const c = S.meta.controls || {};
    const rows = [
      { parameter: "Fibre distance, training (km)", range: (c.fiber_distance_km || []).join(" – ") },
      { parameter: "Fibre distance, held-out (km)", range: (c.heldout_distance_km || []).join(" – ") },
      { parameter: "Detector types", range: (c.detector_types || []).join(", ") },
      ...Object.entries(c.attack_intensity || {}).map(([k, v]) => ({ parameter: `${S.meta.class_names[+k]} intensity, training / held-out`,
        range: `${v.join(" – ")}  /  ${(c.attack_heldout_intensity || {})[k].join(" – ")}` })),
    ];
    $("#tblRanges").innerHTML = table(rows, ["parameter", "range"]) + '<p class="cap" style="margin-top:8px">Full list with sources: config/physics.yaml.</p>';

    const rc = (d.real_calibration || []).filter(r => !r.noise_injected);
    if (rc.length) {
      const srcs = [...new Set(rc.map(r => r.source))];
      const losses = [...new Set(rc.map(r => r.total_loss_db))].sort((a, b) => a - b);
      const ser = srcs.map((sname, i) => ({ name: "real: " + sname.replace("baseline_no_signal_", "").replace("_full_sweep.csv", ""),
        color: ["#be185d", "#7c3aed"][i % 2], data: losses.map(L => { const r = rc.find(x => x.source === sname && x.total_loss_db === L); return r ? r.mean_qber : null; }) }));
      ser.push({ name: "simulated SPAD (same loss)", color: "#0d9488", data: losses.map(L => { const r = rc.find(x => x.total_loss_db === L); return r ? r.sim_qber_ingaas_spad : null; }) });
      ser.push({ name: "simulated SNSPD (same loss)", color: "#94a3b8", data: losses.map(L => { const r = rc.find(x => x.total_loss_db === L); return r ? r.sim_qber_snspd : null; }) });
      Charts.line($("#chReal"), { xs: losses.map(L => L + " dB"), fmtY: v => (v * 100).toFixed(1) + "%", series: ser });
    } else $("#chReal").innerHTML = '<p class="muted">Run scripts/12_real_data_evaluation.py</p>';
    const ood = d.real_ood;
    $("#realOod").innerHTML = ood && ood.real_blocks != null
      ? `<div class="banner warn"><span class="em">${ICONS.warn}</span><div><b>Out-of-distribution test on real benign telemetry.</b>
         The simulation-trained model flagged ${pct(ood.false_alarm_rate, 0)} of ${ood.real_blocks} real blocks as attacks (fixed 11% threshold: ${pct(ood.fixed11_false_alarm_rate, 0)}).
         The real system's baseline QBER is higher than the simulator's, so the model does not transfer without recalibration.</div></div>` : "";
    prov("provReal", P.real);
  }

  function renderOverview() {
    const h = S.meta.headline || {};
    if (h.accuracy) {
      $("#ovAcc").textContent = pct(h.accuracy.mean, 1);
      $("#ovAccSub").innerHTML = `95% CI ${pct(h.accuracy.ci95_low, 1)}&ndash;${pct(h.accuracy.ci95_high, 1)}, ${h.accuracy.n_seeds} seeds, session split`;
    }
    if (h.xgb_binary_fnr) {
      $("#ovFnr").textContent = pct(h.xgb_binary_fnr.mean, 1);
      $("#ovFnrSub").textContent = h.fixed11_binary_fnr ? `fixed 11% threshold misses ${pct(h.fixed11_binary_fnr.mean, 1)}` : "";
      $("#ovPerf").innerHTML = `<b>Model vs static baseline (simulated, multi-seed).</b> Attack-vs-normal, the fixed 11% threshold misses ${h.fixed11_binary_fnr ? pct(h.fixed11_binary_fnr.mean, 1) : "&mdash;"} of attacked windows; XGBoost misses ${pct(h.xgb_binary_fnr.mean, 1)} at a ${h.xgb_binary_fpr ? pct(h.xgb_binary_fpr.mean, 1) : "&mdash;"} false-alarm rate. Most of the remaining misses are the three attacks that leave QBER unchanged by design.`;
    }
    prov("provOverview", S.meta.headline_provenance);
  }

  // ---------------------------------------------------------------- atlas
  function renderAtlas() {
    $("#atlasPick").innerHTML = S.meta.class_names.map((n, i) =>
      `<button class="chip" data-i="${i}" style="cursor:pointer${i === 0 ? ";border-color:var(--accent);color:var(--accent)" : ""}">${esc(n)}</button>`).join("");
    $$("#atlasPick .chip").forEach(c => c.addEventListener("click", () => {
      S.atlas = +c.dataset.i;
      $$("#atlasPick .chip").forEach((x, j) => {
        x.style.borderColor = j === S.atlas ? "var(--accent)" : "";
        x.style.color = j === S.atlas ? "var(--accent)" : "";
      });
      drawAtlasBody();
    }));
    drawAtlasBody();

    $("#featGroups").innerHTML = Object.entries(S.meta.feature_groups).map(([g, f]) => `
      <details><summary>${esc(g.replace(/_/g, " "))} <span class="muted">(${f.length} features)</span></summary>
      <div class="body"><div class="chips">${f.map(x => `<span class="chip">${esc(x)}</span>`).join("")}</div></div>
      </details>`).join("");

    const c = S.meta.controls || {};
    $("#tblNuisance").innerHTML = `<table><thead><tr><th>Parameter</th><th>Range (all classes)</th></tr></thead><tbody>
      <tr><td>Fibre distance (training)</td><td>${(c.fiber_distance_km || []).join(" – ")} km</td></tr>
      <tr><td>Detector</td><td>${(c.detector_types || []).join(", ")}</td></tr>
      <tr><td colspan="2" class="muted">Efficiencies, dark counts, misalignment, drift and loss steps: see config/physics.yaml.</td></tr>
      </tbody></table>`;
  }

  function drawAtlasBody() {
    const p = S.meta.class_physics[S.atlas];
    $("#atlasBody").innerHTML = `
      <div class="banner info"><span class="em">${ICONS.info}</span><div><b>Physical Mechanism:</b> ${esc(p.what)}</div></div>
      <div class="banner warn" style="margin-top:9px"><span class="em">${ICONS.warn}</span>
        <div><b>Observable Telemetry Signature:</b> ${esc(p.watch)}</div></div>
      <div class="banner bad" style="margin-top:9px"><span class="em">${ICONS.alert}</span>
        <div><b>Detection Challenge:</b> ${esc(p.hard)}</div></div>`;
  }

  boot();
})();
