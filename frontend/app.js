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
    mdi: null, bench: null, atlas: 0, drawnOnce: {},
  };

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

    $$("[data-count]").forEach(n => {
      const dec = n.dataset.dec != null ? +n.dataset.dec : 2;
      Charts.countUp(n, +n.dataset.count, dec, n.dataset.suffix || "");
    });

    renderScenarios();
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
    $("#guidedSide").style.display = isBlind ? "none" : "";
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

    Charts.line($("#chQber"), {
      xs, animate: anim, pulseLast: true, yMax: 0.14,
      fmtY: v => (v * 100).toFixed(0) + "%",
      series: [{ name: "window-mean QBER", color: "#0d9488", data: cut(s.qber), fill: true }],
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
      xs, animate: anim, pulseLast: true, yMax: 1,
      fmtY: v => (v * 100).toFixed(0) + "%",
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
    $("#chShap").innerHTML = '<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>';
    let d;
    try { d = await api(`/api/explain?session_id=${S.session.session_id}&block=${block}`); }
    catch (e) {
      $("#chShap").innerHTML = `<div class="banner bad"><span class="em">${ICONS.alert}</span><div>${esc(e.message)}</div></div>`;
      return;
    }
    const truth = S.session.class_id;
    const ok = d.pred === truth;

    $("#explainMeta").innerHTML = `
      <div class="stat ${d.pred === 0 ? "good" : "bad"}"><div class="bar"></div>
        <div class="k">Model Verdict</div><div class="v" style="font-size:1.2rem">${esc(d.pred_name)}</div>
        <div class="s">Confidence ${pct(d.probs[d.pred], 1)}</div></div>
      <div class="stat info"><div class="bar"></div><div class="k">Ground Truth</div>
        <div class="v" style="font-size:1.2rem">${esc(S.meta.class_names[truth])}</div>
        <div class="s">Simulated physical regime</div></div>
      <div class="stat ${ok ? "good" : "warn"}"><div class="bar"></div><div class="k">Audit Outcome</div>
        <div class="v" style="font-size:1.2rem">${ok ? "Attribution Validated" : "Misclassified"}</div>
        <div class="s">Block ${d.block}</div></div>`;

    Charts.hbars($("#chShap"), {
      diverging: true, labelW: 176,
      items: d.top_features.map(f => ({ label: f.feature, value: f.shap })),
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(3),
    });

    const gs = Object.entries(d.groups).sort((a, b) => b[1] - a[1]);
    $("#shapGroups").innerHTML = `<div class="grid g3" style="gap:10px">` + gs.map(([g, v]) => `
      <div class="stat"><div class="bar"></div><div class="k">${esc(g.replace(/_/g, " "))}</div>
      <div class="v" style="font-size:1.35rem">${pct(v, 0)}</div></div>`).join("") + `</div>`;

    const t = d.top_features.slice(0, 2).map(f => `<code>${esc(f.feature)}</code>`);
    const reasons = [
      `${t[0]} and ${t[1]} align with uncompromised quantum channel dynamics: stationary error moments, symmetric conjugate bases, and transmission yield consistent with loss.`,
      `${t[0]} and ${t[1]} exhibit the characteristic signature of intercept-resend: discontinuous step elevation in error variance and temporal jump energy, while scalar QBER remains below 11%.`,
      `${t[0]} and ${t[1]} identify an anomalous yield deficit. Multi-photon pulse splitting depresses detected key yield without proportional error rate elevation &mdash; the unambiguous fingerprint of PNS.`,
      `${t[0]} and ${t[1]} reflect an unnaturally uniform, deterministic pulse train. Blinded single-photon detectors driven by classical laser power yield lower error dispersion than physical quantum Poisson noise.`,
      `${t[0]} and ${t[1]} capture conjugate basis divergence. While mean aggregate QBER remains unremarkable, rectilinear (Z) and diagonal (X) error rates diverge significantly &mdash; exploiting detector timing differences.`,
    ];
    $("#shapWords").innerHTML = `<p style="margin:0;line-height:1.55;">${reasons[d.pred]}</p>` + (ok ? "" :
      `<div class="banner warn" style="margin-top:12px"><span class="em">${ICONS.warn}</span><div>This specific transmission block exhibited transient misclassification. Borderline attribution occurs primarily immediately following onset before the sliding window fully fills with attacked pulses.</div></div>`);
  }

  // ---------------------------------------------------------------- mdi
  async function loadMdi() {
    S.mdi = await api("/api/mdi");
    $("#inCtrl").addEventListener("input", () => { syncRange(); renderMdiCards(); });
    syncRange(); renderMdiCards();

    const a = S.mdi.attack, kr = S.mdi.key_rate;
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
  async function loadBench() {
    S.bench = await api("/api/benchmarks");
    const b = S.bench;
    const sm = {};
    (b.summary || []).forEach(r => { sm[r[Object.keys(r)[0]]] = r; });

    const tile = (k, v, s, kind) =>
      `<div class="stat ${kind}"><div class="bar"></div><div class="k">${k}</div>
       <div class="v">${v}</div><div class="s">${s}</div></div>`;
    $("#benchKpis").innerHTML =
      tile("Held-Out Accuracy", pct(sm.accuracy?.mean ?? 0, 2), "± " + pct(sm.accuracy?.std ?? 0, 2) + " over 5 seeds", "info") +
      tile("Macro-F1 Score", (sm.macro_f1?.mean ?? 0).toFixed(4), "± " + (sm.macro_f1?.std ?? 0).toFixed(4), "info") +
      tile("Temporal Model FNR", pct(sm.xgb_binary_fnr?.mean ?? 0, 2), "Conventional 11% rule: " + pct(sm.fixed11_binary_fnr?.mean ?? 0, 1), "good") +
      tile("Max Single-Feature AUC", (sm.max_single_feature_auc?.mean ?? 0).toFixed(3), "No synthetic label leakage", "warn");

    if (b.multiseed) {
      Charts.dots($("#chSeeds"), {
        values: b.multiseed.map(r => r.accuracy),
        labels: b.multiseed.map(r => r.seed),
      });
    }

    const table = (rows, hi) => {
      if (!rows || !rows.length) return '<p class="muted">Generating benchmarks...</p>';
      const cols = Object.keys(rows[0]);
      return `<table><thead><tr>${cols.map(c => `<th>${esc(c.replace(/_/g, " "))}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map(c => {
          let v = r[c];
          if (typeof v === "number") v = Math.abs(v) < 1 && v !== 0 ? v.toFixed(4) : v.toFixed(3);
          return `<td${hi && hi(c, r) ? ' class="hi"' : ""}>${esc(v)}</td>`;
        }).join("")}</tr>`).join("")}</tbody></table>`;
    };
    $("#tblOverall").innerHTML = table(b.overall);
    $("#tblPerClass").innerHTML = table(b.per_class, c => c === "f1_score");

    if (b.ablation) {
      const rows = [...b.ablation].sort((x, y) => y.accuracy_drop_pct - x.accuracy_drop_pct);
      Charts.hbars($("#chAbl"), {
        labelW: 158,
        items: rows.map(r => ({
          label: r.display_name, value: r.accuracy_drop_pct,
          color: r.accuracy_drop_pct > 3 ? "#dc2626" : "#0d9488",
        })),
        fmt: v => "-" + v.toFixed(2) + "%",
      });
    }
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

    const nr = S.meta.nuisance_ranges;
    $("#tblNuisance").innerHTML = `<table><thead><tr><th>Physical Parameter</th><th>Lower Bound</th><th>Upper Bound</th></tr></thead>
      <tbody>${Object.entries(nr).map(([k, v]) =>
        `<tr><td><code>${esc(k)}</code></td><td>${v[0]}</td><td>${v[1]}</td></tr>`).join("")}</tbody></table>`;
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
