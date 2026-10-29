/* QKD SOC dashboard - state, routing and view rendering. */
(function () {
  "use strict";

  const $ = s => document.querySelector(s);
  const $$ = s => [...document.querySelectorAll(s)];
  const pct = (v, d = 1) => (v * 100).toFixed(d) + "%";
  const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

  const S = {
    meta: null, session: null, scn: null, src: "trial",
    idx: 0, playing: true, timer: null, stage: null,
    mdi: null, bench: null, atlas: 0, drawnOnce: {},
  };

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
      $("#healthDot").classList.toggle("off", !h.model);
      $("#healthTxt").textContent = h.model ? "model online" : "model missing";
    } catch {
      $("#healthDot").classList.add("off");
      $("#healthTxt").textContent = "backend offline";
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
    selectScenario(0);
    run();
  }

  // ---------------------------------------------------------------- monitor
  function renderScenarios() {
    $("#scnList").innerHTML = S.meta.scenarios.map((s, i) => `
      <button class="scn${i === 0 ? " on" : ""}" data-i="${i}">
        <span class="n">${i + 1}</span>
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
    $("#lblBlocks").textContent = $("#inBlocks").value;
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

  $$("#srcSeg button").forEach(b => b.addEventListener("click", () => {
    $$("#srcSeg button").forEach(x => x.classList.toggle("on", x === b));
    S.src = b.dataset.src;
    $("#liveOpts").style.display = S.src === "live" ? "" : "none";
  }));

  $("#btnRun").addEventListener("click", run);

  async function run() {
    const btn = $("#btnRun");
    btn.disabled = true; btn.textContent = "⏳ Simulating…";
    stopPlay();
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
      $("#verdictA").innerHTML = `<div class="banner bad"><span class="em">⚠</span><div>${esc(e.message)}</div></div>`;
    } finally {
      btn.disabled = false; btn.textContent = "▶ Run scenario";
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
      kpi("Mean QBER after onset", pct(m.mean_qber_post, 2), "11% is the abort line",
          m.mean_qber_post > 0.11 ? "bad" : "") +
      kpi("Transmission efficiency", pct(m.final_efficiency, 1), "the PNS giveaway", "info") +
      kpi("Basis asymmetry", pct(m.final_asymmetry, 2), "the time-shift giveaway", "warn") +
      (m.delta_t == null
        ? kpi("Detection latency", "no alarm", "model stayed on Normal", s.class_id === 0 ? "good" : "bad")
        : kpi("Detection latency ΔT", m.delta_t + " blk", "first alarm at block " + m.first_alarm, "bad"));

    $("#verdictA").innerHTML = m.threshold_fires
      ? `<div class="banner bad"><span class="em">⚑</span><div><b>Classical 11% threshold: FIRED.</b>
         Loud enough for the textbook rule.</div></div>`
      : `<div class="banner warn"><span class="em">◌</span><div><b>Classical 11% threshold: SILENT.</b>
         It never crossed 11%, so a conventional link would have kept distilling the key.</div></div>`;

    let vb;
    if (s.class_id === 0) {
      const fp = m.false_positive_rate;
      vb = `<div class="banner ${fp < .1 ? "ok" : "bad"}"><span class="em">${fp < .1 ? "✓" : "⚠"}</span>
        <div><b>Temporal model:</b> flagged ${pct(fp, 1)} of blocks on a benign link (false positives).</div></div>`;
    } else if (m.delta_t != null) {
      vb = `<div class="banner ok"><span class="em">✓</span><div><b>Temporal model: DETECTED</b> in
        ${m.delta_t} blocks, and named the right attack on ${pct(m.post_accuracy, 0)} of post-onset blocks.</div></div>`;
    } else {
      vb = `<div class="banner bad"><span class="em">✕</span><div><b>Temporal model: MISSED.</b>
        This one slipped through.</div></div>`;
    }
    if (s.p_eve > 0 && s.p_eve < 0.12 && s.class_id !== 0) {
      vb += `<div class="banner info" style="margin-top:9px"><span class="em">ℹ</span><div>
        This session drew a very weak attack strength (p<sub>eve</sub> = ${s.p_eve}). Near the detection
        floor the model genuinely struggles, and that is reported rather than hidden.</div></div>`;
    }
    $("#verdictB").innerHTML = vb;

    const ph = S.meta.class_physics[s.class_id];
    $("#scnPhysics").innerHTML = `
      <div class="kv"><span>Attack strength drawn this session</span><span>p_eve = ${s.p_eve}</span></div>
      <div class="kv"><span>Ground truth</span><span>${esc(s.class_full)}</span></div>
      <p style="margin:13px 0 0"><b>What Eve is doing.</b> <span class="muted">${esc(ph.what)}</span></p>
      <p style="margin:9px 0 0"><b>What it looks like.</b> <span class="muted">${esc(ph.watch)}</span></p>
      <p style="margin:9px 0 0"><b>Why it is hard.</b> <span class="muted">${esc(ph.hard)}</span></p>
      <p class="muted" style="margin:13px 0 0;font-size:.82rem">Every session also draws its own baseline QBER,
      channel loss, dark-count rate and detector basis mismatch from overlapping ranges, so no class is
      identifiable from a single number.</p>`;

    drawMonitor(true);
  }

  function drawMonitor(first) {
    const s = S.session, n = S.idx + 1;
    const xs = s.blocks.slice(0, n);
    const cut = a => a.slice(0, n);
    const onsetI = s.blocks.indexOf(s.onset);
    const alarmI = s.summary.first_alarm != null ? s.blocks.indexOf(s.summary.first_alarm) : null;
    const anim = !!first;

    Charts.line($("#chQber"), {
      xs, animate: anim, pulseLast: true, yMax: 0.14,
      fmtY: v => (v * 100).toFixed(0) + "%",
      series: [{ name: "window-mean QBER", color: "#22d3ee", data: cut(s.qber), fill: true }],
      threshold: { y: 0.11, color: "#f43f5e", label: "11% abort" },
      markers: [
        { i: onsetI <= S.idx ? onsetI : null, color: "#fbbf24", label: "onset" },
        { i: alarmI != null && alarmI <= S.idx ? alarmI : null, color: "#34d399", label: "alarm" },
      ],
    });

    Charts.line($("#chBasis"), {
      xs, animate: anim, pulseLast: true,
      fmtY: v => (v * 100).toFixed(0) + "%",
      series: [
        { name: "Q_Z rectilinear", color: "#818cf8", data: cut(s.qber_z) },
        { name: "Q_X diagonal", color: "#fbbf24", data: cut(s.qber_x) },
      ],
      markers: [{ i: onsetI <= S.idx ? onsetI : null, color: "#fbbf24", label: "onset" }],
    });

    Charts.line($("#chEff"), {
      xs, animate: anim, pulseLast: true, yMax: 1,
      fmtY: v => (v * 100).toFixed(0) + "%",
      series: [{ name: "transmission efficiency", color: "#a78bfa", data: cut(s.efficiency), fill: true }],
      markers: [{ i: onsetI <= S.idx ? onsetI : null, color: "#fbbf24", label: "onset" }],
    });

    const p = s.probs[S.idx] || [];
    if (!$("#probBars").children.length) {
      $("#probBars").innerHTML = S.meta.class_names.map((nm, k) => `
        <div class="pb"><div class="row"><span class="nm">${esc(nm)}</span>
        <span class="vl" data-v="${k}">0%</span></div>
        <div class="track"><div class="fill" data-f="${k}"
          style="background:linear-gradient(90deg,${S.meta.class_colors[k]}99,${S.meta.class_colors[k]})"></div></div></div>`).join("");
    }
    p.forEach((v, k) => {
      $(`#probBars .fill[data-f="${k}"]`).style.width = (v * 100).toFixed(1) + "%";
      $(`#probBars .vl[data-v="${k}"]`).textContent = pct(v, 1);
    });

    const pred = s.pred[S.idx];
    $("#monBody").querySelectorAll(".panel")[0].classList.toggle("alarming", pred !== 0);

    $("#scrubLbl").textContent = `block ${s.blocks[S.idx]} / ${s.blocks[s.blocks.length - 1]}`;
    $("#scrub").value = S.idx;
    syncRange();

    const active = s.blocks[S.idx] >= s.onset && s.class_id !== 0;
    S.stage.set({
      attackActive: active, attackClass: s.class_id,
      intercept: active ? Math.max(s.p_eve, .15) : 0,
      qber: s.qber[S.idx], efficiency: s.efficiency[S.idx],
    });
  }

  function startPlay() {
    S.playing = true;
    $("#btnPlay").textContent = "⏸ Pause";
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
    const b = $("#btnPlay"); if (b) b.textContent = "▶ Play";
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
      $("#explainCtl").innerHTML = `<div class="banner info"><span class="em">ℹ</span>
        <div>Run a scenario in <b>Live Monitor</b> first.</div></div>`;
      return;
    }
    const s = S.session;
    if (!$("#exBlock")) {
      $("#explainCtl").innerHTML = `<div class="control" style="max-width:620px;margin:0">
        <label>Block to explain &mdash; <b id="exLbl">${s.blocks[S.idx]}</b></label>
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
      $("#chShap").innerHTML = `<div class="banner bad"><span class="em">⚠</span><div>${esc(e.message)}</div></div>`;
      return;
    }
    const truth = S.session.class_id;
    const ok = d.pred === truth;

    $("#explainMeta").innerHTML = `
      <div class="stat ${d.pred === 0 ? "good" : "bad"}"><div class="bar"></div>
        <div class="k">Model verdict</div><div class="v" style="font-size:1.3rem">${esc(d.pred_name)}</div>
        <div class="s">confidence ${pct(d.probs[d.pred], 1)}</div></div>
      <div class="stat info"><div class="bar"></div><div class="k">Ground truth</div>
        <div class="v" style="font-size:1.3rem">${esc(S.meta.class_names[truth])}</div>
        <div class="s">from the simulator</div></div>
      <div class="stat ${ok ? "good" : "warn"}"><div class="bar"></div><div class="k">Outcome</div>
        <div class="v" style="font-size:1.3rem">${ok ? "Correct" : "Misclassified"}</div>
        <div class="s">block ${d.block}</div></div>`;

    Charts.hbars($("#chShap"), {
      diverging: true, labelW: 176,
      items: d.top_features.map(f => ({ label: f.feature, value: f.shap })),
      fmt: v => (v >= 0 ? "+" : "") + v.toFixed(3),
    });

    const gs = Object.entries(d.groups).sort((a, b) => b[1] - a[1]);
    $("#shapGroups").innerHTML = `<div class="grid g3" style="gap:10px">` + gs.map(([g, v]) => `
      <div class="stat"><div class="bar"></div><div class="k">${esc(g.replace(/_/g, " "))}</div>
      <div class="v" style="font-size:1.45rem">${pct(v, 0)}</div></div>`).join("") + `</div>`;

    const t = d.top_features.slice(0, 2).map(f => `<code>${esc(f.feature)}</code>`);
    const reasons = [
      `${t[0]} and ${t[1]} sit where an unattacked link sits: a stable error rate, symmetric bases, and a yield consistent with the channel loss.`,
      `${t[0]} and ${t[1]} show the temporal signature of intercept-resend &mdash; a step up in error level with raised variance and jump energy, while the absolute QBER stays legal.`,
      `${t[0]} and ${t[1]} point at the yield, not the error rate. The key rate has fallen further than the loss budget allows with no matching QBER rise &mdash; the fingerprint of multi-photon splitting.`,
      `${t[0]} and ${t[1]} show an unnaturally <i>clean</i>, regular channel. Blinded detectors driven by classical pulses produce fewer and more evenly spaced errors than real quantum detection ever does.`,
      `${t[0]} and ${t[1]} capture disagreement between the two bases. The mean error is unremarkable, but Z and X have diverged &mdash; detector efficiency mismatch being exploited.`,
    ];
    $("#shapWords").innerHTML = `<p style="margin:0">${reasons[d.pred]}</p>` + (ok ? "" :
      `<div class="banner warn" style="margin-top:12px"><span class="em">⚠</span><div>This block was
       <b>misclassified</b>. Misses cluster right after onset, before the W=50 window has filled with
       attacked blocks, and on weak attack strengths.</div></div>`);
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
          { name: "BB84 (what Eve lets you see)", color: "#f43f5e", data: a.map(r => r.bb84_check_error) },
          { name: "MDI-QKD X-basis phase error", color: "#22d3ee", data: a.map(r => r.mdi_check_error) },
        ],
        threshold: { y: 0.11, color: "#fbbf24", label: "11% abort" },
      });
    }
    if (kr) {
      Charts.logLine($("#chMdiRate"), {
        xs: kr.map(r => r.distance_km),
        series: [
          { name: "PLOB bound", color: "#64748b", data: kr.map(r => r.plob_bound), dash: true },
          { name: "SNS-TF-QKD", color: "#a78bfa", data: kr.map(r => r.sns_tf_qkd) },
          { name: "MDI-QKD", color: "#22d3ee", data: kr.map(r => r.mdi_qkd) },
          { name: "Decoy BB84", color: "#fbbf24", data: kr.map(r => r.decoy_bb84) },
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
       <div class="v" style="font-size:1.55rem">${val}</div><div class="s">${sub}</div></div>`;

    $("#mdiBB84").innerHTML =
      `<div class="grid" style="gap:10px">
        ${tile("Key bits Eve learns", pct(r.bb84_eve_information, 0), "she reads them directly", "bad")}
        ${tile("Error rate you observe", pct(r.bb84_qber, 2), "below 11% - no abort", "bad")}
      </div>
      <div class="banner bad" style="margin-top:12px"><span class="em">✕</span><div><b>No alarm.</b>
        Eve also controls the check basis, so every quantity you could monitor is one she is forging.</div></div>`;

    $("#mdiMDI").innerHTML =
      `<div class="grid" style="gap:10px">
        ${tile("Z-basis error you observe", pct(r.mdi_qber, 2), "she can keep this clean", "")}
        ${tile("X-basis phase error", pct(r.mdi_check_error, 1), "she cannot forge this",
               r.mdi_aborts ? "bad" : "good")}
      </div>` + (r.mdi_aborts
        ? `<div class="banner ok" style="margin-top:12px"><span class="em">✓</span><div><b>Protocol aborts.</b>
           Measuring the pulses to learn Z bits destroys their X-basis coherence. Alice and Bob detect it by
           comparing Charlie's announcement against what they know they prepared.</div></div>`
        : `<div class="banner info" style="margin-top:12px"><span class="em">ℹ</span><div>Below the abort
           threshold &mdash; Eve is learning little at this level of control.</div></div>`);

    $("#panelMDI").classList.toggle("alarming", !!r.mdi_aborts);
    const first = rows.find(x => x.mdi_aborts);
    if (first && !$("#mdiCross")) {
      const p = document.createElement("p");
      p.id = "mdiCross"; p.className = "muted"; p.style.marginTop = "14px";
      p.innerHTML = `<b>Crossover:</b> MDI-QKD trips its monitor once Eve controls just
        <b style="color:var(--cyan)">${pct(first.p_control, 0)}</b> of the relay. BB84 never trips at any
        control fraction.`;
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
      tile("Accuracy", pct(sm.accuracy?.mean ?? 0, 2), "± " + pct(sm.accuracy?.std ?? 0, 2) + " over 5 seeds", "info") +
      tile("Macro-F1", (sm.macro_f1?.mean ?? 0).toFixed(4), "± " + (sm.macro_f1?.std ?? 0).toFixed(4), "info") +
      tile("Model FNR", pct(sm.xgb_binary_fnr?.mean ?? 0, 2), "the 11% rule: " + pct(sm.fixed11_binary_fnr?.mean ?? 0, 1), "good") +
      tile("Max single-feature AUC", (sm.max_single_feature_auc?.mean ?? 0).toFixed(3), "leakage audit", "warn");

    if (b.multiseed) {
      Charts.dots($("#chSeeds"), {
        values: b.multiseed.map(r => r.accuracy),
        labels: b.multiseed.map(r => r.seed),
      });
    }

    const table = (rows, hi) => {
      if (!rows || !rows.length) return '<p class="muted">not generated yet</p>';
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
          color: r.accuracy_drop_pct > 3 ? "#f43f5e" : "#22d3ee",
        })),
        fmt: v => "-" + v.toFixed(2) + "%",
      });
    }
  }

  // ---------------------------------------------------------------- atlas
  function renderAtlas() {
    $("#atlasPick").innerHTML = S.meta.class_names.map((n, i) =>
      `<button class="chip" data-i="${i}" style="cursor:pointer${i === 0 ? ";border-color:var(--cyan);color:var(--cyan)" : ""}">${esc(n)}</button>`).join("");
    $$("#atlasPick .chip").forEach(c => c.addEventListener("click", () => {
      S.atlas = +c.dataset.i;
      $$("#atlasPick .chip").forEach((x, j) => {
        x.style.borderColor = j === S.atlas ? "var(--cyan)" : "";
        x.style.color = j === S.atlas ? "var(--cyan)" : "";
      });
      drawAtlasBody();
    }));
    drawAtlasBody();

    $("#featGroups").innerHTML = Object.entries(S.meta.feature_groups).map(([g, f]) => `
      <details><summary>${esc(g.replace(/_/g, " "))} <span class="muted">(${f.length})</span></summary>
      <div class="body"><div class="chips">${f.map(x => `<span class="chip">${esc(x)}</span>`).join("")}</div></div>
      </details>`).join("");

    const nr = S.meta.nuisance_ranges;
    $("#tblNuisance").innerHTML = `<table><thead><tr><th>Parameter</th><th>Min</th><th>Max</th></tr></thead>
      <tbody>${Object.entries(nr).map(([k, v]) =>
        `<tr><td><code>${esc(k)}</code></td><td>${v[0]}</td><td>${v[1]}</td></tr>`).join("")}</tbody></table>`;
  }

  function drawAtlasBody() {
    const p = S.meta.class_physics[S.atlas];
    $("#atlasBody").innerHTML = `
      <div class="banner info"><span class="em">◆</span><div><b>What Eve does.</b> ${esc(p.what)}</div></div>
      <div class="banner warn" style="margin-top:9px"><span class="em">◎</span>
        <div><b>Observable signature.</b> ${esc(p.watch)}</div></div>
      <div class="banner bad" style="margin-top:9px"><span class="em">⚠</span>
        <div><b>Why it is hard.</b> ${esc(p.hard)}</div></div>`;
  }

  boot();
})();
