/* Animated SVG chart primitives. No external dependencies. */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a = {}) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    return e;
  };

  function frame(w, h, pad) {
    const svg = el("svg", {
      viewBox: `0 0 ${w} ${h}`, width: "100%",
      style: `height:${h}px;display:block;overflow:visible`,
    });
    return { svg, pad, w, h, iw: w - pad.l - pad.r, ih: h - pad.t - pad.b };
  }

  function grid(f, yTicks, fmtY) {
    for (const t of yTicks) {
      const y = f.pad.t + f.ih - t.p * f.ih;
      f.svg.appendChild(el("line", {
        x1: f.pad.l, x2: f.pad.l + f.iw, y1: y, y2: y,
        stroke: "#e2e8f0", "stroke-width": 1, "stroke-dasharray": "3 5",
      }));
      const lab = el("text", {
        x: f.pad.l - 8, y: y + 4, fill: "#64748b", "font-size": 10,
        "text-anchor": "end", "font-family": "ui-monospace,monospace",
      });
      lab.textContent = fmtY(t.v);
      f.svg.appendChild(lab);
    }
  }

  function xlabels(f, xs, n = 5) {
    if (!xs.length) return;
    for (let i = 0; i < n; i++) {
      const idx = Math.round((xs.length - 1) * (i / (n - 1)));
      const x = f.pad.l + (idx / Math.max(xs.length - 1, 1)) * f.iw;
      const t = el("text", {
        x, y: f.pad.t + f.ih + 16, fill: "#64748b", "font-size": 10,
        "text-anchor": "middle", "font-family": "ui-monospace,monospace",
      });
      t.textContent = xs[idx];
      f.svg.appendChild(t);
    }
  }

  /** Multi-series line chart with draw-in animation.
   *  opts: {xs, series:[{name,color,data,fill?}], yMax, yMin, fmtY, markers:[{x,color,label}],
   *         threshold:{y,color,label}, height, animate, pulseLast} */
  function line(container, opts) {
    const o = Object.assign({
      height: 212, yMin: 0, fmtY: v => v.toFixed(0), markers: [], animate: true,
      pulseLast: false, series: [], xs: [],
    }, opts);
    container.innerHTML = "";
    const pad = { l: 42, r: 12, t: 10, b: 26 };
    const f = frame(620, o.height, pad);

    const all = o.series.flatMap(s => s.data).filter(v => v != null && isFinite(v));
    let yMax = o.yMax != null ? o.yMax : (all.length ? Math.max(...all) * 1.18 : 1);
    if (o.threshold) yMax = Math.max(yMax, o.threshold.y * 1.12);
    const yMin = o.yMin;
    const sx = i => pad.l + (i / Math.max(o.xs.length - 1, 1)) * f.iw;
    const sy = v => pad.t + f.ih - ((v - yMin) / (yMax - yMin || 1)) * f.ih;

    grid(f, [0, .25, .5, .75, 1].map(p => ({ p, v: yMin + p * (yMax - yMin) })), o.fmtY);
    xlabels(f, o.xs);

    // onset / alarm markers
    for (const m of o.markers) {
      if (m.i == null) continue;
      const x = sx(m.i);
      f.svg.appendChild(el("line", {
        x1: x, x2: x, y1: pad.t, y2: pad.t + f.ih, stroke: m.color,
        "stroke-width": 1.6, "stroke-dasharray": "4 4", opacity: .85,
      }));
      const t = el("text", {
        x: x + 4, y: pad.t + 11, fill: m.color, "font-size": 9.5, "font-weight": 600,
      });
      t.textContent = m.label;
      f.svg.appendChild(t);
    }

    if (o.threshold) {
      const y = sy(o.threshold.y);
      f.svg.appendChild(el("line", {
        x1: pad.l, x2: pad.l + f.iw, y1: y, y2: y, stroke: o.threshold.color,
        "stroke-width": 1.6, "stroke-dasharray": "6 4",
      }));
      const t = el("text", {
        x: pad.l + f.iw - 4, y: y - 6, fill: o.threshold.color, "font-size": 9.5,
        "text-anchor": "end", "font-weight": 600,
      });
      t.textContent = o.threshold.label;
      f.svg.appendChild(t);
    }

    o.series.forEach((s, si) => {
      const pts = s.data.map((v, i) => (v == null ? null : [sx(i), sy(v)])).filter(Boolean);
      if (pts.length < 2) return;
      const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");

      if (s.fill) {
        const area = el("path", {
          d: d + ` L${pts[pts.length - 1][0].toFixed(1)} ${pad.t + f.ih} L${pts[0][0].toFixed(1)} ${pad.t + f.ih} Z`,
          fill: s.color, opacity: .10,
        });
        area.style.animation = "fadeUp .7s var(--ease-out) both";
        f.svg.appendChild(area);
      }

      const path = el("path", {
        d, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
        "stroke-linecap": "round", "stroke-linejoin": "round",
      });
      if (o.animate) {
        const len = 2600;
        path.style.strokeDasharray = len;
        path.style.strokeDashoffset = len;
        path.style.animation = `draw 1.05s var(--ease-out) ${si * .09}s forwards`;
      }
      f.svg.appendChild(path);

      if (o.pulseLast) {
        const last = pts[pts.length - 1];
        const halo = el("circle", { cx: last[0], cy: last[1], r: 7, fill: s.color, opacity: .25 });
        halo.style.animation = "haloPulse 1.6s ease-in-out infinite";
        f.svg.appendChild(halo);
        f.svg.appendChild(el("circle", {
          cx: last[0], cy: last[1], r: 3.2, fill: s.color,
          stroke: "#ffffff", "stroke-width": 1.5,
        }));
      }
    });

    // legend
    if (o.series.length > 1 || o.series[0]?.name) {
      const lg = el("g", {});
      let cx = pad.l;
      o.series.forEach(s => {
        lg.appendChild(el("rect", { x: cx, y: 1, width: 11, height: 2.6, rx: 1.3, fill: s.color }));
        const t = el("text", { x: cx + 15, y: 5, fill: "#475569", "font-size": 10 });
        t.textContent = s.name;
        lg.appendChild(t);
        cx += 19 + (s.name.length * 5.6);
      });
      f.svg.appendChild(lg);
    }

    container.appendChild(f.svg);
  }

  /** Log-scale multi-series line (key rate vs distance). */
  function logLine(container, opts) {
    const o = Object.assign({ height: 236, floor: 1e-12 }, opts);
    container.innerHTML = "";
    const pad = { l: 52, r: 12, t: 14, b: 26 };
    const f = frame(620, o.height, pad);

    const vals = o.series.flatMap(s => s.data).filter(v => v && v > 0);
    const hi = Math.ceil(Math.log10(Math.max(...vals)));
    const lo = Math.floor(Math.log10(Math.max(Math.min(...vals), o.floor)));
    const sx = i => pad.l + (i / Math.max(o.xs.length - 1, 1)) * f.iw;
    const sy = v => {
      const L = Math.log10(Math.max(v, Math.pow(10, lo)));
      return pad.t + f.ih - ((L - lo) / (hi - lo || 1)) * f.ih;
    };

    for (let p = lo; p <= hi; p += Math.max(1, Math.round((hi - lo) / 5))) {
      const y = sy(Math.pow(10, p));
      f.svg.appendChild(el("line", {
        x1: pad.l, x2: pad.l + f.iw, y1: y, y2: y, stroke: "#e2e8f0",
        "stroke-width": 1, "stroke-dasharray": "3 5",
      }));
      const t = el("text", {
        x: pad.l - 8, y: y + 4, fill: "#64748b", "font-size": 9.5,
        "text-anchor": "end", "font-family": "ui-monospace,monospace",
      });
      t.textContent = "1e" + p;
      f.svg.appendChild(t);
    }
    xlabels(f, o.xs, 6);

    o.series.forEach((s, si) => {
      const pts = [];
      s.data.forEach((v, i) => { if (v && v > 0) pts.push([sx(i), sy(v)]); });
      if (pts.length < 2) return;
      const d = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
      const path = el("path", {
        d, fill: "none", stroke: s.color, "stroke-width": s.dash ? 1.7 : 2.2,
        "stroke-linecap": "round", "stroke-dasharray": s.dash ? "6 4" : "",
      });
      if (!s.dash) {
        const len = 2600;
        path.style.strokeDasharray = len;
        path.style.strokeDashoffset = len;
        path.style.animation = `draw 1.1s var(--ease-out) ${si * .1}s forwards`;
      }
      f.svg.appendChild(path);
    });

    const lg = el("g", {});
    let cy = pad.t + 2;
    o.series.forEach(s => {
      lg.appendChild(el("rect", { x: pad.l + f.iw - 150, y: cy, width: 11, height: 2.6, rx: 1.3, fill: s.color }));
      const t = el("text", { x: pad.l + f.iw - 134, y: cy + 4, fill: "#475569", "font-size": 9.5 });
      t.textContent = s.name;
      lg.appendChild(t);
      cy += 14;
    });
    f.svg.appendChild(lg);
    container.appendChild(f.svg);
  }

  /** Horizontal bars; supports signed values (diverging) for SHAP. */
  function hbars(container, opts) {
    const o = Object.assign({ height: null, diverging: false, fmt: v => v.toFixed(3), labelW: 168 }, opts);
    container.innerHTML = "";
    const rows = o.items.length;
    const rh = 22, h = o.height || rows * rh + 16;
    const pad = { l: o.labelW, r: 54, t: 8, b: 8 };
    const f = frame(620, h, pad);

    const mx = Math.max(...o.items.map(i => Math.abs(i.value)), 1e-9);
    const zero = o.diverging ? pad.l + f.iw / 2 : pad.l;
    const scale = o.diverging ? (f.iw / 2) / mx : f.iw / mx;

    if (o.diverging) {
      f.svg.appendChild(el("line", {
        x1: zero, x2: zero, y1: pad.t, y2: pad.t + rows * rh, stroke: "#cbd5e1", "stroke-width": 1,
      }));
    }

    o.items.forEach((it, i) => {
      const y = pad.t + i * rh + 4;
      const w = Math.abs(it.value) * scale;
      const x = it.value < 0 && o.diverging ? zero - w : zero;
      const color = it.color || (o.diverging ? (it.value >= 0 ? "#dc2626" : "#0d9488") : "#0d9488");

      const lab = el("text", {
        x: pad.l - 10, y: y + 11, fill: "#334155", "font-size": 10.5, "text-anchor": "end",
        "font-family": "ui-monospace,monospace",
      });
      lab.textContent = it.label.length > 26 ? it.label.slice(0, 25) + "…" : it.label;
      f.svg.appendChild(lab);

      const r = el("rect", { x, y, width: 0, height: rh - 8, rx: 3, fill: color, opacity: .92 });
      f.svg.appendChild(r);
      requestAnimationFrame(() => {
        r.style.transition = `width .75s var(--ease-out) ${i * 0.035}s`;
        r.setAttribute("width", Math.max(w, 1.5));
      });

      const vt = el("text", {
        x: (it.value < 0 && o.diverging ? x - 6 : x + Math.max(w, 1.5) + 6),
        y: y + 11, fill: "#475569", "font-size": 10,
        "text-anchor": it.value < 0 && o.diverging ? "end" : "start",
        "font-family": "ui-monospace,monospace",
      });
      vt.textContent = o.fmt(it.value);
      vt.style.opacity = 0;
      vt.style.animation = `fadeUp .4s var(--ease-out) ${0.35 + i * 0.035}s forwards`;
      f.svg.appendChild(vt);
    });

    container.appendChild(f.svg);
  }

  /** Per-seed dot plot with a mean band. */
  function dots(container, opts) {
    container.innerHTML = "";
    const pad = { l: 44, r: 16, t: 14, b: 28 };
    const f = frame(620, 170, pad);
    const vals = opts.values;
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1 || 1));
    const lo = Math.min(...vals) - sd * 1.6, hi = Math.max(...vals) + sd * 1.6;
    const sy = v => pad.t + f.ih - ((v - lo) / (hi - lo || 1)) * f.ih;
    const sx = i => pad.l + ((i + .5) / vals.length) * f.iw;

    grid(f, [0, .5, 1].map(p => ({ p, v: lo + p * (hi - lo) })), v => (v * 100).toFixed(1) + "%");

    f.svg.appendChild(el("rect", {
      x: pad.l, y: sy(mean + sd), width: f.iw, height: Math.abs(sy(mean - sd) - sy(mean + sd)),
      fill: "#0d9488", opacity: .09, rx: 4,
    }));
    f.svg.appendChild(el("line", {
      x1: pad.l, x2: pad.l + f.iw, y1: sy(mean), y2: sy(mean),
      stroke: "#0d9488", "stroke-width": 1.6, "stroke-dasharray": "6 4",
    }));

    vals.forEach((v, i) => {
      const c = el("circle", { cx: sx(i), cy: sy(v), r: 0, fill: "#0284c7", stroke: "#ffffff", "stroke-width": 2 });
      f.svg.appendChild(c);
      requestAnimationFrame(() => {
        c.style.transition = `r .5s var(--ease-out) ${i * .08}s`;
        c.setAttribute("r", 6);
      });
      const t = el("text", {
        x: sx(i), y: pad.t + f.ih + 17, fill: "#64748b", "font-size": 10,
        "text-anchor": "middle", "font-family": "ui-monospace,monospace",
      });
      t.textContent = "s" + (opts.labels ? opts.labels[i] : i);
      f.svg.appendChild(t);
    });

    const mt = el("text", { x: pad.l + f.iw - 2, y: sy(mean) - 7, fill: "#0d9488", "font-size": 10, "text-anchor": "end", "font-weight": 600 });
    mt.textContent = `mean ${(mean * 100).toFixed(2)}% ± ${(sd * 100).toFixed(2)}%`;
    f.svg.appendChild(mt);

    container.appendChild(f.svg);
  }

  /** Animated number count-up. */
  function countUp(node, target, dec = 2, suffix = "") {
    const dur = 900, t0 = performance.now();
    const step = now => {
      const p = Math.min((now - t0) / dur, 1);
      const e = 1 - Math.pow(1 - p, 3);
      node.textContent = (target * e).toFixed(dec) + suffix;
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  // keyframes used by the paths above
  const sheet = document.createElement("style");
  sheet.textContent = `
    @keyframes draw { to { stroke-dashoffset: 0; } }
    @keyframes haloPulse { 0%,100% { r:6; opacity:.30 } 50% { r:11; opacity:.08 } }`;
  document.head.appendChild(sheet);

  global.Charts = { line, logLine, hbars, dots, countUp };
})(window);
