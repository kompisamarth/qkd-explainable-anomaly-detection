/* Animated BB84 quantum channel: Alice -> (Eve) -> Bob, on a canvas. */
(function (global) {
  "use strict";

  const C = {
    photon: "#22d3ee", intercepted: "#f43f5e", sifted: "#34d399",
    lost: "#64748b", node: "#121b30", line: "#1e2a47", text: "#8b9ab5",
    eve: "#f43f5e", violet: "#a78bfa",
  };

  function create(canvas) {
    const ctx = canvas.getContext("2d");
    let W = 0, H = 0, dpr = 1;
    let photons = [], flashes = [], raf = null, last = 0, acc = 0, t = 0;

    const state = {
      attackActive: false, attackClass: 0, intercept: 0,
      qber: 0.03, efficiency: 0.5, running: true, speed: 1,
    };

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = canvas.clientWidth; H = canvas.clientHeight || 190;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    const ax = () => 72, bx = () => W - 72, ex = () => W / 2, by = () => H / 2;

    function spawn() {
      const basisA = Math.random() < .5 ? 0 : 1;
      const bit = Math.random() < .5 ? 0 : 1;
      const basisB = Math.random() < .5 ? 0 : 1;
      // polarization angle: Z basis -> 0/90, X basis -> 45/135
      const ang = basisA === 0 ? (bit ? 90 : 0) : (bit ? 135 : 45);
      photons.push({
        x: ax(), y: by() + (Math.random() - .5) * 9, ang, basisA, basisB, bit,
        v: 0.9 + Math.random() * 0.5, state: "fly", alpha: 1, r: 2.6 + Math.random() * 1.1,
        willIntercept: state.attackActive && Math.random() < state.intercept,
        willSurvive: Math.random() < Math.max(state.efficiency, .12),
        spin: (Math.random() - .5) * 1.4,
      });
    }

    function flash(x, y, color, size) { flashes.push({ x, y, color, r: 1, max: size || 22, a: 1 }); }

    function node(x, y, label, color, active, pulse) {
      ctx.save();
      if (active) { ctx.shadowColor = color; ctx.shadowBlur = 18 + Math.sin(t * 3) * 7; }
      ctx.fillStyle = C.node;
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2 : 1.4;
      ctx.beginPath();
      ctx.roundRect(x - 26, y - 26, 52, 52, 13);
      ctx.fill(); ctx.stroke();
      ctx.restore();

      if (pulse && active) {
        const p = (t * .8) % 1;
        ctx.save();
        ctx.globalAlpha = (1 - p) * .5;
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.roundRect(x - 26 - p * 16, y - 26 - p * 16, 52 + p * 32, 52 + p * 32, 13 + p * 8);
        ctx.stroke(); ctx.restore();
      }

      ctx.fillStyle = active ? color : C.text;
      ctx.font = "600 10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText(label, x, y + 44);
    }

    function polarizationGlyph(x, y, ang, color, size) {
      const r = (ang * Math.PI) / 180;
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.globalAlpha = .85;
      ctx.beginPath();
      ctx.moveTo(x - Math.cos(r) * size, y + Math.sin(r) * size);
      ctx.lineTo(x + Math.cos(r) * size, y - Math.sin(r) * size);
      ctx.stroke(); ctx.restore();
    }

    function draw(dt) {
      ctx.clearRect(0, 0, W, H);
      const y = by(), A = ax(), B = bx(), E = ex();

      // fiber beam with flowing dashes
      ctx.save();
      ctx.strokeStyle = C.line; ctx.lineWidth = 16; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(A, y); ctx.lineTo(B, y); ctx.stroke();
      const g = ctx.createLinearGradient(A, 0, B, 0);
      g.addColorStop(0, "rgba(34,211,238,.20)");
      g.addColorStop(.5, "rgba(167,139,250,.14)");
      g.addColorStop(1, "rgba(34,211,238,.20)");
      ctx.strokeStyle = g; ctx.lineWidth = 14;
      ctx.beginPath(); ctx.moveTo(A, y); ctx.lineTo(B, y); ctx.stroke();
      ctx.strokeStyle = "rgba(34,211,238,.30)"; ctx.lineWidth = 1;
      ctx.setLineDash([2, 14]); ctx.lineDashOffset = -t * 42;
      ctx.beginPath(); ctx.moveTo(A, y); ctx.lineTo(B, y); ctx.stroke();
      ctx.restore();

      // photons
      for (const p of photons) {
        if (state.running) {
          p.x += p.v * dt * 0.075 * state.speed;
          p.ang += p.spin;
        }
        if (p.state === "fly") {
          if (state.attackActive && p.willIntercept && p.x >= E) {
            p.state = "eve"; p.ang = Math.random() * 180;
            flash(E, y, C.eve, 26);
          } else if (!p.willSurvive && p.x >= A + (B - A) * 0.55) {
            p.state = "lost";
          }
        }
        if (p.state === "lost") p.alpha -= dt * 0.004;
        if (p.x >= B - 4 && p.state !== "done" && p.state !== "lost") {
          p.state = "done";
          const kept = p.basisA === p.basisB;
          flash(B, y, kept ? C.sifted : C.lost, kept ? 24 : 14);
          p.alpha = 0;
        }
        if (p.alpha <= 0 || p.x > B + 20) continue;

        const col = p.state === "eve" ? C.intercepted : (p.state === "lost" ? C.lost : C.photon);
        ctx.save();
        ctx.globalAlpha = Math.max(p.alpha, 0);
        ctx.shadowColor = col; ctx.shadowBlur = 13;
        ctx.fillStyle = col;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        ctx.restore();
        polarizationGlyph(p.x, p.y, p.ang, col, 7);
      }
      photons = photons.filter(p => p.alpha > 0 && p.x <= B + 20);

      // flashes
      for (const fl of flashes) {
        fl.r += dt * 0.055; fl.a -= dt * 0.0026;
        if (fl.a <= 0) continue;
        ctx.save();
        ctx.globalAlpha = Math.max(fl.a, 0);
        ctx.strokeStyle = fl.color; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(fl.x, fl.y, Math.min(fl.r, fl.max), 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
      }
      flashes = flashes.filter(f => f.a > 0);

      // nodes
      node(A, y, "ALICE", C.photon, true, false);
      node(B, y, "BOB", C.violet, true, false);
      if (state.attackActive) node(E, y, "EVE", C.eve, true, true);

      // node glyphs
      ctx.save();
      ctx.strokeStyle = C.photon; ctx.lineWidth = 1.6; ctx.globalAlpha = .9;
      ctx.beginPath(); ctx.arc(A, y, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(A - 5, y); ctx.lineTo(A + 5, y); ctx.stroke();
      ctx.strokeStyle = C.violet;
      ctx.beginPath(); ctx.moveTo(B - 7, y - 7); ctx.lineTo(B + 7, y + 7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(B + 7, y - 7); ctx.lineTo(B - 7, y + 7); ctx.stroke();
      if (state.attackActive) {
        ctx.strokeStyle = C.eve;
        ctx.beginPath(); ctx.arc(E, y, 8, 0, Math.PI * 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(E - 4, y - 4); ctx.lineTo(E + 4, y + 4); ctx.stroke();
      }
      ctx.restore();

      // status strip
      ctx.fillStyle = C.text;
      ctx.font = "600 10px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.fillText(
        state.attackActive
          ? `EVE ACTIVE · intercepting ${(state.intercept * 100).toFixed(0)}% of pulses`
          : "CHANNEL CLEAR · no eavesdropper",
        14, 20
      );
      ctx.textAlign = "right";
      ctx.fillText(`QBER ${(state.qber * 100).toFixed(2)}%  ·  yield ${(state.efficiency * 100).toFixed(0)}%`, W - 14, 20);
    }

    function loop(now) {
      const dt = Math.min(now - last, 50); last = now;
      if (state.running) {
        t += dt / 1000;
        acc += dt;
        const every = 85;
        while (acc > every) { spawn(); acc -= every; }
      }
      draw(dt);
      raf = requestAnimationFrame(loop);
    }

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    resize();
    last = performance.now();
    raf = requestAnimationFrame(loop);

    return {
      set(patch) { Object.assign(state, patch); },
      pause() { state.running = false; },
      resume() { state.running = true; },
      destroy() { cancelAnimationFrame(raf); ro.disconnect(); },
    };
  }

  global.Photons = { create };
})(window);
