/* Animated BB84 quantum channel: Alice -> (Eve) -> Bob, on a canvas (Light Mode). */
(function (global) {
  "use strict";

  const C = {
    bg: "#ffffff",
    photon: "#0284c7",
    intercepted: "#dc2626",
    sifted: "#059669",
    lost: "#94a3b8",
    nodeBg: "#ffffff",
    nodeBorder: "#cbd5e1",
    line: "#e2e8f0",
    text: "#475569",
    alice: "#0d9488",
    bob: "#0284c7",
    eve: "#dc2626",
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
        x: ax(), y: by() + (Math.random() - .5) * 8, ang, basisA, basisB, bit,
        v: 0.9 + Math.random() * 0.5, state: "fly", alpha: 1, r: 2.6 + Math.random() * 1.1,
        willIntercept: state.attackActive && Math.random() < state.intercept,
        willSurvive: Math.random() < Math.max(state.efficiency, .12),
        spin: (Math.random() - .5) * 1.4,
      });
    }

    function flash(x, y, color, size) { flashes.push({ x, y, color, r: 1, max: size || 22, a: 1 }); }

    function node(x, y, label, color, active, pulse) {
      ctx.save();
      // Drop subtle card shadow
      ctx.shadowColor = "rgba(15, 23, 42, 0.08)";
      ctx.shadowBlur = 6;
      ctx.shadowOffsetY = 2;
      ctx.fillStyle = C.nodeBg;
      ctx.strokeStyle = color;
      ctx.lineWidth = active ? 2 : 1.4;
      ctx.beginPath();
      ctx.roundRect(x - 26, y - 26, 52, 52, 12);
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      if (pulse && active) {
        const p = (t * .8) % 1;
        ctx.save();
        ctx.globalAlpha = (1 - p) * .45;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.roundRect(x - 26 - p * 12, y - 26 - p * 12, 52 + p * 24, 52 + p * 24, 12 + p * 6);
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = active ? color : C.text;
      ctx.font = "700 11px var(--font-sans), 'Plus Jakarta Sans', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(label, x, y + 43);
    }

    function polarizationGlyph(x, y, ang, color, size) {
      const r = (ang * Math.PI) / 180;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = .9;
      ctx.beginPath();
      ctx.moveTo(x - Math.cos(r) * size, y + Math.sin(r) * size);
      ctx.lineTo(x + Math.cos(r) * size, y - Math.sin(r) * size);
      ctx.stroke();
      ctx.restore();
    }

    function draw(dt) {
      // Clear white canvas
      ctx.clearRect(0, 0, W, H);
      const y = by(), A = ax(), B = bx(), E = ex();

      // Optical fiber core waveguide
      ctx.save();
      // Outer fiber track
      ctx.strokeStyle = "#f1f5f9";
      ctx.lineWidth = 18;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(A, y);
      ctx.lineTo(B, y);
      ctx.stroke();

      // Inner fiber channel
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 12;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(A, y);
      ctx.lineTo(B, y);
      ctx.stroke();

      // Beam optical tint gradient
      const g = ctx.createLinearGradient(A, 0, B, 0);
      g.addColorStop(0, "rgba(13, 148, 136, 0.18)");
      g.addColorStop(0.5, "rgba(2, 132, 199, 0.14)");
      g.addColorStop(1, "rgba(13, 148, 136, 0.18)");
      ctx.strokeStyle = g;
      ctx.lineWidth = 10;
      ctx.beginPath();
      ctx.moveTo(A, y);
      ctx.lineTo(B, y);
      ctx.stroke();

      // Dashed photon trajectory guide
      ctx.strokeStyle = "rgba(13, 148, 136, 0.40)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 12]);
      ctx.lineDashOffset = -t * 40;
      ctx.beginPath();
      ctx.moveTo(A, y);
      ctx.lineTo(B, y);
      ctx.stroke();
      ctx.restore();

      // Photons
      for (const p of photons) {
        if (state.running) {
          p.x += p.v * dt * 0.075 * state.speed;
          p.ang += p.spin;
        }
        if (p.state === "fly") {
          if (state.attackActive && p.willIntercept && p.x >= E) {
            p.state = "eve";
            p.ang = Math.random() * 180;
            flash(E, y, C.eve, 24);
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
        ctx.fillStyle = col;
        ctx.shadowColor = col;
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        polarizationGlyph(p.x, p.y, p.ang, col, 7);
      }
      photons = photons.filter(p => p.alpha > 0 && p.x <= B + 20);

      // Flashes
      for (const fl of flashes) {
        fl.r += dt * 0.055;
        fl.a -= dt * 0.0028;
        if (fl.a <= 0) continue;
        ctx.save();
        ctx.globalAlpha = Math.max(fl.a, 0);
        ctx.strokeStyle = fl.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(fl.x, fl.y, Math.min(fl.r, fl.max), 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }
      flashes = flashes.filter(f => f.a > 0);

      // Nodes
      node(A, y, "ALICE", C.alice, true, false);
      node(B, y, "BOB", C.bob, true, false);
      if (state.attackActive) node(E, y, "EVE", C.eve, true, true);

      // Node internal glyphs
      ctx.save();
      // Alice: Z-basis circle + bar
      ctx.strokeStyle = C.alice;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(A, y, 9, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(A - 5, y);
      ctx.lineTo(A + 5, y);
      ctx.stroke();

      // Bob: X-basis cross
      ctx.strokeStyle = C.bob;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(B - 7, y - 7);
      ctx.lineTo(B + 7, y + 7);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(B + 7, y - 7);
      ctx.lineTo(B - 7, y + 7);
      ctx.stroke();

      // Eve: Forbidden state icon
      if (state.attackActive) {
        ctx.strokeStyle = C.eve;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(E, y, 9, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(E - 5, y - 5);
        ctx.lineTo(E + 5, y + 5);
        ctx.stroke();
      }
      ctx.restore();

      // Top telemetry status strip
      ctx.font = "600 10.5px ui-monospace, monospace";
      ctx.textAlign = "left";
      if (state.attackActive) {
        ctx.fillStyle = C.eve;
        ctx.fillText(`EVE ACTIVE · intercepting ${(state.intercept * 100).toFixed(0)}% of pulses`, 14, 20);
      } else {
        ctx.fillStyle = C.sifted;
        ctx.fillText("CHANNEL CLEAR · no eavesdropper", 14, 20);
      }

      ctx.fillStyle = C.text;
      ctx.textAlign = "right";
      ctx.fillText(`QBER ${(state.qber * 100).toFixed(2)}%  ·  yield ${(state.efficiency * 100).toFixed(0)}%`, W - 14, 20);
    }

    function loop(now) {
      const dt = Math.min(now - last, 50);
      last = now;
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
