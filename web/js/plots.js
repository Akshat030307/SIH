/* The three small plots at the foot of the analysis view (CLAUDE.md §7).
 *
 * §7's layout ends with `[ constellation ] [ spectrum ] [ inst. frequency ]`, and asks for
 * Plotly vendored as a local file. Plotly is not in this checkout and cannot be fetched
 * offline (see web/vendor/README.md), so these are drawn on a plain 2-D canvas instead.
 *
 * For what these three plots actually are that is not a compromise: each is a single
 * series with no interaction beyond looking at it, and canvas gives exact control over
 * §7's palette and hairline weights, costs nothing to load, and cannot pull a font or a
 * script from anywhere. The spectrogram -- the one genuinely interactive element -- gets
 * its own click handling in app.js.
 */
(function (global) {
  "use strict";

  const COLOUR = {
    axis: "#2d333b",
    text: "#8b949e",
    trace: "#c9d1d9",
    accent: "#4dd0c4",
    bg: "#0d1117"
  };
  const PAD = { left: 52, right: 12, top: 12, bottom: 26 };

  function prepare(canvas) {
    /* Match the backing store to the CSS size so lines are crisp on any DPI. */
    const ratio = global.devicePixelRatio || 1;
    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 240;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = COLOUR.bg;
    ctx.fillRect(0, 0, width, height);
    return { ctx: ctx, w: width, h: height };
  }

  function axes(ctx, w, h, xLabel, yLabel, xLo, xHi, yLo, yHi) {
    const x0 = PAD.left, x1 = w - PAD.right, y0 = PAD.top, y1 = h - PAD.bottom;
    ctx.strokeStyle = COLOUR.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.lineTo(x1, y1);
    ctx.stroke();

    ctx.fillStyle = COLOUR.text;
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(fmt(yHi), x0 - 6, y0 + 4);
    ctx.fillText(fmt(yLo), x0 - 6, y1 - 4);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(fmt(xLo), x0, y1 + 6);
    ctx.textAlign = "right";
    ctx.fillText(fmt(xHi), x1, y1 + 6);
    ctx.textAlign = "left";
    ctx.fillText(xLabel, x0, y1 + 6 + 12);
    ctx.save();
    ctx.translate(12, y0);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "right";
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
    return { x0: x0, x1: x1, y0: y0, y1: y1 };
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return (v / 1e3).toFixed(1) + "k";
    if (a >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }

  function extent(values) {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    return [lo, hi];
  }

  const Plots = {
    /* I against Q. Four tight clusters is QPSK; a ring is constant-modulus; a smear is
       noise or a burst that was never a single signal. */
    constellation: function (canvas, data) {
      const p = prepare(canvas);
      const lim = Math.max(
        Math.abs(extent(data.i)[0]), Math.abs(extent(data.i)[1]),
        Math.abs(extent(data.q)[0]), Math.abs(extent(data.q)[1])
      ) || 1;
      const box = axes(p.ctx, p.w, p.h, "I", "Q", -lim, lim, -lim, lim);
      const sx = (box.x1 - box.x0) / (2 * lim);
      const sy = (box.y1 - box.y0) / (2 * lim);

      p.ctx.strokeStyle = COLOUR.axis;
      p.ctx.beginPath();
      p.ctx.moveTo(box.x0, (box.y0 + box.y1) / 2); p.ctx.lineTo(box.x1, (box.y0 + box.y1) / 2);
      p.ctx.moveTo((box.x0 + box.x1) / 2, box.y0); p.ctx.lineTo((box.x0 + box.x1) / 2, box.y1);
      p.ctx.stroke();

      p.ctx.fillStyle = COLOUR.accent;
      p.ctx.globalAlpha = 0.55;
      for (let n = 0; n < data.i.length; n++) {
        const x = box.x0 + (data.i[n] + lim) * sx;
        const y = box.y1 - (data.q[n] + lim) * sy;
        p.ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
      }
      p.ctx.globalAlpha = 1;
      return data.i.length + " symbols, power-normalised";
    },

    spectrum: function (canvas, data) {
      const p = prepare(canvas);
      const [yLo, yHi] = extent(data.spectrum_db);
      const xLo = data.spectrum_hz[0];
      const xHi = data.spectrum_hz[data.spectrum_hz.length - 1];
      const box = axes(p.ctx, p.w, p.h, "Hz from burst centre", "dB", xLo, xHi, yLo, yHi);
      const sx = (box.x1 - box.x0) / (xHi - xLo || 1);
      const sy = (box.y1 - box.y0) / (yHi - yLo || 1);

      p.ctx.strokeStyle = COLOUR.trace;
      p.ctx.lineWidth = 1;
      p.ctx.beginPath();
      for (let n = 0; n < data.spectrum_db.length; n++) {
        const x = box.x0 + (data.spectrum_hz[n] - xLo) * sx;
        const y = box.y1 - (data.spectrum_db[n] - yLo) * sy;
        n === 0 ? p.ctx.moveTo(x, y) : p.ctx.lineTo(x, y);
      }
      p.ctx.stroke();
      return "welch-free periodogram of the isolated burst at " +
             fmt(data.fs_b) + " Hz";
    },

    finst: function (canvas, data) {
      const p = prepare(canvas);
      const [yLo, yHi] = extent(data.f_inst);
      const xHi = data.f_inst.length;
      const box = axes(p.ctx, p.w, p.h, "sample", "Hz", 0, xHi, yLo, yHi);
      const sx = (box.x1 - box.x0) / (xHi || 1);
      const sy = (box.y1 - box.y0) / (yHi - yLo || 1);

      p.ctx.strokeStyle = COLOUR.trace;
      p.ctx.lineWidth = 1;
      p.ctx.beginPath();
      for (let n = 0; n < data.f_inst.length; n++) {
        const x = box.x0 + n * sx;
        const y = box.y1 - (data.f_inst[n] - yLo) * sy;
        n === 0 ? p.ctx.moveTo(x, y) : p.ctx.lineTo(x, y);
      }
      p.ctx.stroke();
      return "instantaneous frequency; flat levels mean FSK tones";
    }
  };

  global.Plots = Plots;
})(window);
