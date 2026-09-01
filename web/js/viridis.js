/* Viridis colormap (CLAUDE.md §7).
 *
 * §7 is specific and gives the reason: "Perceptually uniform, standard in the field, and
 * colourblind-safe. Not jet -- jet invents features that are not there and any RF engineer
 * will notice." So this is the only colormap in the dashboard.
 *
 * 32 anchors sampled from matplotlib's viridis, interpolated to 256 levels at load. Same
 * anchors as sigscope/report/png.py, so the PNG the server renders and the canvas the
 * browser draws are the same picture in the same colours.
 */
(function (global) {
  "use strict";

  const ANCHORS = [
    [68,1,84],[71,13,96],[72,24,106],[72,35,116],[71,46,124],[69,56,130],
    [66,65,134],[62,74,137],[58,84,140],[54,93,141],[50,101,142],[46,109,142],
    [43,117,142],[40,125,142],[37,132,142],[34,140,141],[31,148,140],[30,156,137],
    [32,163,134],[37,171,130],[46,179,124],[58,186,118],[72,193,110],[88,199,101],
    [108,205,90],[127,211,78],[147,215,65],[168,219,52],[192,223,37],[213,226,26],
    [234,229,26],[253,231,37]
  ];

  const LUT = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const x = (i / 255) * (ANCHORS.length - 1);
    const lo = Math.floor(x);
    const hi = Math.min(lo + 1, ANCHORS.length - 1);
    const t = x - lo;
    for (let c = 0; c < 3; c++) {
      LUT[i * 3 + c] = Math.round(ANCHORS[lo][c] * (1 - t) + ANCHORS[hi][c] * t);
    }
  }

  global.Viridis = {
    lut: LUT,
    /* Map a 2-D array of dB values to ImageData. Row 0 of `z` is the lowest frequency,
       so the image is flipped vertically: §7 wants frequency up. */
    toImageData: function (z, vmin, vmax) {
      const rows = z.length;
      const cols = rows ? z[0].length : 0;
      const image = new ImageData(cols, rows);
      const data = image.data;
      const span = (vmax - vmin) || 1;
      for (let r = 0; r < rows; r++) {
        const src = z[rows - 1 - r];           // flip: frequency increases upward
        const rowOffset = r * cols * 4;
        for (let c = 0; c < cols; c++) {
          let v = (src[c] - vmin) / span;
          v = v < 0 ? 0 : (v > 1 ? 1 : v);
          const idx = (v * 255) | 0;
          const o = rowOffset + c * 4;
          data[o] = LUT[idx * 3];
          data[o + 1] = LUT[idx * 3 + 1];
          data[o + 2] = LUT[idx * 3 + 2];
          data[o + 3] = 255;
        }
      }
      return image;
    }
  };
})(window);
