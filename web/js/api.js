/* Thin client for the local API (CLAUDE.md §7).
 *
 * Every call is same-origin and relative. There is no base URL to configure and no CDN to
 * fall back on: the page is served by the same process that answers these routes, which is
 * what makes the whole thing work with the network off.
 *
 * Errors carry the server's own message. §7 requires empty and error states to say what
 * happened and what to do, and the API already phrases its detail strings that way ("no
 * signals found above 12.0 dB over the noise floor. Try lowering the detection
 * threshold."), so the UI shows them verbatim rather than replacing them with something
 * vaguer.
 */
(function (global) {
  "use strict";

  async function readError(response) {
    let detail = response.statusText || ("HTTP " + response.status);
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
    } catch (_) { /* a non-JSON error body is fine; keep the status text */ }
    const error = new Error(detail);
    error.status = response.status;
    return error;
  }

  async function get(path) {
    const response = await fetch(path, { headers: { Accept: "application/json" } });
    if (!response.ok) throw await readError(response);
    return response.json();
  }

  const Api = {
    health: () => get("/api/health"),

    /* POST /api/analyse -- multipart upload plus optional fs / fc (§7).
       Uses XMLHttpRequest rather than fetch purely for upload progress, which fetch still
       cannot report; a 2 GB capture needs a moving bar or the page looks hung. */
    analyse(file, options, onUploadProgress) {
      const form = new FormData();
      form.append("file", file, file.name);
      ["fs", "fc", "dtype", "threshold_db"].forEach(function (key) {
        const value = options && options[key];
        if (value !== undefined && value !== null && value !== "") form.append(key, value);
      });

      return new Promise(function (resolve, reject) {
        const request = new XMLHttpRequest();
        request.open("POST", "/api/analyse");
        request.upload.onprogress = function (event) {
          if (onUploadProgress && event.lengthComputable) {
            onUploadProgress(event.loaded / event.total);
          }
        };
        request.onload = function () {
          let body = {};
          try { body = JSON.parse(request.responseText); } catch (_) { /* keep {} */ }
          if (request.status >= 200 && request.status < 300) {
            resolve(body);
          } else {
            const error = new Error(body.detail || ("HTTP " + request.status));
            error.status = request.status;
            reject(error);
          }
        };
        request.onerror = function () {
          reject(new Error("could not reach the local API. Is `sigscope serve` running?"));
        };
        request.send(form);
      });
    },

    job: (id) => get("/api/jobs/" + id),
    jobs: () => get("/api/jobs"),
    report: (id) => get("/api/jobs/" + id + "/report"),
    spectrogram: (id) => get("/api/jobs/" + id + "/spectrogram.json"),
    detectionIq: (id, n) => get("/api/jobs/" + id + "/detections/" + n + "/iq.json"),

    batch: async function (path, pattern) {
      const response = await fetch("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: path, pattern: pattern || "*" })
      });
      if (!response.ok) throw await readError(response);
      return response.json();
    },

    /* Poll until the job leaves queued/running. §7 defines progress as 0-1, so the
       callback gets the whole job object and the caller decides what to show. */
    poll: function (id, onUpdate, intervalMs) {
      const every = intervalMs || 400;
      return new Promise(function (resolve, reject) {
        (function tick() {
          Api.job(id).then(function (job) {
            if (onUpdate) onUpdate(job);
            if (job.state === "done") return resolve(job);
            if (job.state === "failed") {
              const error = new Error(job.error || "the job failed");
              error.job = job;
              return reject(error);
            }
            setTimeout(tick, every);
          }).catch(reject);
        })();
      });
    },

    urls: {
      spectrogramPng: (id) => "/api/jobs/" + id + "/spectrogram.png",
      sigmf: (id) => "/api/jobs/" + id + "/sigmf",
      report: (id) => "/api/jobs/" + id + "/report",
      csv: (id) => "/api/jobs/" + id + "/csv",
      audio: (id, n) => "/api/jobs/" + id + "/detections/" + n + "/audio.wav"
    }
  };

  global.Api = Api;
})(window);
