"use strict";

// --- plumbing ---------------------------------------------------------------

async function api(route, body) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(route, options);
  const payload = await response.json();
  if (payload && payload.error) throw new Error(payload.error);
  return payload;
}

// The file itself, not a JSON envelope: base64 would inflate a measurement by
// a third and buy nothing.
async function uploadRequest(route, file) {
  let response;
  try {
    response = await fetch(route, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
  } catch (_error) {
    throw new Error(
      "HamLeT could not reach its local server while uploading. Check that the " +
      "terminal running ‘hamlet gui’ is still open, then reload this page.",
    );
  }
  let payload;
  try {
    payload = await response.json();
  } catch (_error) {
    throw new Error(`The upload server returned an unreadable response (HTTP ${response.status}).`);
  }
  if (!response.ok || (payload && payload.error)) {
    throw new Error(payload?.error || `Upload failed with HTTP ${response.status}.`);
  }
  return payload;
}

async function upload(file) {
  return uploadRequest(`/api/upload?name=${encodeURIComponent(file.name)}`, file);
}

async function uploadFolder(files, onProgress) {
  const accepted = [...files].filter((file) => /\.(dat|txt)$/i.test(file.name));
  if (!accepted.length) {
    throw new Error("That folder contains no .dat or .txt STS spectra.");
  }
  const firstPath = accepted[0].webkitRelativePath || accepted[0].name;
  const folderName = firstPath.split("/")[0] || "experiment";
  const session = `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
  let cursor = 0;
  let completed = 0;
  let folderPath = null;

  async function worker() {
    while (cursor < accepted.length) {
      const file = accepted[cursor];
      cursor += 1;
      const route = "/api/upload-folder" +
        `?session=${encodeURIComponent(session)}` +
        `&folder=${encodeURIComponent(folderName)}` +
        `&name=${encodeURIComponent(file.name)}`;
      const stored = await uploadRequest(route, file);
      folderPath = stored.folder_path;
      completed += 1;
      onProgress(completed, accepted.length, file.name);
    }
  }
  await Promise.all(Array.from({ length: Math.min(4, accepted.length) }, worker));
  return { path: folderPath, count: accepted.length, folderName };
}

const el = (id) => document.getElementById(id);
const esc = (text) =>
  String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function showError(node, error) {
  node.innerHTML = `<div class="error"><b>Error</b><br>${esc(error.message)}</div>`;
}

// `extra` is trusted markup the caller appends under the message -- the
// waiting quote, in practice. The message itself is still escaped.
function busy(node, message, extra = "") {
  node.innerHTML = `<div class="box">${esc(message)}${extra}</div>`;
}

function num(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

function fileLink(path, label) {
  return `<a href="/api/file?path=${encodeURIComponent(path)}" target="_blank"
    rel="noopener">${esc(label)}</a>`;
}

function humanSize(bytes) {
  if (bytes === null || bytes === undefined) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} kB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// --- tabs -------------------------------------------------------------------

function activate(name) {
  document.querySelectorAll("#tabs button").forEach((b) => {
    const current = b.dataset.panel === name;
    b.classList.toggle("active", current);
    if (current) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `panel-${name}`));
}
document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => activate(b.dataset.panel)));

// --- start ------------------------------------------------------------------

const SITUATION_PANEL = {
  inspect: "data", advise: "reuse", models: "models", analyse: "analyse",
  train: "train", dmi: "dmi",
};
const SITUATION_TITLE = {
  inspect: "Inspect a measurement", advise: "Find a suitable model",
  models: "Explore existing models", analyse: "Get my couplings",
  train: "Train for my system", dmi: "Design a DMI experiment",
};

api("/api/overview").then(({ situations }) => {
  el("situations").innerHTML = situations.map((s) => `
    <button type="button" class="card situation-card" data-go="${esc(SITUATION_PANEL[s.id] || "start")}">
      <span class="have">${esc(SITUATION_TITLE[s.id] || s.have)}<span class="card-arrow" aria-hidden="true">↗</span></span>
      <span class="does">${esc(s.does)}</span>
      <span class="meta"><span><b>Needs</b> ${esc(s.needs)}</span><span class="card-time"><b>Time</b> ${esc(s.cost)}</span></span>
    </button>`).join("");
  document.querySelectorAll("#situations .card").forEach((card) =>
    card.addEventListener("click", () => activate(card.dataset.go)));
}).catch((e) => showError(el("situations"), e));

// --- choosing a file --------------------------------------------------------
// Three ways in, because the file can be in two places. Drag-and-drop and the
// file picker send it from the machine showing this page; Browse picks one
// that is already on the machine running the server, which is what you need
// over an SSH tunnel. The chosen path is kept in a hidden text input so the
// rest of the page reads it the same way regardless of how it got there.

const fileFields = [];

// A file dropped anywhere but a drop zone would otherwise navigate the page to
// it, losing whatever was on screen. Missing the target is easy; the cost of
// missing it should not be starting over.
["dragover", "drop"].forEach((name) =>
  document.addEventListener(name, (event) => {
    if (!event.target.closest || !event.target.closest(".filefield")) {
      event.preventDefault();
    }
  }));

function attachFileField({ prefix, onPicked }) {
  const drop = el(`${prefix}-drop`);
  const path = el(`${prefix}-path`);
  const chosen = el(`${prefix}-chosen`);
  const picker = el(`${prefix}-upload`);
  const folderPicker = el(`${prefix}-folder-upload`);
  const field = { prefix, autofilled: false };

  function announce(value, detail, { autofilled = false } = {}) {
    path.value = value;
    field.autofilled = autofilled;
    chosen.hidden = false;
    chosen.innerHTML = `<b>Selected:</b> <code>${esc(value)}</code>${detail ? ` <span class="hint">${esc(detail)}</span>` : ""}`;
    if (onPicked) onPicked(value);
  }

  async function send(file) {
    chosen.hidden = false;
    chosen.textContent = `Uploading ${file.name}…`;
    try {
      const stored = await upload(file);
      announce(stored.path, `${humanSize(stored.size_bytes)}; copied to the workspace`);
    } catch (e) {
      chosen.innerHTML = `<span class="failtext">${esc(e.message)}</span>`;
    }
  }

  async function sendFolder(files) {
    chosen.hidden = false;
    chosen.textContent = "Preparing the folder…";
    try {
      const stored = await uploadFolder(files, (done, total, name) => {
        chosen.textContent = `Uploading spectrum ${done} of ${total}: ${name}`;
      });
      announce(
        stored.path,
        `${stored.count} per-site STS files copied to the workspace`,
      );
    } catch (e) {
      chosen.innerHTML = `<span class="failtext">${esc(e.message)}</span>`;
    } finally {
      folderPicker.value = "";
    }
  }

  drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("dragging");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("dragging"));
  drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("dragging");
    const file = event.dataTransfer.files[0];
    if (file) send(file);
  });
  el(`${prefix}-choose`).addEventListener("click", () => picker.click());
  picker.addEventListener("change", () => {
    if (picker.files[0]) send(picker.files[0]);
  });
  el(`${prefix}-choose-folder`).addEventListener("click", () => folderPicker.click());
  folderPicker.addEventListener("change", () => {
    if (folderPicker.files.length) sendFolder(folderPicker.files);
  });
  el(`${prefix}-browse`).addEventListener("click", () => openBrowser(announce));

  field.announce = announce;
  fileFields.push(field);
  return field;
}

// Announce a path everywhere, so inspecting a file leaves the other pages
// pointed at the same measurement rather than asking for it again. A field the
// user chose for themselves is left alone; one that was carried over before is
// updated, so re-inspecting a second file does not leave the first behind.
function shareChosenFile(value, except) {
  fileFields.forEach((field) => {
    if (field.prefix === except) return;
    if (el(`${field.prefix}-path`).value && !field.autofilled) return;
    field.announce(value, "selected in the previous step", { autofilled: true });
  });
}

// --- the server-side file browser -------------------------------------------

let browserTarget = null;
let browserPath = null;
let browserFolderInfo = null;

function openBrowser(onPicked) {
  browserTarget = onPicked;
  el("browser-backdrop").hidden = false;
  loadBrowser(browserPath);
}

function closeBrowser() {
  el("browser-backdrop").hidden = true;
  browserTarget = null;
}

async function loadBrowser(path) {
  const list = el("browser-list");
  list.innerHTML = `<p class="hint">Reading…</p>`;
  try {
    const query = path ? `?path=${encodeURIComponent(path)}` : "";
    const d = await api(`/api/browse${query}`);
    browserPath = d.path;
    browserFolderInfo = d;
    el("browser-path").textContent = d.path;
    el("browser-up").disabled = !d.parent;
    el("browser-shortcuts").innerHTML = d.shortcuts.map((s) =>
      `<button data-shortcut="${esc(s.path)}">${esc(s.label)}</button>`).join("");
    el("browser-shortcuts").querySelectorAll("button[data-shortcut]").forEach((b) =>
      b.addEventListener("click", () => loadBrowser(b.dataset.shortcut)));
    list.innerHTML = d.entries.length
      ? `<table>${d.entries.map((entry) => `<tr>
          <td><button class="linky" data-kind="${entry.kind}" data-path="${esc(entry.path)}">${
            entry.kind === "directory" ? "📁 " : "📄 "}${esc(entry.name)}</button></td>
          <td class="num hint">${entry.kind === "file" ? esc(humanSize(entry.size_bytes)) : ""}</td>
        </tr>`).join("")}</table>`
      : `<p class="hint">Nothing here that this workflow can read.</p>`;
    list.querySelectorAll("button[data-path]").forEach((b) =>
      b.addEventListener("click", () => {
        if (b.dataset.kind === "directory") {
          loadBrowser(b.dataset.path);
        } else if (browserTarget) {
          browserTarget(b.dataset.path, "stored on this machine");
          closeBrowser();
        }
      }));
    el("browser-note").textContent =
      `Showing directories and ${d.readable_suffixes.join(", ")} files` +
      (d.n_hidden_other_files ? `; ${d.n_hidden_other_files} other file(s) hidden.` : ".") +
      (d.truncated ? " The listing was truncated." : "");
    el("browser-select-folder").disabled = !d.selectable_as_measurement;
    el("browser-folder-summary").textContent = d.selectable_as_measurement
      ? `${d.raw_sts_file_count} raw STS file(s) found. Files will be combined in natural filename order.`
      : "Open a folder containing one .dat or .txt STS file per site to select it.";
  } catch (e) {
    browserFolderInfo = null;
    el("browser-select-folder").disabled = true;
    el("browser-folder-summary").textContent = "";
    showError(list, e);
  }
}

el("browser-close").addEventListener("click", closeBrowser);
el("browser-up").addEventListener("click", async () => {
  const d = await api(`/api/browse?path=${encodeURIComponent(browserPath)}`);
  if (d.parent) loadBrowser(d.parent);
});
el("browser-select-folder").addEventListener("click", () => {
  if (!browserTarget || !browserFolderInfo?.selectable_as_measurement) return;
  browserTarget(
    browserFolderInfo.path,
    `${browserFolderInfo.raw_sts_file_count} per-site STS files`,
  );
  closeBrowser();
});
el("browser-backdrop").addEventListener("click", (event) => {
  if (event.target === el("browser-backdrop")) closeBrowser();
});

// --- inspect data -----------------------------------------------------------

const PLOT_COLOURS = [
  "#00798c", "#d1495b", "#edae49", "#30638e",
  "#6a4c93", "#2a9d8f", "#e76f51", "#5f6f52",
];

function spectraSvg(plot, { cutoffMev = null, compact = false } = {}) {
  const bias = plot.bias_mev;
  const sites = plot.sites;
  const labels = plot.site_labels || sites.map((_, index) => index + 1);
  const w = 920, h = compact ? 230 : 300, padL = 64, padR = 18, padT = 24, padB = 48;
  const flat = sites.flat().filter(Number.isFinite);
  const lo = flat.length ? Math.min(...flat) : 0;
  const hi = flat.length ? Math.max(...flat) : 1;
  const span = hi - lo || 1;
  const biasLo = Math.min(...bias), biasHi = Math.max(...bias);
  const biasSpan = biasHi - biasLo || 1;
  const xValue = (value) => padL + ((value - biasLo) / biasSpan) * (w - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / span) * (h - padT - padB);
  const paths = sites.map((row, index) => {
    const colour = PLOT_COLOURS[index % PLOT_COLOURS.length];
    let connected = false;
    const d = row.map((v, i) => {
      if (!Number.isFinite(v)) { connected = false; return ""; }
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${xValue(bias[i]).toFixed(1)},${y(v).toFixed(1)}`;
    }).join("");
    return `<path d="${d}" fill="none" stroke="${colour}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
  }).join("");
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const value = biasLo + f * biasSpan;
    const px = xValue(value);
    return `<line class="plot-grid" x1="${px}" x2="${px}" y1="${padT}" y2="${h - padB}"/>
      <text class="plot-tick" x="${px}" y="${h - 27}" text-anchor="middle">${value.toFixed(1)}</text>`;
  }).join("");
  const yTicks = [0, 0.5, 1].map((f) => {
    const value = lo + f * span;
    const py = y(value);
    return `<line class="plot-grid" x1="${padL}" x2="${w - padR}" y1="${py}" y2="${py}"/>
      <text class="plot-tick" x="${padL - 9}" y="${py + 4}" text-anchor="end">${value.toExponential(1)}</text>`;
  }).join("");
  let cutoff = "";
  if (Number.isFinite(Number(cutoffMev)) && Number(cutoffMev) >= biasLo && Number(cutoffMev) <= biasHi) {
    const left = Math.max(biasLo, 0);
    const right = Number(cutoffMev);
    const x0 = xValue(left), xc = xValue(right);
    cutoff = `<rect class="outside-window" x="${padL}" y="${padT}" width="${Math.max(0, x0 - padL)}" height="${h - padT - padB}"/>
      <rect class="outside-window" x="${xc}" y="${padT}" width="${Math.max(0, w - padR - xc)}" height="${h - padT - padB}"/>
      <line class="cutoff-line" x1="${xc}" x2="${xc}" y1="${padT}" y2="${h - padB}"/>
      <text class="cutoff-label" x="${Math.min(xc + 7, w - 105)}" y="${padT + 14}">${right.toFixed(1)} meV cutoff</text>`;
  }
  const legend = labels.map((label, index) => `<span class="legend-item">
    <i style="background:${PLOT_COLOURS[index % PLOT_COLOURS.length]}"></i>site ${esc(label)}</span>`).join("");
  return `<div class="spectrum-figure">
    <svg class="spectra${compact ? " compact" : ""}" viewBox="0 0 ${w} ${h}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(plot.title || "signal")} spectra by site">
      ${xTicks}${yTicks}${cutoff}${paths}
      <line class="plot-axis" x1="${padL}" x2="${w - padR}" y1="${h - padB}" y2="${h - padB}"/>
      <line class="plot-axis" x1="${padL}" x2="${padL}" y1="${padT}" y2="${h - padB}"/>
      <text class="plot-axis-label" x="${(padL + w - padR) / 2}" y="${h - 7}" text-anchor="middle">Bias [meV]</text>
      <text class="plot-axis-label" transform="translate(15 ${(padT + h - padB) / 2}) rotate(-90)" text-anchor="middle">${esc(plot.y_label || "dI/dV [A]")}</text>
    </svg>
    <div class="plot-legend" aria-label="Site legend">${legend}</div>
  </div>`;
}

attachFileField({ prefix: "data" });
attachFileField({ prefix: "reuse" });
attachFileField({ prefix: "an" });

el("data-go").addEventListener("click", async () => {
  const out = el("data-out");
  const path = el("data-path").value.trim();
  if (!path) { showError(out, new Error("choose a measurement first")); return; }
  busy(out, "Reading the measurement…");
  try {
    const d = await api("/api/inspect", { path });
    const problems = [];
    if (!d.is_complete) problems.push(`${d.missing_points} data point(s) are missing; incomplete maps cannot be analysed`);
    if (!d.covers_zero) problems.push(`the bias window ${num(d.bias_min_mev, 2)} to ${num(d.bias_max_mev, 2)} meV does not include zero`);
    if (d.bias_units !== "meV") problems.push(`bias axis is in ${d.bias_units}, not meV`);
    const maximumCutoff = Math.max(1, Math.floor(d.bias_max_mev));
    const initialCutoff = Math.min(maximumCutoff, Math.max(1, Number(el("reuse-cutoff").value) || 20));
    const plots = d.plots || { [d.primary_channel]: d.plot };
    const channelNames = Object.keys(plots);
    const channelChoices = channelNames.length > 1
      ? `<div class="plot-channel-switch" role="group" aria-label="Signal to plot">${channelNames.map((name) => {
          const plot = plots[name];
          return `<button type="button" data-plot-channel="${esc(name)}" class="${name === d.primary_channel ? "active" : ""}">${esc(plot.title || name)} <small>${esc(plot.role || "")}</small></button>`;
        }).join("")}</div>`
      : "";
    const imported = d.input?.input_kind === "sts_folder";
    out.innerHTML = `
      ${imported ? `<div class="import-banner"><b>Folder imported</b><span>${d.input.source_file_count} per-site spectra combined</span><span>${d.input.auxiliary_d2idv2 ? "d²I/dV² available for plotting" : "dI/dV imported"}</span></div>` : ""}
      <div class="box">
        <table>
          <tr><th>Sites in the chain</th><td class="num"><b>${d.n_sites}</b></td></tr>
          <tr><th>Bias points</th><td class="num">${d.n_bias_points}</td></tr>
          <tr><th>Bias window</th><td class="num">${num(d.bias_min_mev, 2)} to ${num(d.bias_max_mev, 2)} ${esc(d.bias_units)}</td></tr>
          <tr><th>Channel</th><td class="num">${esc(d.primary_channel)}</td></tr>
          <tr><th>Read as</th><td class="num">${esc(d.source)}</td></tr>
          ${d.declared_system_type ? `<tr><th>Declared system</th><td class="num">${esc(d.declared_system_type)}</td></tr>` : ""}
        </table>
        ${problems.length
          ? `<ul class="checks">${problems.map((p) => `<li class="fail">${esc(p)}</li>`).join("")}</ul>`
          : `<ul class="checks"><li class="pass">Complete data in meV, including zero bias.</li></ul>`}
      </div>
      <div class="box plot-box">
        <div class="plot-heading"><div><span class="eyebrow">Measured signal</span><h3>Site-resolved dI/dV</h3></div>
          <span class="plot-window" id="data-window-label">Analysis window 0–${initialCutoff.toFixed(1)} meV</span></div>
        ${channelChoices}
        <div id="data-spectrum">${spectraSvg(d.plot, { cutoffMev: initialCutoff })}</div>
        <p class="hint">One line per site. ${d.n_bias_points > d.plot.bias_mev.length
          ? `Thinned to ${d.plot.bias_mev.length} points for display.` : ""}</p>
      </div>
      <div class="cutoff-control">
        <div><span class="eyebrow">Your analysis choice</span><h3>Choose the positive-bias cutoff</h3>
          <p class="hint">Adjust the marker on the plot. A compatible model must use the same cutoff.</p></div>
        <div class="cutoff-inputs">
          <input type="range" id="data-cutoff-range" min="1" max="${maximumCutoff}" step="0.5" value="${initialCutoff}">
          <label><input type="number" id="data-cutoff-number" min="1" max="${maximumCutoff}" step="0.5" value="${initialCutoff}"> meV</label>
          <button id="data-use-cutoff" class="primary">Check matching models →</button>
        </div>
      </div>`;
    const range = out.querySelector("#data-cutoff-range");
    const number = out.querySelector("#data-cutoff-number");
    let selectedChannel = d.primary_channel;
    const updateCutoff = (value) => {
      const selected = Math.min(maximumCutoff, Math.max(1, Number(value) || initialCutoff));
      range.value = selected;
      number.value = selected;
      el("reuse-cutoff").value = selected;
      out.querySelector("#data-window-label").textContent = `Analysis window 0–${selected.toFixed(1)} meV`;
      out.querySelector("#data-spectrum").innerHTML = spectraSvg(plots[selectedChannel], { cutoffMev: selected });
    };
    out.querySelectorAll("button[data-plot-channel]").forEach((button) => {
      button.addEventListener("click", () => {
        selectedChannel = button.dataset.plotChannel;
        out.querySelectorAll("button[data-plot-channel]").forEach((candidate) =>
          candidate.classList.toggle("active", candidate === button));
        out.querySelector(".plot-heading h3").textContent =
          `Site-resolved ${plots[selectedChannel].title || selectedChannel}`;
        updateCutoff(number.value);
      });
    });
    range.addEventListener("input", () => updateCutoff(range.value));
    number.addEventListener("input", () => updateCutoff(number.value));
    out.querySelector("#data-use-cutoff").addEventListener("click", () => {
      updateCutoff(number.value);
      activate("reuse");
    });
    updateCutoff(initialCutoff);
    shareChosenFile(d.path, "data");
  } catch (e) { showError(out, e); }
});

// --- advisor ----------------------------------------------------------------

el("reuse-go").addEventListener("click", async () => {
  const out = el("reuse-out");
  const path = el("reuse-path").value.trim();
  if (!path) { showError(out, new Error("choose a measurement first")); return; }
  busy(out, "Checking model compatibility…");
  try {
    const d = await api("/api/advise", { path, cutoff_mev: el("reuse-cutoff").value });
    shareChosenFile(d.path, "reuse");
    const klass = d.can_use_existing_model ? "good" : "warn";
    const usable = d.artifacts.filter((a) => a.compatible);
    out.innerHTML = `
      <div class="box">
        <div class="verdict ${klass}">${esc(d.action.replace(/_/g, " "))}</div>
        <p>${esc(d.summary)}</p>
        <table>
          <tr><th>System</th><td>${esc(d.system_type)}</td></tr>
          <tr><th>View</th><td>${esc(d.view)}</td></tr>
          <tr><th>Sites</th><td class="num">${d.n_sites}</td></tr>
          <tr><th>Chosen cutoff</th><td class="num">${num(d.manual_cutoff_mev, 1)} meV</td></tr>
        </table>
        <ul class="checks">${d.experiment_checks.map((c) => {
          const failed = c.startsWith("FAIL");
          return `<li class="${failed ? "fail" : "pass"}">${esc(c.replace(/^(PASS|FAIL): /, ""))}</li>`;
        }).join("")}</ul>
      </div>
      <div class="box">
        <h3>Model compatibility</h3>
        ${d.artifacts.length ? `<table>
          <tr><th>Model</th><th>Usable</th><th>Reason</th></tr>
          ${d.artifacts.map((a) => `<tr>
            <td><code>${esc(a.path.split("/").pop())}</code></td>
            <td>${a.compatible ? "<b style='color:var(--good)'>yes</b>" : "no"}</td>
            <td>${a.reasons.length ? a.reasons.map(esc).join("<br>") : "—"}</td>
          </tr>`).join("")}
        </table>` : "<p class='hint'>No models were found to compare against.</p>"}
      </div>
      ${usable.length ? `<div class="box"><b>Next:</b> open <em>Get my couplings</em>.
        The measurement is already selected.</div>` : ""}
      ${d.next_steps.length ? `<div class="box"><h3>What to do next</h3>
        <ul class="checks">${d.next_steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}`;
  } catch (e) { showError(out, e); }
});

// --- models -----------------------------------------------------------------

let knownModels = [];

function conditionsText(conditions) {
  const keys = Object.keys(conditions || {});
  if (!keys.length) return "none beyond the system itself";
  return keys.map((k) => `${esc(k)} = ${esc(JSON.stringify(conditions[k]))}`).join("<br>");
}

async function loadModels() {
  const out = el("models-out");
  busy(out, "Reading models…");
  try {
    const { models } = await api("/api/models");
    knownModels = models;
    renderAnalysisModels();
    if (!models.length) {
      out.innerHTML = `<div class="box">No models found yet. Train one on the
        <em>Train a model</em> page and it will appear here.</div>`;
      return;
    }
    out.innerHTML = models.map((m) => `
      <div class="box">
        <div class="verdict">${esc(m.label || m.name)}
          <span class="pill ${m.origin === "yours" ? "finished" : "running"}">${
            m.origin === "yours" ? "you trained this" : "published"}</span></div>
        <table>
          <tr><th>System</th><td>${esc(m.system_type)} · ${esc(m.view)} view · L = ${m.n_sites}</td></tr>
          <tr><th>Bias window</th><td>0 to ${num(m.bias_cutoff_mev, 1)} meV · observable ${esc(m.observable)}</td></tr>
          <tr><th>Trained on</th><td>${m.n_training_chains ?? "—"} simulated chains · ${esc(m.model)} (${esc(m.preset)})</td></tr>
          <tr><th>Held-out MAE</th><td class="num">${num(m.test_mae_mev)} meV</td></tr>
          <tr><th>Couplings</th><td>${m.parameters.map((p) =>
            `${esc(p.name)}${p.trained_range_mev ? ` <span class="hint">[${p.trained_range_mev.join(", ")}]</span>` : ""}`).join(" · ")}</td></tr>
          <tr><th>Must match exactly</th><td>${conditionsText(m.fixed_conditions)}</td></tr>
        </table>
        ${m.has_model_card ? `<button data-card="${esc(m.label || m.name)}">Read the model card</button>` : ""}
      </div>`).join("");
    out.querySelectorAll("button[data-card]").forEach((b) =>
      b.addEventListener("click", () => showCard(b.dataset.card)));
  } catch (e) { showError(out, e); }
}

async function showCard(name) {
  const out = el("model-card-out");
  busy(out, "Loading…");
  try {
    const { markdown } = await api(`/api/model-card?name=${encodeURIComponent(name)}`);
    out.innerHTML = `<div class="box md"><pre class="log">${esc(markdown)}</pre></div>`;
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { showError(out, e); }
}

// --- get my couplings -------------------------------------------------------

let chosenAnalysisModel = null;

function renderAnalysisModels() {
  const box = el("an-models");
  if (!knownModels.length) {
    box.innerHTML = `<div class="box hint">No models yet. Train one, or check the
      published models on <em>Existing models</em>.</div>`;
    return;
  }
  box.innerHTML = knownModels.map((m) => `
    <div class="card${(m.label || m.name) === chosenAnalysisModel ? " chosen" : ""}"
         data-model="${esc(m.label || m.name)}">
      <div class="have">${esc(m.label || m.name)}
        <span class="pill ${m.origin === "yours" ? "finished" : "running"}">${
          m.origin === "yours" ? "yours" : "published"}</span></div>
      <div class="does">${esc(m.system_type)} · L = ${m.n_sites} · ${esc(m.view)} view</div>
      <div class="meta"><b>Applies to:</b> 0 to ${num(m.bias_cutoff_mev, 1)} meV of
        ${esc(m.observable)} &nbsp;·&nbsp; <b>Held-out MAE:</b> ${num(m.test_mae_mev)} meV
        &nbsp;·&nbsp; ${esc(m.preset || "")} preset</div>
    </div>`).join("");
  box.querySelectorAll(".card").forEach((c) =>
    c.addEventListener("click", () => {
      chosenAnalysisModel = c.dataset.model;
      renderAnalysisModels();
    }));
}

el("an-go").addEventListener("click", async () => {
  const out = el("an-out");
  const path = el("an-path").value.trim();
  if (!path) { showError(out, new Error("choose a measurement first")); return; }
  if (!chosenAnalysisModel) { showError(out, new Error("choose a model first")); return; }
  busy(out, "Checking the model against this measurement…");
  try {
    const built = await api("/api/build-analysis", {
      path,
      model: chosenAnalysisModel,
      name: el("an-name").value.trim(),
      allow_development_artifacts: el("an-allow-dev").checked,
    });
    const job = await api("/api/run-analysis", { config_path: built.config_path });
    out.innerHTML = `<div class="box">
      <p>Running at the model's own cutoff of <b>${num(built.cutoff_mev, 1)} meV</b>.
        Results appear here and on the <em>Running</em> tab.</p>
      <p class="hint">Saved as <code>${esc(built.config_path)}</code>, so the same
        analysis can be repeated with <code>hamlet run ${esc(built.config_path)}</code>.</p>
    </div>`;
    pollJob(job.job_id,
      (done) => {
        if (done.status === "failed") {
          showError(out, new Error(done.error || "the analysis failed"));
          return;
        }
        out.innerHTML = analysisResult(done.result) +
          `<pre class="log">${esc(done.lines.join("\n"))}</pre>`;
        refreshJobs();
      },
      (job) => {
        if (!job.lines.length) return;
        busy(out, job.lines[job.lines.length - 1],
             quoteFor(job.job_id, job.elapsed_seconds));
      });
  } catch (e) { showError(out, e); }
});

function couplingRows(table) {
  const valueIndex = table.columns.indexOf("coupling");
  if (valueIndex < 0 || !table.rows.length) return [];
  const uncertaintyIndex = table.columns.indexOf("uncertainty");
  const parameterIndex = table.columns.indexOf("parameter");
  const leftIndex = table.columns.indexOf("left_site");
  const rightIndex = table.columns.indexOf("right_site");
  const bondIndex = table.columns.indexOf("bond");
  return table.rows.slice(0, 20).map((row, index) => ({
    label: parameterIndex >= 0 ? row[parameterIndex]
      : leftIndex >= 0 && rightIndex >= 0 ? `${row[leftIndex]}–${row[rightIndex]}`
      : bondIndex >= 0 ? `bond ${row[bondIndex]}` : `parameter ${index + 1}`,
    left: leftIndex >= 0 ? Number(row[leftIndex]) : null,
    right: rightIndex >= 0 ? Number(row[rightIndex]) : null,
    value: Number(row[valueIndex]),
    uncertainty: uncertaintyIndex >= 0 ? Math.abs(Number(row[uncertaintyIndex])) : 0,
  })).filter((row) => Number.isFinite(row.value));
}

function bondChainChart(rows) {
  const sites = [...new Set(rows.flatMap((row) => [row.left, row.right]))]
    .filter(Number.isFinite).sort((a, b) => a - b);
  const sitePosition = new Map(sites.map((site, index) => [site, index]));
  const gap = sites.length <= 10 ? 96 : 112;
  const edge = sites.length <= 10 ? 48 : 62;
  const y = 126;
  const w = Math.max(760, edge * 2 + Math.max(1, sites.length - 1) * gap);
  const h = 230;
  const magnitudes = rows.map((row) => Math.abs(row.value));
  const maxMagnitude = Math.max(...magnitudes, 1e-9);
  // Shading spans the couplings that are actually present, not zero to the
  // largest. A chain of 32-38 meV bonds occupies the top sixth of an
  // absolute scale, so every bond came out within a tenth of full ink and the
  // picture said "all the same" about a set that varies by 18%. Contrast is
  // what the eye reads a chain with, so it is spent on the range that exists
  // -- and the key prints the two end values, because full contrast over a
  // narrow range would otherwise make a 0.1 meV spread look dramatic.
  const minMagnitude = Math.min(...magnitudes);
  const magnitudeSpan = maxMagnitude - minMagnitude;
  const bonds = rows.map((row) => {
    const leftPosition = sitePosition.get(row.left);
    const rightPosition = sitePosition.get(row.right);
    if (leftPosition === undefined || rightPosition === undefined) return "";
    const x1 = edge + leftPosition * gap;
    const x2 = edge + rightPosition * gap;
    const mid = (x1 + x2) / 2;
    const width = 5 + 5 * Math.abs(row.value) / maxMagnitude;
    const colourClass = row.value < 0 ? "negative" : "positive";
    // Two encodings, deliberately different. Thickness is absolute -- it
    // stays proportional to |J| against the largest bond, so a chain whose
    // couplings really are equal looks equal. Ink is relative, stretched
    // across the observed range, which is what makes a 5 meV difference
    // among 35 meV bonds visible at all. Stroke opacity rather than a
    // computed colour, because the hue still has to come from CSS to mean
    // the sign of J, and that is where the light and dark palettes live.
    const strength = magnitudeSpan > 1e-9
      ? (Math.abs(row.value) - minMagnitude) / magnitudeSpan
      : 1;
    const ink = (0.3 + 0.7 * strength).toFixed(2);
    const uncertainty = Number.isFinite(row.uncertainty) && row.uncertainty > 0
      ? `± ${row.uncertainty.toFixed(2)}` : "";
    return `<g class="chain-bond ${colourClass}">
      <title>sites ${esc(row.label)}: ${row.value.toFixed(3)} meV${uncertainty ? ` ${uncertainty} meV` : ""}${
        magnitudeSpan > 1e-9
          ? ` — ${(100 * Math.abs(row.value) / maxMagnitude).toFixed(0)}% of the strongest bond`
          : ""}</title>
      <line x1="${x1 + 23}" x2="${x2 - 23}" y1="${y}" y2="${y}"
        style="stroke-width:${width.toFixed(1)};stroke-opacity:${ink}"/>
      <rect class="bond-value-bg" x="${mid - 43}" y="35" width="86" height="52" rx="8"/>
      <text class="bond-symbol" x="${mid}" y="53" text-anchor="middle">J${esc(row.left)},${esc(row.right)}</text>
      <text class="bond-value" x="${mid}" y="70" text-anchor="middle">${row.value.toFixed(2)} meV</text>
      ${uncertainty ? `<text class="bond-uncertainty" x="${mid}" y="83" text-anchor="middle">${uncertainty}</text>` : ""}
    </g>`;
  }).join("");
  const atoms = sites.map((site, index) => {
    const x = edge + index * gap;
    return `<g class="chain-site"><circle cx="${x}" cy="${y}" r="23"/>
      <circle class="site-highlight" cx="${x - 7}" cy="${y - 8}" r="5"/>
      <text x="${x}" y="${y + 5}" text-anchor="middle">${esc(site)}</text></g>`;
  }).join("");
  return `<div class="coupling-chart chain-coupling-chart">
    <div class="chain-scroll"><svg viewBox="0 0 ${w} ${h}" style="min-width:${w}px" role="img" aria-label="Spin chain with inferred nearest-neighbour couplings in meV">
      <text class="chain-axis-title" x="${edge}" y="18">INFERRED BOND COUPLINGS</text>
      ${bonds}${atoms}
      <text class="chain-caption" x="${w / 2}" y="181" text-anchor="middle">chain site</text>
      <g class="chain-key" transform="translate(${Math.max(edge, w / 2 - 265)} 204)">
        <line x1="0" x2="32" y1="0" y2="0" class="positive"/><text x="42" y="4">positive J</text>
        <line x1="132" x2="164" y1="0" y2="0" class="negative"/><text x="174" y="4">negative J</text>
        <line x1="272" x2="292" y1="0" y2="0" class="positive"
          style="stroke-width:7;stroke-opacity:.3"/>
        <line x1="296" x2="316" y1="0" y2="0" class="positive"
          style="stroke-width:7;stroke-opacity:.65"/>
        <line x1="320" x2="340" y1="0" y2="0" class="positive"
          style="stroke-width:7;stroke-opacity:1"/>
        <text x="350" y="4">${magnitudeSpan > 1e-9
          ? `shading spans ${minMagnitude.toFixed(2)} to ${maxMagnitude.toFixed(2)} meV`
          : "all bonds equal"}</text>
      </g>
    </svg></div>
    <p class="chart-explanation">Circles mark measured sites. Labels give the inferred coupling and model spread; line thickness scales with |J|.</p>
  </div>`;
}

function parameterCouplingChart(rows) {
  if (!rows.length) return "";
  const bounds = rows.flatMap((row) => [row.value - row.uncertainty, row.value + row.uncertainty, 0]);
  let lo = Math.min(...bounds), hi = Math.max(...bounds);
  if (lo === hi) { lo -= 1; hi += 1; }
  const w = 880, labelWidth = 118, right = 32, top = 28, rowHeight = 35, bottom = 38;
  const h = top + rows.length * rowHeight + bottom;
  const x = (value) => labelWidth + ((value - lo) / (hi - lo)) * (w - labelWidth - right);
  const zero = x(0);
  const ticks = [0, .25, .5, .75, 1].map((fraction) => {
    const value = lo + fraction * (hi - lo), px = x(value);
    return `<line class="plot-grid" x1="${px}" x2="${px}" y1="${top - 8}" y2="${h - bottom}"/>
      <text class="plot-tick" x="${px}" y="${h - 13}" text-anchor="middle">${value.toFixed(1)}</text>`;
  }).join("");
  const bars = rows.map((row, index) => {
    const y = top + index * rowHeight + 6;
    const valueX = x(row.value), errorLo = x(row.value - row.uncertainty), errorHi = x(row.value + row.uncertainty);
    return `<text class="coupling-label" x="${labelWidth - 10}" y="${y + 13}" text-anchor="end">${esc(row.label)}</text>
      <rect class="coupling-bar" x="${Math.min(zero, valueX)}" y="${y}" width="${Math.max(2, Math.abs(valueX - zero))}" height="18" rx="4"/>
      ${row.uncertainty ? `<line class="error-bar" x1="${errorLo}" x2="${errorHi}" y1="${y + 9}" y2="${y + 9}"/>
        <line class="error-bar" x1="${errorLo}" x2="${errorLo}" y1="${y + 4}" y2="${y + 14}"/>
        <line class="error-bar" x1="${errorHi}" x2="${errorHi}" y1="${y + 4}" y2="${y + 14}"/>` : ""}
      <text class="coupling-value" x="${Math.min(w - right, Math.max(labelWidth, valueX))}" y="${y + 14}" dx="${valueX >= zero ? 7 : -7}" text-anchor="${valueX >= zero ? "start" : "end"}">${row.value.toFixed(2)}</text>`;
  }).join("");
  return `<div class="coupling-chart"><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Inferred couplings in meV">
    ${ticks}<line class="zero-line" x1="${zero}" x2="${zero}" y1="${top - 8}" y2="${h - bottom}"/>${bars}
    <text class="plot-axis-label" x="${(labelWidth + w - right) / 2}" y="${h - 1}" text-anchor="middle">Coupling [meV]</text>
  </svg></div>`;
}

function couplingChart(table) {
  const rows = couplingRows(table);
  if (!rows.length) return "";
  const isBondChain = rows.every((row) =>
    Number.isFinite(row.left) && Number.isFinite(row.right));
  return isBondChain ? bondChainChart(rows) : parameterCouplingChart(rows);
}

function prettyTable(table) {
  const numeric = new Set(["coupling", "uncertainty"]);
  return `<div class="table-scroll"><table class="coupling-table">
    <tr>${table.columns.map((column) => `<th>${esc(column.replace(/_/g, " "))}</th>`).join("")}</tr>
    ${table.rows.map((row) => `<tr>${row.map((value, index) => {
      const column = table.columns[index];
      const shown = numeric.has(column) && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : value;
      return `<td class="${numeric.has(column) ? "num" : ""}">${esc(shown)}</td>`;
    }).join("")}</tr>`).join("")}
  </table></div>`;
}

function analysisResult(result) {
  if (!result) return "";
  const table = result.couplings || { columns: [], rows: [] };
  const diagnostics = result.diagnostics || {};
  const warnings = diagnostics.warnings || [];
  const statusLabel = result.status === "ok" ? "Complete" : "Check warnings";
  const hasBondRows = table.columns.includes("left_site") && table.columns.includes("right_site");
  return `<div class="result-dashboard">
    <div class="result-hero">
      <div><span class="eyebrow">Inference results</span><h3>Estimated couplings</h3>
        <p class="hint">Model: ${esc(result.model_label || "selected model")}</p></div>
      <span class="result-status ${result.status === "ok" ? "good" : "warn"}">${esc(statusLabel)}</span>
    </div>
    <div class="result-facts">
      <div><span>Model</span><strong>${esc(result.model_name || result.model_label || "—")}</strong></div>
      <div><span>Analysis window</span><strong>0–${num(result.cutoff_mev, 1)} meV</strong></div>
      <div><span>Chain</span><strong>${result.n_sites || "—"} sites</strong></div>
      <div><span>View</span><strong>${esc((result.view || "—").replace(/_/g, " "))}</strong></div>
    </div>
    ${table.rows.length ? `<div class="result-section"><div class="plot-heading"><div><span class="eyebrow">Estimated Hamiltonian</span><h3>Couplings</h3></div>
      <span class="hint">${hasBondRows
        ? "Bond positions follow the measured chain. Values are also listed below."
        : "Bars show estimates; whiskers show model spread where available."}</span></div>
      ${couplingChart(table)}${prettyTable(table)}
      ${table.truncated ? `<p class="hint">Showing the first ${table.rows.length} of
      ${table.n_rows} rows; the CSV has all of them.</p>` : ""}` : ""}
    </div>
    <div class="result-section diagnostics-panel">
      <span class="eyebrow">Quality checks</span><h3>${warnings.length ? "Warnings" : "Checks passed"}</h3>
      ${warnings.length ? `<ul class="checks">${warnings.map((warning) => `<li class="fail">${esc(warning)}</li>`).join("")}</ul>`
        : `<ul class="checks"><li class="pass">No warnings.</li></ul>`}
      ${diagnostics.ensemble_size ? `<p class="hint">Ensemble: ${diagnostics.ensemble_size} model${diagnostics.ensemble_size === 1 ? "" : "s"}
        ${Number.isFinite(diagnostics.max_ensemble_std) ? ` · largest spread ${num(diagnostics.max_ensemble_std)} meV` : ""}</p>` : ""}
    </div>
    ${result.summary_png ? `<div class="result-section"><span class="eyebrow">Quality-control figure</span><h3>Spectra and reconstruction</h3>
      <a href="/api/file?path=${encodeURIComponent(result.summary_png)}" target="_blank" rel="noopener">
        <img class="result-figure" src="/api/file?path=${encodeURIComponent(result.summary_png)}" alt="Quality-control summary of the inferred couplings">
      </a></div>` : ""}
    <div class="result-section"><span class="eyebrow">Files</span><h3>Exported results</h3>
    <div class="result-files">${[
      ["report_pdf", "report.pdf", "a paper-shaped summary to send to a colleague"],
      ["report_tex", "report.tex", "its LaTeX source — compiles on Overleaf as-is"],
      ["report_html", "report.html", "the full report, self-contained"],
      ["summary_png", "summary.png", "quality-control figure"],
      ["couplings_csv", "couplings.csv", "the coupling table"],
      ["report_json", "report.json", "the same numbers, machine-readable"],
    ].filter(([key]) => result[key])
     .map(([key, label, what]) => `<div>${fileLink(result[key], label)}<span>${esc(what)}</span></div>`)
     .join("")}</div></div>
  </div>`;
}

// --- train: a guided form, no YAML ------------------------------------------
// The options come from the server so the form cannot offer choices the
// library would reject.

let builder = null;
let chosenSystem = null;
let chosenModel = null;
let builtConfig = null;

function systemSpec() {
  return builder.systems.find((s) => s.system_type === chosenSystem);
}

function modelSpec() {
  return builder.models.find((m) => m.name === chosenModel);
}

function renderCouplingRows() {
  const spec = systemSpec();
  const body = el("f-couplings").querySelector("tbody");
  const header =
    "<tr><th>coupling</th><th class='num'>from [meV]</th><th class='num'>to [meV]</th><th></th></tr>";
  body.innerHTML = header + spec.couplings.map((c, i) => `
    <tr>
      <td><code>${esc(c.name)}</code></td>
      <td class="num"><input type="number" step="0.1" data-range="${i}" data-edge="low"
          value="${c.low}" style="width:7em"></td>
      <td class="num"><input type="number" step="0.1" data-range="${i}" data-edge="high"
          value="${c.high}" style="width:7em"></td>
      <td class="hint">${c.min !== undefined ? `must stay ≥ ${c.min}` : ""}</td>
    </tr>`).join("");
  if (spec.coupling_mode === "single_range") {
    body.insertAdjacentHTML("beforeend",
      `<tr><td colspan="4" class="hint">One range shared by every bond; each bond
       is drawn from it independently, which is what makes the chain
       inhomogeneous.</td></tr>`);
  }
}

function impurityRow(imp) {
  return `<tr>
    <td>site <input type="number" class="imp-site" value="${imp.site}" min="0" step="1" style="width:5em"></td>
    <td>spin <select class="imp-spin">${builder.spins.map((s) =>
      `<option value="${esc(s)}"${s === imp.spin ? " selected" : ""}>${esc(s)}</option>`).join("")}</select></td>
    <td>transverse E <input type="number" class="imp-transverse" value="${imp.transverse_mev}"
      step="0.1" style="width:5.5em"> meV</td>
    <td>axial D <input type="number" class="imp-axial" value="${imp.axial_mev || 0}"
      step="0.1" style="width:5.5em"> meV</td>
    <td><button class="imp-remove">remove</button></td>
  </tr>`;
}

function renderImpurities(list) {
  const body = el("f-impurities").querySelector("tbody");
  body.innerHTML = list.map(impurityRow).join("");
  body.querySelectorAll(".imp-remove").forEach((b) =>
    b.addEventListener("click", () => {
      b.closest("tr").remove();
      drawImpurityChain();
    }));
  body.querySelectorAll(".imp-site, .imp-spin").forEach((input) =>
    input.addEventListener("change", drawImpurityChain));
  drawImpurityChain();
}

function readImpurities() {
  return [...el("f-impurities").querySelectorAll("tbody tr")].map((tr) => ({
    site: Number(tr.querySelector(".imp-site").value),
    spin: tr.querySelector(".imp-spin").value,
    transverse_mev: Number(tr.querySelector(".imp-transverse").value),
    axial_mev: Number(tr.querySelector(".imp-axial").value),
  }));
}

// --- the chain, drawn -------------------------------------------------------
// Impurity positions are the thing people get wrong: they are zero-based, they
// have to be distinct, and whether an arrangement can expose DMI at all depends
// on where they sit relative to the ends. A row of numbers hides all of that; a
// picture of the chain does not, and clicking the site you mean is a shorter
// path than typing its index.

const CHAIN_GEOMETRY = { radius: 13, padX: 20, padY: 30, longChain: 14 };

function chainSvg(nSites, marked, labelFor) {
  const { padX, padY, longChain } = CHAIN_GEOMETRY;
  // Long chains draw smaller and closer together so a 20-site chain still fits
  // across the panel. Both shrink, so the sticks stay visible rather than the
  // balls growing into each other; the circles stay large enough to click.
  const long = nSites > longChain;
  const radius = long ? 11 : CHAIN_GEOMETRY.radius;
  const gap = long ? 36 : 46;
  const width = 2 * (padX + radius) + Math.max(nSites - 1, 0) * gap;
  const height = 2 * (padY + radius);
  const cy = height / 2;
  const x = (i) => padX + radius + i * gap;

  const bonds = [];
  for (let i = 0; i < nSites - 1; i += 1) {
    bonds.push(`<line class="bond" x1="${x(i) + radius}" y1="${cy}"
      x2="${x(i + 1) - radius}" y2="${cy}"/>`);
  }
  const sites = [];
  for (let i = 0; i < nSites; i += 1) {
    const on = marked.includes(i);
    const label = on && labelFor ? labelFor(i) : "";
    sites.push(`<g class="site${on ? " marked" : ""}" data-site="${i}"
        role="button" tabindex="0">
      <title>site ${i}${on ? " — impurity here; click to remove" : " — click to put an impurity here"}</title>
      <circle class="hit" cx="${x(i)}" cy="${cy}" r="${radius + 8}"/>
      <circle class="ball" cx="${x(i)}" cy="${cy}" r="${radius}"/>
      <text class="index" x="${x(i)}" y="${cy + radius + 15}">${i}</text>
      ${label ? `<text class="site-label" x="${x(i)}" y="${cy - radius - 8}">${esc(label)}</text>` : ""}
    </g>`);
  }
  return `<svg class="chain-svg" viewBox="0 0 ${width} ${height}"
    style="max-width:${width}px" preserveAspectRatio="xMidYMid meet"
    role="group" aria-label="chain of ${nSites} sites">
    ${bonds.join("")}${sites.join("")}</svg>`;
}

function renderChain(host, { nSites, marked, onToggle, labelFor }) {
  host.innerHTML = chainSvg(nSites, marked, labelFor);
  host.querySelectorAll("[data-site]").forEach((node) => {
    const site = Number(node.dataset.site);
    node.addEventListener("click", () => onToggle(site));
    // Reachable by keyboard as well: the diagram is the primary control here,
    // not decoration on top of one.
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onToggle(site);
      }
    });
  });
}

// A site that fell off the end when the chain was shortened cannot be drawn.
// Saying so beats dropping it silently, which would change the design behind
// the user's back, or hiding it, which would leave a run to fail later.
function offChainWarning(sites, nSites) {
  const stray = sites.filter((site) => site >= nSites || site < 0);
  if (!stray.length) return "";
  return `<p class="hint failtext">Impurity site(s) ${stray.join(", ")} lie
    outside a ${nSites}-site chain, so they are not drawn. Remove them, or make
    the chain longer.</p>`;
}

function drawImpurityChain() {
  const host = el("f-chain");
  if (!host) return;
  const nSites = Number(el("f-n-sites").value) || 0;
  const impurities = readImpurities();
  const sites = impurities.map((item) => item.site);
  const bySite = new Map(impurities.map((item) => [item.site, item]));
  renderChain(host, {
    nSites,
    marked: sites.filter((site) => site >= 0 && site < nSites),
    labelFor: (site) => (bySite.get(site) || {}).spin || "",
    onToggle: (site) => {
      const existing = readImpurities();
      const already = existing.some((item) => item.site === site);
      renderImpurities(
        already
          ? existing.filter((item) => item.site !== site)
          : [...existing, { site, ...defaultImpurity() }].sort(
              (a, b) => a.site - b.site
            )
      );
    },
  });
  host.insertAdjacentHTML("beforeend", offChainWarning(sites, nSites));
}

// A new impurity copies the system's own default arrangement rather than an
// invented one, so clicking a site produces something that can actually expose
// DMI instead of an inert S=1 with no transverse anisotropy.
function defaultImpurity() {
  const template = (systemSpec().default_impurities || [])[0];
  return {
    spin: (template && template.spin) || "S=1",
    transverse_mev: template ? template.transverse_mev : 2.0,
    axial_mev: template ? template.axial_mev : 0.0,
  };
}

// --- hyperparameters, including the layer stack -----------------------------
// A network's shape is the thing people most want to change and the thing a
// text box expresses worst, so layers get their own editor: one row per layer,
// added and removed like the impurity list above.

function currentWidths(name) {
  return [...el(`layers-${name}`).querySelectorAll(".layer-units")]
    .map((input) => Number(input.value));
}

// Re-rendered rather than mutated: replacing the container's contents drops the
// old listeners with them, so adding a layer cannot leave a second handler
// behind on the ones that were already there.
function renderLayers(name, widths) {
  const host = el(`layers-${name}`);
  host.innerHTML = widths.map((units, index) => `<span class="layer">
      <label>${index + 1}<input type="number" class="layer-units" min="1" step="16"
        value="${units}" style="width:6.5em"></label>
      <button class="layer-remove" title="remove this layer"${
        widths.length > 1 ? "" : " disabled"}>×</button>
    </span>`).join("") + `<button class="layer-add">+ layer</button>`;
  host.querySelector(".layer-add").addEventListener("click", () => {
    const current = currentWidths(name);
    const last = current.length ? current[current.length - 1] : 128;
    renderLayers(name, [...current, Math.max(16, Math.round(last / 2))]);
  });
  host.querySelectorAll(".layer-remove").forEach((button, index) =>
    button.addEventListener("click", () => {
      const current = currentWidths(name);
      if (current.length > 1) renderLayers(name, current.filter((_, i) => i !== index));
    }));
}

/** A small (i) that reveals the explanation belonging to `id`. */
function infoButton(id) {
  return ` <button type="button" class="info" data-info="${esc(id)}"
    aria-expanded="false" aria-controls="${esc(id)}"
    title="What does this do?">i</button>`;
}

// Delegated and registered once: the hyperparameter rows are rebuilt whenever
// the model changes, so a listener bound to each button would be discarded
// with the row that carried it.
document.addEventListener("click", (event) => {
  const target = event.target;
  const button = target instanceof Element ? target.closest("[data-info]") : null;
  if (!button) return;
  const body = document.getElementById(button.dataset.info);
  if (!body) return;
  const opening = body.hidden;
  body.hidden = !opening;
  button.setAttribute("aria-expanded", String(opening));
  button.classList.toggle("open", opening);
});

function optionField(option) {
  const common = `data-option="${esc(option.name)}" data-type="${esc(option.type)}"`;
  if (option.type === "layers") {
    return `<div class="layers" id="layers-${esc(option.name)}"
      data-layers="${esc(option.name)}"></div>`;
  }
  if (option.type === "boolean") {
    return `<input type="checkbox" ${common}${option.default ? " checked" : ""}>`;
  }
  if (option.type === "choice") {
    return `<select ${common}>${option.choices.map((c) =>
      `<option value="${esc(c)}"${c === option.default ? " selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
  }
  const step = option.type === "integer" ? 1 : "any";
  const value = option.default === null || option.default === undefined ? "" : option.default;
  return `<input type="number" ${common} value="${value}" step="${step}" style="width:9em"
    placeholder="library default">`;
}

function renderModelOptions() {
  const spec = modelSpec();
  const box = el("f-model-options");
  if (!spec.options.length) {
    box.innerHTML = `<div class="box hint">No adjustable hyperparameters are
      available for ${esc(spec.title)}. Library defaults will be used.</div>`;
    return;
  }
  box.innerHTML = `<div class="box"><p class="hint">Fields initially show the
    library defaults. Clear a field to restore its default value.</p>` +
    spec.options.map((o) => {
      // Explanations sit behind an (i) rather than under every field. Eight
      // of them open at once turns a form you can scan into a wall of prose,
      // and the person who already knows what dropout does should not have
      // to scroll past a paragraph saying so.
      const id = `info-${spec.name}-${o.name}`;
      return `<div class="option">
      <label class="option-label">${esc(o.label)}${o.hint ? infoButton(id) : ""}</label>
      <div class="option-input">${optionField(o)}</div>
      ${o.hint ? `<div class="hint option-hint info-body" id="${id}" hidden>${
        esc(o.hint)}</div>` : ""}
    </div>`;
    }).join("") + "</div>";
  spec.options.filter((o) => o.type === "layers").forEach((o) =>
    renderLayers(o.name, o.default || []));
}

function readModelOptions() {
  const options = {};
  const box = el("f-model-options");
  box.querySelectorAll("[data-option]").forEach((input) => {
    const name = input.dataset.option;
    if (input.dataset.type === "boolean") options[name] = input.checked;
    else if (input.dataset.type === "choice") options[name] = input.value;
    else if (input.value !== "") options[name] = Number(input.value);
  });
  box.querySelectorAll(".layers").forEach((editor) => {
    options[editor.dataset.layers] = currentWidths(editor.dataset.layers);
  });
  return options;
}

function renderTuningNote() {
  const tuning = builder.tuning;
  const searched = tuning.searched[chosenModel];
  const box = el("f-tune-note");
  const enabled = el("f-tune").checked;
  if (!searched) {
    box.innerHTML = `<div class="box hint">There is no search space for
      ${esc(modelSpec().title)}.</div>`;
    el("f-tune").checked = false;
    el("f-tune").disabled = true;
    return;
  }
  el("f-tune").disabled = false;
  if (!enabled) {
    box.innerHTML = `<p class="hint">Search disabled. The settings above will be used.</p>`;
    return;
  }
  const trials = Number(el("f-tune-trials").value) || 0;
  box.innerHTML = `<div class="box">
    <p>Will vary: ${searched.map((s) => `<code>${esc(s)}</code>`).join(", ")}.</p>
    <p class="hint">
      ${trials} trial(s), each one short training run at the
      <code>${esc(tuning.trial_preset)}</code> preset, after generation and
      before the real training. Using <b>${esc(tuning.backend)}</b>${
        tuning.optuna_available ? "" : " — install the <code>tune</code> extra for Optuna's TPE sampler, which spends later trials near the good region"}.
      The first trial uses the library defaults. Model selection uses the
      validation split; the test split remains held out. Other settings remain
      fixed.
    </p>
  </div>`;
}

function selectSystem(systemType) {
  chosenSystem = systemType;
  const spec = systemSpec();
  document.querySelectorAll("#sys-cards .card").forEach((c) =>
    c.classList.toggle("chosen", c.dataset.system === systemType));
  el("sys-warning").innerHTML = spec.warning
    ? `<div class="error"><b>Read this first.</b><br>${esc(spec.warning)}</div>` : "";

  const d = spec.defaults;
  el("f-n-sites").value = d.n_sites;
  el("f-n-samples").value = 500;
  // Starts at one rather than at the core count: parallel chains multiply
  // memory use as well as throughput, and a default that quietly saturates
  // the machine is not a good surprise on a shared login node. The hint says
  // how to ask for all of them.
  el("f-workers").value = 1;
  if (!el("f-site-spin").options.length) {
    el("f-site-spin").innerHTML = (builder.chain_spins || ["S=1/2"]).map((s) =>
      `<option value="${esc(s)}">${esc(s)}</option>`).join("");
  }
  el("f-site-spin").value = "S=1/2";
  el("f-bias-lo").value = d.bias_range_mev[0];
  el("f-bias-hi").value = d.bias_range_mev[1];
  el("f-bias-points").value = d.bias_points;
  el("f-broadening").value = d.broadening_mev;
  el("f-observable").value = d.observable;
  el("f-cutoff").value = d.cutoff_mev;
  el("f-output-points").value = d.output_points;
  renderCouplingRows();
  el("f-impurity-block").hidden = !spec.supports_impurities;
  if (spec.supports_impurities) renderImpurities(spec.default_impurities || []);
  selectModel(d.model);
  el("f-plan-out").innerHTML = "";
  el("f-preview-out").innerHTML = "";
  el("f-run-zone").hidden = true;
  builtConfig = null;
}

function selectModel(name) {
  chosenModel = name;
  document.querySelectorAll("#f-model-cards .card").forEach((c) =>
    c.classList.toggle("chosen", c.dataset.model === name));
  renderModelOptions();
  renderTuningNote();
}

function readForm() {
  const spec = systemSpec();
  const rows = [...el("f-couplings").querySelectorAll("input[data-range]")];
  const ranges = spec.couplings.map((_, i) => [
    Number(rows.find((r) => +r.dataset.range === i && r.dataset.edge === "low").value),
    Number(rows.find((r) => +r.dataset.range === i && r.dataset.edge === "high").value),
  ]);
  const form = {
    name: el("f-name").value.trim() || spec.title,
    system_type: chosenSystem,
    n_sites: Number(el("f-n-sites").value),
    n_samples: Number(el("f-n-samples").value),
    site_spin: el("f-site-spin").value || "S=1/2",
    // An empty field means the default, not zero -- zero is the explicit
    // "one per core" request and is too big a difference to arrive by
    // clearing a box.
    workers: el("f-workers").value === "" ? 1 : Number(el("f-workers").value),
    coupling_ranges_mev: ranges,
    bias_range_mev: [Number(el("f-bias-lo").value), Number(el("f-bias-hi").value)],
    bias_points: Number(el("f-bias-points").value),
    broadening_mev: Number(el("f-broadening").value),
    observable: el("f-observable").value,
    cutoff_mev: Number(el("f-cutoff").value),
    output_points: Number(el("f-output-points").value),
    model: chosenModel,
    preset: el("f-preset").value,
    model_options: readModelOptions(),
    device: chosenDevice,
    tuning: {
      enabled: el("f-tune").checked,
      n_trials: Number(el("f-tune-trials").value),
      timeout_minutes: el("f-tune-timeout").value === "" ? null : Number(el("f-tune-timeout").value),
    },
  };
  if (spec.supports_impurities) {
    form.impurities = readImpurities();
    form.transverse_field_mev = Number(el("f-field").value);
  }
  return form;
}

function sampleSvg(bias, sites, siteLabels, cutoffMev) {
  return spectraSvg(
    { bias_mev: bias, sites, site_labels: siteLabels },
    { cutoffMev, compact: true },
  );
}

// Decoration must never be load-bearing. `waitingQuote` lives in a separate
// file, and if that file is missing -- an older install, a stale cache, a
// proxy that ate it -- calling it throws a ReferenceError from inside
// jobBlock(). That used to propagate to refreshJobs(), whose catch reports the
// server as stopped: a missing quote would black out the jobs page and claim
// the run had died. Now the worst case is no quote.
function quoteFor(seed, elapsedSeconds) {
  try {
    return typeof waitingQuote === "function" ? waitingQuote(seed, elapsedSeconds) : "";
  } catch (_error) {
    return "";
  }
}

async function pollJob(jobId, onDone, onTick) {
  const timer = setInterval(async () => {
    try {
      const job = await api(`/api/job?id=${encodeURIComponent(jobId)}`);
      if (onTick) onTick(job);
      if (job.status !== "running") { clearInterval(timer); onDone(job); }
    } catch (e) { clearInterval(timer); onDone({ status: "failed", error: e.message, lines: [] }); }
  }, 2000);
}

el("f-preview").addEventListener("click", async () => {
  const out = el("f-preview-out");
  busy(out, "Simulating a sample chain…");
  try {
    const job = await api("/api/preview-samples", { form: readForm(), n_samples: 1 });
    pollJob(job.job_id,
      (done) => {
        if (done.status === "failed") {
          showError(out, new Error(done.error || "the simulation failed"));
          return;
        }
        const r = done.result;
        out.innerHTML = `<div class="box">` + r.samples.map((sample, i) => `
          <h4>Sample ${i + 1}</h4>
          <p class="hint">${r.target_names.map((n, j) =>
            `${esc(n)} = ${sample.couplings_mev[j].toFixed(2)}`).join(" · ")} meV</p>
          ${sampleSvg(
            r.bias_mev, sample.sites, r.evaluated_sites, Number(el("f-cutoff").value)
          )}`).join("") +
          `<p class="hint">
             ${r.showing_all_sites
               ? `One line per site, all ${r.n_sites} of them.`
               : `One line per site, showing sites ${r.evaluated_sites.join(", ")}
                  of ${r.n_sites}.`}
             Simulated with ${esc(r.dynamics_mode)}${
               r.dynamics_mode === "DMRG"
                 ? " (approximate; the full run uses the same setting)"
                 : " (exact)"} at a basis size of ${r.hilbert_dimension}.
             Adjust the bias range or broadening if spectral features are cut
             off or poorly resolved.</p></div>`;
      },
      (job) => busy(
        out,
        job.lines.length ? job.lines[job.lines.length - 1]
          : "Simulating a sample chain…",
        quoteFor(job.job_id, job.elapsed_seconds),
      ));
  } catch (e) { showError(out, e); }
});

el("f-plan").addEventListener("click", async () => {
  const out = el("f-plan-out");
  busy(out, "Assembling the run…");
  el("f-run-zone").hidden = true;
  try {
    builtConfig = await api("/api/build-config", { form: readForm() });
    const plan = await api("/api/plan", { config_path: builtConfig.config_path });
    const hours = plan.estimated_generation_seconds
      ? (plan.estimated_generation_seconds / 3600) : null;
    out.innerHTML = `
      <div class="box">
        <div class="verdict">${esc(plan.name || builtConfig.name)}</div>
        ${builtConfig.resuming
          // Stopping a run, changing a number and starting again is ordinary,
          // and the two cases want different words: one picks up where it
          // left off, the other is a new run that leaves the old one alone.
          ? `<p><b>Continuing the run with these exact settings.</b> Chains it
             already simulated are kept and generation resumes from them.
             Change any setting and it becomes a separate run instead.</p>`
          : `<p class="hint">A new run, in its own folder. Earlier runs of this
             project are untouched, and coming back to these settings later
             will resume this one.</p>`}
        <table>
          <tr><th>System</th><td>${esc(plan.system_type)} · ${esc(plan.view)} view</td></tr>
          <tr><th>Stages</th><td>${(plan.stages || []).map(esc).join(" → ")}</td></tr>
          ${plan.generation_chains ? `<tr><th>Chains to simulate</th>
            <td class="num">${plan.generation_chains}</td></tr>` : ""}
          ${hours ? `<tr><th>Rough compute</th><td class="num">${hours.toFixed(1)} h serial
            <span class="hint">(at ${num(plan.seconds_per_chain, 0)} s per chain)</span></td></tr>` : ""}
        </table>
        <h3>Files it will write</h3>
        <table><tr><th>Status</th><th>Path</th><th>What</th></tr>
        ${(plan.outputs || []).map((o) => `<tr><td>${esc(o.status || "")}</td>
          <td><code>${esc(o.path || "")}</code></td>
          <td>${esc(o.description || "")}</td></tr>`).join("")}</table>
        ${(plan.notes || []).length ? `<ul class="checks">${plan.notes.map((n) =>
          `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
        ${(plan.blocking_issues || []).length
          ? `<ul class="checks">${plan.blocking_issues.map((r) =>
              `<li class="fail">${esc(r)}</li>`).join("")}</ul>`
          : ""}
        <p class="hint">Saved as <code>${esc(builtConfig.config_path)}</code>,
          writing into <code>${esc(builtConfig.run_dir)}</code>. Repeat this run
          or submit it to a cluster with:<br>
          <code>hamlet run ${esc(builtConfig.config_path)}</code></p>
      </div>`;
    el("f-run-zone").hidden = (plan.blocking_issues || []).length > 0;
  } catch (e) { showError(out, e); builtConfig = null; }
});

el("f-run").addEventListener("click", async () => {
  if (!builtConfig) return;
  // Stopping is cooperative: the run finishes the chunk it is on, which can
  // be a minute of simulation. Starting the next one during that window is
  // the usual way to end up with two heavy jobs sharing the cores and both
  // crawling, which reads as the interface having gone slow.
  let warning = "Generation and training can take hours. Start it now?";
  try {
    const { jobs } = await api("/api/jobs");
    const busy = jobs.filter((j) => j.status === "running" && j.kind === "project");
    if (busy.length) {
      warning = `${busy.length} run(s) still going:\n`
        + busy.map((j) => `  ${j.label}${j.stopping ? " (stopping)" : ""}`).join("\n")
        + "\n\nThey share the same cores, so starting another makes all of them"
        + " slower. Start it anyway?";
    }
  } catch (e) { /* the confirmation below is still worth asking */ }
  if (!confirm(warning)) return;
  try {
    await api("/api/run-project", { config_path: builtConfig.config_path });
    activate("jobs"); refreshJobs();
  } catch (e) { showError(el("f-plan-out"), e); }
});

el("f-add-impurity").addEventListener("click", () => {
  const existing = readImpurities();
  const used = new Set(existing.map((i) => i.site));
  let site = 0;
  while (used.has(site)) site += 1;
  renderImpurities([...existing, { site, spin: "S=1", transverse_mev: 2.0, axial_mev: 0 }]);
});

// The diagram has to follow the chain length, or it shows a chain that is no
// longer the one being configured.
el("f-n-sites").addEventListener("input", drawImpurityChain);

el("f-toggle-advanced").addEventListener("click", () => {
  const box = el("f-model-options");
  box.hidden = !box.hidden;
});
el("f-tune").addEventListener("change", renderTuningNote);
el("f-tune-trials").addEventListener("input", renderTuningNote);

api("/api/builder-options").then((options) => {
  builder = options;
  el("sys-cards").innerHTML = options.systems.map((s) => `
    <div class="card" data-system="${esc(s.system_type)}">
      <div class="have">${esc(s.title)}</div>
      <div class="does">${esc(s.when)}</div>
      <div class="meta"><b>Recovers:</b> ${esc(s.recovers)}</div>
    </div>`).join("");
  document.querySelectorAll("#sys-cards .card").forEach((c) =>
    c.addEventListener("click", () => selectSystem(c.dataset.system)));

  el("f-model-cards").innerHTML = options.models.map((m) => {
    const blocked = m.needs_tensorflow && !options.tensorflow_available;
    return `<div class="card${blocked ? " disabled" : ""}" data-model="${esc(m.name)}">
      <div class="have">${esc(m.title)}</div>
      <div class="does">${esc(m.notes)}</div>
      ${blocked ? `<div class="meta"><b>Unavailable:</b> TensorFlow is not
        installed in this environment.</div>` : ""}
    </div>`;
  }).join("");
  document.querySelectorAll("#f-model-cards .card").forEach((c) =>
    c.addEventListener("click", () => {
      if (!c.classList.contains("disabled")) selectModel(c.dataset.model);
    }));

  el("f-observable").innerHTML = options.observables.map((o) =>
    `<option value="${esc(o.name)}">${esc(o.title)}</option>`).join("");
  el("f-preset").innerHTML = options.presets.map((p) =>
    `<option value="${esc(p.name)}">${esc(p.title)} — ${esc(p.notes)}</option>`).join("");
  el("f-tune-trials").value = options.tuning.default_trials;
  el("f-tune-trials").max = options.tuning.max_trials;

  selectSystem(options.systems[0].system_type);
}).catch((e) => showError(el("sys-cards"), e));

// --- dmi sample design ------------------------------------------------------

let screening = null;
let builtScreening = null;

// One card per candidate, each with its own chain to click sites on. This page
// is entirely about *where* the impurities go, so the sites are the control and
// the numbers are the annotation, not the other way round.

/** Impurity species that are legal in the chosen host.
 *
 * A substituted site has to differ from the chain it sits in, so the host's
 * own spin is not an impurity. Filtering the menu rather than validating
 * afterwards matters here because the page ships default arrangements: raise
 * the host to S=1 with an unfiltered menu and every default becomes invalid
 * at once, which reads as the page breaking rather than as a rule.
 */
function impuritySpins() {
  const host = el("d-site-spin") ? el("d-site-spin").value : "S=1/2";
  return screening.spins.filter((s) => s !== host);
}

function spinOptions(selected) {
  const legal = impuritySpins();
  const value = legal.includes(selected) ? selected : legal[0];
  return legal.map((s) =>
    `<option value="${esc(s)}"${s === value ? " selected" : ""}>${esc(s)}</option>`
  ).join("");
}

function candidateCard(entry, index) {
  return `<div class="candidate" data-candidate="${index}">
    <div class="row" style="justify-content:space-between">
      <label>name <input type="text" class="cand-label" value="${esc(entry.label || "")}"
        placeholder="what you would build" size="26"></label>
      <button class="cand-remove">remove</button>
    </div>
    <div class="chain cand-chain"></div>
    <input type="hidden" class="cand-sites" value="${esc(siteText(entry.sites))}">
    <input type="hidden" class="cand-spins" value="${esc((entry.spins || []).join(","))}">
    <div class="cand-spin-rows"></div>
    <div class="row">
      <label>new sites get <select class="cand-spin">${spinOptions(entry.spin)}</select></label>
      <label>transverse E <input type="number" class="cand-transverse" step="0.1"
        value="${entry.transverse_mev ?? 2.0}" style="width:5.5em"> meV</label>
      <label>axial D <input type="number" class="cand-axial" step="0.1"
        value="${entry.axial_mev ?? 0}" style="width:5.5em"> meV</label>
      <label>field B <input type="number" class="cand-field" step="0.1"
        value="${entry.transverse_field_mev ?? 0}" style="width:5.5em"> meV</label>
    </div>
    <p class="hint cand-verdict"></p>
  </div>`;
}

function siteText(sites) {
  if (Array.isArray(sites)) return sites.join(", ");
  return sites || "";
}

function parseSites(text) {
  return String(text || "")
    .replace(/;/g, ",")
    .split(",")
    .map((part) => Number(part.trim()))
    .filter((value) => Number.isInteger(value) && value >= 0);
}

// The free symmetry rule, restated on each card as the sites are chosen: one
// impurity can never break it, and neither can none without a field. Saying so
// while the design is being drawn beats saying it after a screening run.
function candidateVerdict(sites, field) {
  const distinct = new Set(sites).size;
  if (distinct >= 2) {
    return ["good", `${distinct} impurities at distinct sites can break the symmetry that hides D_z.`];
  }
  if (field > 0) {
    return ["good", "A transverse field can break the symmetry that hides D_z."];
  }
  if (distinct === 1) {
    return ["bad", "One impurity cannot break the symmetry: D_z stays hidden however good the data."];
  }
  return ["bad", "Nothing here breaks the symmetry that hides D_z."];
}

/** The species at each chosen site, in site order.
 *
 * Held beside the site list rather than inside it, because the sites are what
 * the chain drawing edits and a spin annotation smuggled into that string
 * would be parsed away by parseSites(). A site with no entry yet takes the
 * card's default, which is what "new sites get" selects.
 */
function candidateSpins(card, sites) {
  const stored = String(card.querySelector(".cand-spins").value || "")
    .split(",").map((part) => part.trim()).filter(Boolean);
  const fallback = card.querySelector(".cand-spin").value;
  return sites.map((_, index) => stored[index] || fallback);
}

function renderSpinRows(card, sites, spins) {
  const host = card.querySelector(".cand-spin-rows");
  if (!sites.length) { host.innerHTML = ""; return; }
  // Shown only once there is more than one site to tell apart: a single
  // impurity has nothing to differ from, and the row would just repeat the
  // default above it.
  host.innerHTML = `<div class="row spin-rows">
    <span class="hint">species at each site</span>
    ${sites.map((site, index) => `<label>${site}
      <select class="cand-site-spin" data-index="${index}">${
        spinOptions(spins[index])}</select></label>`).join("")}
  </div>`;
  host.querySelectorAll(".cand-site-spin").forEach((select) =>
    select.addEventListener("change", () => {
      const current = candidateSpins(card, sites);
      current[Number(select.dataset.index)] = select.value;
      card.querySelector(".cand-spins").value = current.join(",");
      drawCandidateChain(card);
    }));
}

function drawCandidateChain(card) {
  const host = card.querySelector(".cand-chain");
  const store = card.querySelector(".cand-sites");
  const nSites = Number(el("d-n-sites").value) || 0;
  const sites = parseSites(store.value);
  const spins = candidateSpins(card, sites);
  // Normalised back into the store so a site that has just been added, and
  // took the default, keeps that species when another site changes.
  card.querySelector(".cand-spins").value = spins.join(",");
  const spinAt = new Map(sites.map((site, index) => [site, spins[index]]));
  renderChain(host, {
    nSites,
    marked: sites.filter((site) => site < nSites),
    labelFor: (site) => spinAt.get(site) || "",
    onToggle: (site) => {
      const current = parseSites(store.value);
      const currentSpins = candidateSpins(card, current);
      const keep = new Map(current.map((s, i) => [s, currentSpins[i]]));
      const next = current.includes(site)
        ? current.filter((value) => value !== site)
        : [...current, site].sort((a, b) => a - b);
      store.value = next.join(", ");
      card.querySelector(".cand-spins").value = next
        .map((s) => keep.get(s) || card.querySelector(".cand-spin").value)
        .join(",");
      drawCandidateChain(card);
    },
  });
  // Paired before filtering. Filtering the two lists separately shifts every
  // species after an off-chain site by one, which silently reassigns them --
  // a site past the chain end is ordinary while the site count is being
  // lowered.
  const onChain = sites
    .map((site, index) => [site, spins[index]])
    .filter(([site]) => site < nSites);
  renderSpinRows(card, onChain.map(([site]) => site), onChain.map(([, spin]) => spin));
  host.insertAdjacentHTML("beforeend", offChainWarning(sites, nSites));
  const [tone, message] = candidateVerdict(
    sites.filter((site) => site < nSites),
    Number(card.querySelector(".cand-field").value) || 0
  );
  const verdict = card.querySelector(".cand-verdict");
  verdict.className = `hint cand-verdict ${tone === "good" ? "goodtext" : "failtext"}`;
  verdict.textContent = message;
}

function renderCandidates(list) {
  const host = el("d-candidates");
  host.innerHTML = list.map(candidateCard).join("");
  host.querySelectorAll(".candidate").forEach((card) => {
    card.querySelector(".cand-remove").addEventListener("click", () => {
      const remaining = readCandidates().filter(
        (_, index) => index !== Number(card.dataset.candidate)
      );
      renderCandidates(remaining);
    });
    card.querySelector(".cand-spin").addEventListener("change", () =>
      drawCandidateChain(card));
    card.querySelector(".cand-field").addEventListener("input", () =>
      drawCandidateChain(card));
    drawCandidateChain(card);
  });
}

function readCandidates() {
  return [...el("d-candidates").querySelectorAll(".candidate")].map((card) => ({
    label: card.querySelector(".cand-label").value.trim(),
    sites: card.querySelector(".cand-sites").value,
    // The card's default for a site clicked next; not sent to the server.
    spin: card.querySelector(".cand-spin").value,
    // `spins` only. Sending both leans on the server preferring one of them,
    // and the library rejects a configuration that gives both -- a
    // disagreement worth not having in the first place.
    spins: candidateSpins(card, parseSites(card.querySelector(".cand-sites").value)),
    transverse_mev: Number(card.querySelector(".cand-transverse").value),
    axial_mev: Number(card.querySelector(".cand-axial").value),
    transverse_field_mev: Number(card.querySelector(".cand-field").value),
  }));
}

function readScreeningForm() {
  return {
    name: el("d-name").value.trim(),
    chain: {
      n_sites: Number(el("d-n-sites").value),
      j_eff_mev: Number(el("d-jeff").value),
      d_z_mev: Number(el("d-dz").value),
      jz_mev: Number(el("d-jz").value),
      j2_mev: Number(el("d-j2").value),
      j3_mev: Number(el("d-j3").value),
      site_spin: el("d-site-spin").value || "S=1/2",
    },
    protocol: {
      bias_range_mev: [Number(el("d-bias-lo").value), Number(el("d-bias-hi").value)],
      bias_points: Number(el("d-bias-points").value),
      broadening_mev: Number(el("d-broadening").value),
      observable: el("d-observable").value,
    },
    candidates: readCandidates(),
  };
}

function symmetryTable(d) {
  const incompatible = d.n_candidates - d.n_can_break_symmetry;
  return `<div class="box">
    <div class="verdict ${d.n_can_break_symmetry ? "good" : "bad"}">
      ${d.n_can_break_symmetry} of ${d.n_candidates} arrangement(s) can break the symmetry
    </div>
    ${incompatible ? `<p class="hint">${incompatible} will be skipped because
      the symmetry prevents them from constraining D_z.</p>` : ""}
    <table><tr><th>Design</th><th>Impurities</th><th>Field</th><th>Can expose DMI?</th></tr>
    ${d.candidates.map((c) => `<tr>
      <td>${esc(c.label)}</td>
      <td>${c.impurities.map((i) => `site ${i.site} ${esc(i.spin)} E=${i.transverse_mev}`).join("<br>") || "none"}</td>
      <td class="num">${c.transverse_field_mev || 0} meV</td>
      <td>${c.breaks_symmetry ? "<b style='color:var(--good)'>yes</b>"
        : "<span class='pill hidden'>no</span>"}</td></tr>`).join("")}
    </table>
    <p class="hint">Saved as <code>${esc(d.config_path)}</code>, so the same screen
      runs from the command line with
      <code>hamlet screen-dmi ${esc(d.config_path)}</code>.</p>
  </div>`;
}

function calibrationBox() {
  return `<div class="box">
    <h3>What an imprint means</h3>
    <p class="hint">The imprint measures the spectral difference between the
      two chains in a gauge pair. Thresholds are based on the D_z prediction
      obtained for the corresponding design.</p>
    <table><tr><th class="num">Imprint</th><th>Measured outcome</th></tr>
      ${screening.calibration.map((c) => `<tr>
        <td class="num">${c.imprint.toExponential(2)}</td>
        <td>${esc(c.note)}</td></tr>`).join("")}
    </table>
    <table><tr><th>Verdict</th><th>Means</th></tr>
      ${screening.verdicts.map((v) => `<tr>
        <td><span class="pill ${esc(v.name)}">${esc(v.name)}</span></td>
        <td>${esc(v.means)}</td></tr>`).join("")}
    </table>
  </div>`;
}

el("d-check").addEventListener("click", async () => {
  const out = el("dmi-out");
  busy(out, "Checking the symmetry rule…");
  try {
    builtScreening = await api("/api/build-screening", { form: readScreeningForm() });
    out.innerHTML = symmetryTable(builtScreening);
    el("dmi-calibration").innerHTML = calibrationBox();
  } catch (e) { showError(out, e); builtScreening = null; }
});

el("d-run").addEventListener("click", async () => {
  const out = el("dmi-out");
  try {
    // Always rebuilt first, so the run screens what is on the page rather than
    // whatever an earlier symmetry check happened to save.
    builtScreening = await api("/api/build-screening", { form: readScreeningForm() });
    if (!builtScreening.n_can_break_symmetry) {
      out.innerHTML = symmetryTable(builtScreening) +
        `<div class="error">None of these arrangements breaks the symmetry that
          hides D_z. Add transverse anisotropy at two distinct impurity sites or
          apply a transverse field before screening.</div>`;
      return;
    }
    await api("/api/run-screening", { config_path: builtScreening.config_path });
    activate("jobs"); refreshJobs();
  } catch (e) { showError(out, e); }
});

el("d-n-sites").addEventListener("input", () =>
  el("d-candidates").querySelectorAll(".candidate").forEach(drawCandidateChain));

el("d-add").addEventListener("click", () => {
  renderCandidates([...readCandidates(), {
    label: "", sites: [], spin: "S=1", transverse_mev: 2.0,
    axial_mev: 0, transverse_field_mev: 0,
  }]);
});

el("d-reset").addEventListener("click", () => {
  renderCandidates(screening.defaults.candidates);
});

api("/api/screening-options").then((options) => {
  screening = options;
  const d = options.defaults;
  el("d-name").value = d.name;
  el("d-n-sites").value = d.chain.n_sites;
  el("d-jeff").value = d.chain.j_eff_mev;
  el("d-dz").value = d.chain.d_z_mev;
  el("d-jz").value = d.chain.jz_mev;
  el("d-j2").value = d.chain.j2_mev;
  el("d-j3").value = d.chain.j3_mev;
  el("d-site-spin").innerHTML = (options.chain_spins || ["S=1/2"]).map((s) =>
    `<option value="${esc(s)}"${s === (d.chain.site_spin || "S=1/2")
      ? " selected" : ""}>${esc(s)}</option>`).join("");
  // The arrangements are redrawn when the host changes: every card shows the
  // species at each site, and which of those are legal depends on the host.
  el("d-site-spin").addEventListener("change", () => {
    // Anything now equal to the host is remapped rather than left invalid,
    // so raising the host never leaves the page in a state that cannot run.
    const legal = impuritySpins();
    renderCandidates(readCandidates().map((entry) => ({
      ...entry,
      spin: legal.includes(entry.spin) ? entry.spin : legal[0],
      spins: (entry.spins || []).map((s) => (legal.includes(s) ? s : legal[0])),
    })));
  });
  el("d-bias-lo").value = d.protocol.bias_range_mev[0];
  el("d-bias-hi").value = d.protocol.bias_range_mev[1];
  el("d-bias-points").value = d.protocol.bias_points;
  el("d-broadening").value = d.protocol.broadening_mev;
  el("d-observable").innerHTML = options.observables.map((o) =>
    `<option value="${esc(o)}"${o === d.protocol.observable ? " selected" : ""}>${esc(o)}</option>`).join("");
  renderCandidates(d.candidates);
  el("dmi-calibration").innerHTML = calibrationBox();
}).catch((e) => showError(el("dmi-out"), e));

// --- where the files go -----------------------------------------------------

// Answered on the front page rather than on request, because the workspace is
// not in the same place for everyone -- beside a checkout, under the home
// directory for an installed package, or wherever HAMLET_WORKSPACE points --
// and someone who does not know which case they are in cannot find their own
// results.
async function loadLocations() {
  try {
    const info = await api("/api/locations");
    el("locations-lead").textContent = info.explanation;
    el("locations-out").innerHTML = `<div class="box">
      <table>
        <tr><th>What</th><th>Folder</th><th class="num">Items</th></tr>
        ${info.locations.map((loc) => `<tr>
          <td><b>${esc(loc.title)}</b><div class="hint">${esc(loc.purpose)}</div></td>
          <td><code>${esc(loc.path)}</code></td>
          <td class="num">${loc.exists ? loc.entries : "—"}</td>
        </tr>`).join("")}
      </table>
      <p class="hint">A folder appears the first time something is written to
        it. Every path a run reports is inside one of these, and the same
        layout is printed by <code>hamlet where</code>.</p>
    </div>`;
  } catch (e) {
    // Not worth an error banner on the front page: the rest of the page works
    // and every result still prints its own full path.
    el("locations-lead").textContent =
      "Run `hamlet where` in the terminal to list the output folders.";
  }
}

// --- where it runs ----------------------------------------------------------

let compute = null;
let chosenDevice = "auto";

function renderCompute() {
  const gpus = compute.accelerators.filter((a) => a.kind === "gpu");
  el("compute-out").innerHTML = `<div class="box">
    <table>
      <tr><th>CPU cores</th><td class="num">${compute.cpu_count}</td></tr>
      <tr><th>TensorFlow</th><td>${compute.tensorflow_available
        ? "installed" : "<b>not installed</b> — the neural models are unavailable"}</td></tr>
      <tr><th>GPU</th><td>${gpus.length
        ? gpus.map((g) => `${esc(g.name)}${g.detail ? ` <span class="hint">(${esc(g.detail)})</span>` : ""}`).join("<br>")
        : "none visible"}</td></tr>
      <tr><th>Can use a GPU</th><td>${compute.gpu_capable_models.map((m) =>
        `<code>${esc(m)}</code>`).join(", ")}</td></tr>
    </table>
    <ul class="checks">${compute.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
  </div>`;

  el("device-cards").innerHTML = compute.devices.map((d) => `
    <div class="card${d.name === chosenDevice ? " chosen" : ""}${
      d.unavailable_here ? " unavailable" : ""}" data-device="${esc(d.name)}">
      <div class="have">${esc(d.title)}${d.unavailable_here
        ? ` <span class="pill cancelled">not on this machine</span>` : ""}</div>
      <div class="does">${esc(d.notes)}</div>
      ${d.unavailable_here
        ? `<div class="meta">${esc(d.unavailable_here)}</div>` : ""}
    </div>`).join("");
  // Chosen but unavailable is a legitimate state -- a configuration built here
  // may be meant for a cluster -- so it warns instead of refusing, and says
  // what would happen if it were run here.
  const chosenCard = compute.devices.find((d) => d.name === chosenDevice);
  const warning = el("device-warning");
  if (chosenCard && chosenCard.unavailable_here) {
    warning.hidden = false;
    warning.innerHTML = `<b>This machine will train on the CPU.</b>
      ${esc(chosenCard.unavailable_here)}
      The setting is still saved, which is what you want if this configuration
      is going to a cluster with a GPU.
      <a href="${esc(compute.gpu_help_url)}" target="_blank" rel="noopener">How
      to get a GPU working</a>.`;
  } else {
    warning.hidden = true;
  }
  el("device-cards").querySelectorAll(".card").forEach((c) =>
    c.addEventListener("click", () => {
      chosenDevice = c.dataset.device;
      renderCompute();
    }));

}

api("/api/compute").then((options) => {
  compute = options;
  renderCompute();
}).catch((e) => showError(el("compute-out"), e));

// --- the cluster form -------------------------------------------------------
// Two fields carry the whole decision: where to ssh, and which batch system.
// The scheduler profile behind the name is not something anyone chooses, so it
// is no longer shown, and the YAML editor that used to be here is now only the
// escape hatch for a site whose scheduler is not one of the five.

const CLUSTER_FIELDS = {
  host: "cl-host", remote_dir: "cl-remote-dir", scheduler: "cl-scheduler",
  cpus: "cl-cpus", gpus: "cl-gpus", memory: "cl-memory",
  walltime: "cl-walltime", queue: "cl-queue", account: "cl-account",
  setup: "cl-setup",
};

function readClusterForm() {
  const form = {};
  for (const [name, id] of Object.entries(CLUSTER_FIELDS)) form[name] = el(id).value;
  return form;
}

api("/api/cluster-form").then((saved) => {
  el("cl-scheduler").innerHTML = saved.schedulers.map((s) =>
    `<option value="${esc(s.name)}">${esc(s.title)}</option>`).join("");
  for (const [name, id] of Object.entries(CLUSTER_FIELDS)) {
    const value = saved.form[name];
    if (value !== undefined && value !== null) el(id).value = value;
  }
  el("cluster-custom-note").hidden = !saved.form.custom_scheduler;
  el("cluster-path").textContent = saved.configured
    ? `saved at ${saved.config_path}`
    : `will be saved at ${saved.config_path}`;
}).catch((e) => showError(el("cluster-out"), e));

el("cluster-save").addEventListener("click", async () => {
  const out = el("cluster-out");
  busy(out, "Checking the settings…");
  try {
    const saved = await api("/api/save-cluster-config", { form: readClusterForm() });
    out.innerHTML = `<div class="box">
      <div class="verdict good">Saved</div>
      <table>
        <tr><th>Host</th><td>${esc(saved.summary.host || "this machine")}</td></tr>
        <tr><th>Directory there</th><td><code>${esc(saved.summary.remote_dir)}</code></td></tr>
        <tr><th>Scheduler</th><td>${esc(saved.summary.scheduler)}</td></tr>
        <tr><th>Asking for</th><td>${Object.entries(saved.summary.resources)
          .map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(" · ") || "the queue default"}</td></tr>
      </table>
      <p class="hint">Test the connection before submitting a run.</p>
    </div>`;
  } catch (e) { showError(out, e); }
});

el("cluster-check").addEventListener("click", async () => {
  const out = el("cluster-out");
  busy(out, "Connecting through SSH… Check the terminal for any prompt.");
  try {
    const d = await api("/api/check-cluster", {});
    const ok = d.reachable && d.scheduler_found;
    out.innerHTML = `<div class="box">
      <div class="verdict ${ok ? "good" : "bad"}">
        ${ok ? "Connection and scheduler available"
             : d.reachable ? "Reachable, but the scheduler was not found"
                           : "Could not reach it"}</div>
      <table>
        <tr><th>Host</th><td>${esc(d.host)}</td></tr>
        <tr><th>Scheduler</th><td>${esc(d.scheduler)} — ${
          d.scheduler_found ? "found" : "<b>not on the PATH there</b>"}</td></tr>
      </table>
      ${d.detail ? `<pre class="log">${esc(d.detail)}</pre>` : ""}
      ${d.hint ? (d.needs_key
        // The key setup is three commands to copy, so it is shown as commands
        // rather than folded into a paragraph that eats the line breaks.
        ? `<p><b>This needs a key, not a password.</b></p>
           <pre class="log">${esc(d.hint)}</pre>`
        : `<p class="hint">${esc(d.hint)}</p>`) : ""}
      ${d.reachable && !d.scheduler_found ? `<p class="hint">If the scheduler
        needs a module loaded first, add that command to the setup lines
        above, or pick the batch system your site actually runs.</p>` : ""}
    </div>`;
  } catch (e) { showError(out, e); }
});

function requireBuiltConfig(out) {
  if (!builtConfig) {
    showError(out, new Error(
      "prepare a configuration on the Train a model page before submitting to a cluster"));
    return null;
  }
  return builtConfig.config_path;
}

el("cluster-script-btn").addEventListener("click", async () => {
  const out = el("cluster-submit-out");
  const configPath = requireBuiltConfig(out);
  if (!configPath) return;
  busy(out, "Building the job script…");
  try {
    const d = await api("/api/cluster-script", { config_path: configPath });
    out.innerHTML = `<div class="box">
      <table>
        <tr><th>Goes to</th><td>${esc(d.host)}:<code>${esc(d.remote_dir)}</code></td></tr>
        <tr><th>Scheduler</th><td>${esc(d.scheduler)}</td></tr>
      </table>
      ${d.portable ? "" : `<div class="error"><b>These paths point outside the
        project directory and will not be copied:</b><br>${
        d.outside_project_dir.map(esc).join("<br>")}</div>`}
      <pre class="log">${esc(d.script)}</pre>
      <p class="hint">Preview only; nothing has been submitted. Add any required
        site-specific directives before manual submission.</p>
    </div>`;
  } catch (e) { showError(out, e); }
});

el("cluster-submit").addEventListener("click", async () => {
  const out = el("cluster-submit-out");
  const configPath = requireBuiltConfig(out);
  if (!configPath) return;
  if (!confirm("Copy the project to the cluster and submit it?")) return;
  try {
    const job = await api("/api/submit-to-cluster", { config_path: configPath });
    activate("jobs");
    refreshJobs();
  } catch (e) { showError(out, e); }
});

// --- jobs -------------------------------------------------------------------

function screeningTable(results) {
  return `<table><tr><th>Design</th><th class="num">Imprint</th><th>Verdict</th></tr>
    ${results.map((r) => `<tr><td>${esc(r.label)}</td>
      <td class="num">${r.imprint.toExponential(3)}</td>
      <td><span class="pill ${esc(r.verdict)}">${esc(r.verdict)}</span></td></tr>`).join("")}</table>`;
}

function trainingResult(result) {
  const tuning = result.tuning;
  return `<table>
    <tr><th>Artifact</th><td><code>${esc(result.artifact_path)}</code></td></tr>
    <tr><th>Validation MAE</th><td class="num">${num(result.validation_mae_mev)} meV</td></tr>
    <tr><th>Held-out MAE</th><td class="num">${num(result.test_mae_mev)} meV</td></tr>
  </table>
  ${tuning ? `<p class="hint">Hyperparameter search (${esc(tuning.backend)}):
    ${tuning.kept_defaults
      ? "the library defaults gave the lowest validation error."
      : `validation MAE improved by ${num(tuning.improvement_mev, 4)} meV —
         <code>${esc(JSON.stringify(tuning.best_options))}</code>.`}
    Full trial table in <code>${esc(tuning.report_path)}</code>.</p>` : ""}
  <p class="hint">This model is now available under <em>Get my couplings</em>
    and <em>Existing models</em>.</p>`;
}

function clusterResult(result) {
  return `<table>
      <tr><th>Job id there</th><td><code>${esc(result.job_id)}</code></td></tr>
      <tr><th>Host</th><td>${esc(result.host)}</td></tr>
      <tr><th>Scheduler</th><td>${esc(result.scheduler)}</td></tr>
      <tr><th>Script</th><td><code>${esc(result.script)}</code></td></tr>
    </table>
    <div class="row">
      <button data-cluster-status="${esc(result.job_id)}">Check status</button>
      <button data-cluster-cancel="${esc(result.job_id)}">Cancel job</button>
      <button data-cluster-fetch="${esc(result.project_dir)}">Fetch results</button>
    </div>
    <p class="hint">The job is running on the cluster and is independent of this
      browser session.</p>`;
}

function jobResult(job) {
  const result = job.result;
  if (!result) return "";
  if (Array.isArray(result)) return screeningTable(result);
  if (result.kind === "analysis") return analysisResult(result);
  if (result.kind === "training") return trainingResult(result);
  if (result.kind === "cluster") return clusterResult(result);
  if (result.kind === "fetch") {
    return `<p class="hint">Fetched into <code>${esc(result.project_dir)}</code>.</p>`;
  }
  return "";
}

function jobBlock(job) {
  const running = job.status === "running";
  return `<div class="box">
    <div class="row" style="justify-content:space-between">
      <b>${esc(job.label)}</b>
      <span>
        ${running ? `<button data-stop="${esc(job.job_id)}"${
          job.stopping ? " disabled" : ""}>${
          job.stopping ? "stopping…" : "Stop"}</button>` : ""}
        <span class="pill ${esc(job.status)}">${esc(job.status)}</span>
        <span class="hint">${job.elapsed_seconds}s</span></span>
    </div>
    ${running ? quoteFor(job.job_id, job.elapsed_seconds) : ""}
    ${job.stopping ? `<p class="hint">Stopping after the current step finishes.</p>` : ""}
    ${job.error ? `<div class="${job.status === "cancelled" ? "box" : "error"}">${
      esc(job.error)}</div>` : ""}
    ${jobResult(job)}
    ${job.lines.length ? `<pre class="log">${esc(job.lines.join("\n"))}</pre>` : ""}
  </div>`;
}

// Stopping is cooperative -- a thread cannot be killed, and killing one
// mid-write would leave a dataset the next run has to distrust -- so the
// confirmation says what will actually happen rather than implying an
// instant halt.
async function stopJob(jobId) {
  try {
    const answer = await api("/api/cancel-job", { job_id: jobId });
    if (answer.stopping) {
      const note = el("jobs-note");
      note.hidden = false;
      note.textContent = answer.detail;
    }
    refreshJobs();
  } catch (e) {
    showError(el("jobs-out"), e);
  }
}

let finishedJobs = new Set();

async function refreshJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    const running = jobs.filter((j) => j.status === "running").length;
    const badge = el("job-badge");
    badge.hidden = running === 0;
    badge.textContent = running;
    el("jobs-out").innerHTML = jobs.length
      ? jobs.map(jobBlock).join("")
      : `<div class="box">Nothing has been run yet.</div>`;
    el("jobs-out").querySelectorAll("button[data-cluster-status]").forEach((b) =>
      b.addEventListener("click", async () => {
        const note = el("jobs-note");
        note.hidden = false;
        note.textContent = "Checking cluster status…";
        try {
          const d = await api("/api/cluster-status", { job_id: b.dataset.clusterStatus });
          note.textContent = (d.known ? "still queued or running. " : "not listed. ")
            + d.note + " " + (d.detail || "");
        } catch (e) { note.textContent = e.message; }
      }));
    el("jobs-out").querySelectorAll("button[data-cluster-cancel]").forEach((b) =>
      b.addEventListener("click", async () => {
        if (!confirm("Cancel this job on the cluster?")) return;
        try {
          const d = await api("/api/cancel-cluster-job", { job_id: b.dataset.clusterCancel });
          const note = el("jobs-note");
          note.hidden = false;
          note.textContent = d.cancelled
            ? `cancelled ${d.job_id} on the cluster. ${d.detail}`
            : `the cluster refused: ${d.detail}`;
        } catch (e) { showError(el("jobs-out"), e); }
      }));
    el("jobs-out").querySelectorAll("button[data-cluster-fetch]").forEach((b) =>
      b.addEventListener("click", async () => {
        try {
          await api("/api/fetch-from-cluster", { project_dir: b.dataset.clusterFetch });
          refreshJobs();
        } catch (e) { showError(el("jobs-out"), e); }
      }));
    el("jobs-out").querySelectorAll("button[data-stop]").forEach((b) =>
      b.addEventListener("click", () => {
        if (confirm("Stop this run? It finishes the step it is on first, and "
                    + "anything already written is kept.")) {
          stopJob(b.dataset.stop);
        }
      }));
    // A finished training run produces a model, and the pages that offer models
    // are stale until they are told.
    jobs.filter((j) => j.kind === "project" && j.status === "finished")
      .forEach((j) => {
        if (!finishedJobs.has(j.job_id)) {
          finishedJobs.add(j.job_id);
          loadModels();
        }
      });
  } catch (e) {
    // A failed poll is how this page learns the server was stopped from the
    // terminal, which is otherwise indistinguishable from a hung interface.
    markStopped();
  }
}

// --- stopping the server ---------------------------------------------------

let stopped = false;
// Declared before the first poll: markStopped() reads it, and that poll can
// fail immediately, which would otherwise hit the const's temporal dead zone.
let jobTimer = null;

function markStopped() {
  if (stopped) return;
  stopped = true;
  document.body.classList.add("stopped");
  el("stopped-banner").hidden = false;
  if (jobTimer) clearInterval(jobTimer);
}

el("stop-server").addEventListener("click", async () => {
  let warning = "Stop the local server?";
  try {
    const { jobs } = await api("/api/jobs");
    const running = jobs.filter((j) => j.status === "running");
    if (running.length) {
      warning = `${running.length} job(s) are still running and will be lost:\n`
        + running.map((j) => `  ${j.label}`).join("\n")
        + "\n\nStop the server anyway?";
    }
  } catch (e) { /* fall through to the plain confirmation */ }
  if (!confirm(warning)) return;
  try {
    await api("/api/shutdown", {});
  } catch (e) { /* the socket usually closes before a reply arrives */ }
  markStopped();
});

refreshJobs();
jobTimer = setInterval(refreshJobs, 3000);
loadModels();
loadLocations();
