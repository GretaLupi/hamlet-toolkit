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

const el = (id) => document.getElementById(id);
const esc = (text) =>
  String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function showError(node, error) {
  node.innerHTML = `<div class="error"><b>Could not do that.</b><br>${esc(error.message)}</div>`;
}

function busy(node, message) {
  node.innerHTML = `<div class="box">${esc(message)}</div>`;
}

function num(value, digits = 3) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(digits);
}

// --- tabs -------------------------------------------------------------------

function activate(name) {
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b.dataset.panel === name));
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `panel-${name}`));
}
document.querySelectorAll("#tabs button").forEach((b) =>
  b.addEventListener("click", () => activate(b.dataset.panel)));

// --- start ------------------------------------------------------------------

const SITUATION_PANEL = {
  inspect: "data", advise: "reuse", models: "models", train: "train", dmi: "dmi",
};

api("/api/overview").then(({ situations }) => {
  el("situations").innerHTML = situations.map((s) => `
    <div class="card" data-go="${esc(SITUATION_PANEL[s.id] || "start")}">
      <div class="have">${esc(s.have)}</div>
      <div class="does">${esc(s.does)}</div>
      <div class="meta"><b>Needs:</b> ${esc(s.needs)} &nbsp;·&nbsp; <b>Takes:</b> ${esc(s.cost)}</div>
    </div>`).join("");
  document.querySelectorAll("#situations .card").forEach((card) =>
    card.addEventListener("click", () => activate(card.dataset.go)));
}).catch((e) => showError(el("situations"), e));

// --- inspect data -----------------------------------------------------------

function spectraSvg(plot) {
  const bias = plot.bias_mev;
  const sites = plot.sites;
  const w = 920, h = 260, padL = 46, padR = 12, padT = 12, padB = 30;
  const flat = sites.flat().filter(Number.isFinite);
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const span = hi - lo || 1;
  const x = (i) => padL + (i / (bias.length - 1)) * (w - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / span) * (h - padT - padB);
  const paths = sites.map((row, index) => {
    const hue = Math.round((index / Math.max(sites.length, 1)) * 300);
    const d = row.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
    return `<path d="${d}" fill="none" stroke="hsl(${hue} 62% 48%)" stroke-width="1.4"/>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const i = Math.round(f * (bias.length - 1));
    return `<text x="${x(i).toFixed(1)}" y="${h - 10}" font-size="11" text-anchor="middle" fill="currentColor" opacity="0.65">${bias[i].toFixed(1)}</text>`;
  }).join("");
  return `<svg class="spectra" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    ${paths}${ticks}
    <text x="${w / 2}" y="${h - 22}" font-size="11" text-anchor="middle" fill="currentColor" opacity="0.65">bias [meV]</text>
    <text x="6" y="${padT + 10}" font-size="11" fill="currentColor" opacity="0.65">dI/dV</text>
  </svg>`;
}

el("data-go").addEventListener("click", async () => {
  const out = el("data-out");
  busy(out, "Reading…");
  try {
    const d = await api("/api/inspect", { path: el("data-path").value.trim() });
    const problems = [];
    if (!d.is_complete) problems.push(`${d.missing_points} missing data point(s) — the workflow refuses incomplete maps`);
    if (!d.starts_at_zero) problems.push(`the bias axis starts at ${num(d.bias_min_mev, 2)} meV, not 0`);
    if (d.bias_units !== "meV") problems.push(`bias axis is in ${d.bias_units}, not meV`);
    out.innerHTML = `
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
          : `<ul class="checks"><li class="pass">Structurally usable: complete, in meV, and starting at zero bias.</li></ul>`}
      </div>
      <div class="box">${spectraSvg(d.plot)}
        <p class="hint">One line per site. ${d.n_bias_points > d.plot.bias_mev.length
          ? `Thinned to ${d.plot.bias_mev.length} points for display.` : ""}</p>
      </div>
      <div class="box"><b>Next:</b> ask whether a published model fits this measurement — go to
        <em>Can I reuse a model?</em> and use the same path.</div>`;
    el("reuse-path").value = d.path;
  } catch (e) { showError(out, e); }
});

// --- advisor ----------------------------------------------------------------

el("reuse-go").addEventListener("click", async () => {
  const out = el("reuse-out");
  busy(out, "Comparing against every published model's contract…");
  try {
    const d = await api("/api/advise", {
      path: el("reuse-path").value.trim(),
      cutoff_mev: el("reuse-cutoff").value,
    });
    const klass = d.can_use_existing_model ? "good" : "warn";
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
        <h3>Why each model was or was not usable</h3>
        ${d.artifacts.length ? `<table>
          <tr><th>Model</th><th>Usable</th><th>Reason</th></tr>
          ${d.artifacts.map((a) => `<tr>
            <td><code>${esc(a.path.split("/").pop())}</code></td>
            <td>${a.compatible ? "<b style='color:var(--good)'>yes</b>" : "no"}</td>
            <td>${a.reasons.length ? a.reasons.map(esc).join("<br>") : "—"}</td>
          </tr>`).join("")}
        </table>` : "<p class='hint'>No published models were found to compare against.</p>"}
      </div>
      ${d.next_steps.length ? `<div class="box"><h3>What to do next</h3>
        <ul class="checks">${d.next_steps.map((s) => `<li>${esc(s)}</li>`).join("")}</ul></div>` : ""}`;
  } catch (e) { showError(out, e); }
});

// --- models -----------------------------------------------------------------

function conditionsText(conditions) {
  const keys = Object.keys(conditions || {});
  if (!keys.length) return "none beyond the system itself";
  return keys.map((k) => `${esc(k)} = ${esc(JSON.stringify(conditions[k]))}`).join("<br>");
}

async function loadModels() {
  const out = el("models-out");
  busy(out, "Reading published models…");
  try {
    const { models, root } = await api("/api/models");
    if (!models.length) {
      out.innerHTML = `<div class="box">No published models in <code>${esc(root)}</code>.</div>`;
      return;
    }
    out.innerHTML = models.map((m) => `
      <div class="box">
        <div class="verdict">${esc(m.name)}</div>
        <table>
          <tr><th>System</th><td>${esc(m.system_type)} · ${esc(m.view)} view · L = ${m.n_sites}</td></tr>
          <tr><th>Bias window</th><td>0 to ${num(m.bias_cutoff_mev, 1)} meV · observable ${esc(m.observable)}</td></tr>
          <tr><th>Trained on</th><td>${m.n_training_chains ?? "—"} simulated chains · ${esc(m.model)} (${esc(m.preset)})</td></tr>
          <tr><th>Held-out MAE</th><td class="num">${num(m.test_mae_mev)} meV</td></tr>
          <tr><th>Couplings</th><td>${m.parameters.map((p) =>
            `${esc(p.name)}${p.trained_range_mev ? ` <span class="hint">[${p.trained_range_mev.join(", ")}]</span>` : ""}`).join(" · ")}</td></tr>
          <tr><th>Must match exactly</th><td>${conditionsText(m.fixed_conditions)}</td></tr>
        </table>
        ${m.has_model_card ? `<button data-card="${esc(m.name)}">Read the model card</button>` : ""}
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

// --- train ------------------------------------------------------------------

api("/api/examples").then(({ configs }) => {
  el("train-example").innerHTML = configs.map((c) =>
    `<option value="${esc(c.path)}">${esc(c.name)}${c.description ? ` — ${esc(c.description)}` : ""}</option>`).join("");
}).catch(() => {});

el("train-load").addEventListener("click", async () => {
  try {
    const { text } = await api(`/api/config?path=${encodeURIComponent(el("train-example").value)}`);
    el("train-text").value = text;
    el("train-out").innerHTML = `<div class="box">Loaded. Edit it, then <b>Plan</b> to see what it would do.</div>`;
  } catch (e) { showError(el("train-out"), e); }
});

el("train-save").addEventListener("click", async () => {
  const out = el("train-out");
  try {
    await api("/api/save-config", {
      path: el("train-example").value, text: el("train-text").value,
    });
    out.innerHTML = `<div class="box">Saved and validated.</div>`;
  } catch (e) { showError(out, e); }
});

el("train-plan").addEventListener("click", async () => {
  const out = el("train-out");
  busy(out, "Planning (nothing is written)…");
  try {
    const plan = await api("/api/plan", { config_path: el("train-example").value });
    const stages = plan.stages || [];
    const outputs = plan.outputs || [];
    const budget = plan.budget || {};
    out.innerHTML = `
      <div class="box">
        <h3>Stages</h3>
        <ul class="checks">${stages.map((s) => `<li>${esc(typeof s === "string" ? s : JSON.stringify(s))}</li>`).join("")}</ul>
        ${Object.keys(budget).length ? `<h3>Compute</h3><table>${Object.entries(budget).map(
          ([k, v]) => `<tr><th>${esc(k.replace(/_/g, " "))}</th><td class="num">${esc(v)}</td></tr>`).join("")}</table>` : ""}
        <h3>Files it would write</h3>
        <table><tr><th>Status</th><th>Path</th><th>What</th></tr>
        ${outputs.map((o) => `<tr>
          <td>${esc(o.status || "")}</td>
          <td><code>${esc(o.path || "")}</code></td>
          <td>${esc(o.description || "")}</td></tr>`).join("")}</table>
        ${plan.refusals && plan.refusals.length
          ? `<ul class="checks">${plan.refusals.map((r) => `<li class="fail">${esc(r)}</li>`).join("")}</ul>` : ""}
      </div>`;
  } catch (e) { showError(out, e); }
});

el("train-run").addEventListener("click", async () => {
  if (!confirm("Generation and training can take hours. Start it now?")) return;
  try {
    await api("/api/run-project", { config_path: el("train-example").value });
    activate("jobs"); refreshJobs();
  } catch (e) { showError(el("train-out"), e); }
});

// --- dmi --------------------------------------------------------------------

el("dmi-preview").addEventListener("click", async () => {
  const out = el("dmi-out");
  busy(out, "Checking the symmetry rule…");
  try {
    const d = await api("/api/screening-preview", { config_path: el("dmi-path").value.trim() });
    const hopeless = d.n_candidates - d.n_can_break_symmetry;
    out.innerHTML = `
      <div class="box">
        <div class="verdict ${d.n_can_break_symmetry ? "good" : "bad"}">
          ${d.n_can_break_symmetry} of ${d.n_candidates} candidate(s) can break the symmetry
        </div>
        ${hopeless ? `<p class="hint">${hopeless} cannot, and will be skipped without simulating —
          they cannot constrain D_z however good the data is.</p>` : ""}
        <table><tr><th>Design</th><th>Impurities</th><th>Field</th><th>Can expose DMI?</th></tr>
        ${d.candidates.map((c) => `<tr>
          <td>${esc(c.label)}</td>
          <td>${c.impurities.map((i) => `site ${i.site} ${esc(i.spin)} E=${i.transverse_mev}`).join("<br>") || "none"}</td>
          <td class="num">${c.transverse_field_mev || 0} meV</td>
          <td>${c.breaks_symmetry ? "<b style='color:var(--good)'>yes</b>"
            : "<span class='pill hidden'>no</span>"}</td></tr>`).join("")}
        </table>
      </div>`;
  } catch (e) { showError(out, e); }
});

el("dmi-run").addEventListener("click", async () => {
  try {
    await api("/api/run-screening", { config_path: el("dmi-path").value.trim() });
    activate("jobs"); refreshJobs();
  } catch (e) { showError(el("dmi-out"), e); }
});

// --- jobs -------------------------------------------------------------------

function jobBlock(job) {
  const results = Array.isArray(job.result) ? job.result : null;
  return `<div class="box">
    <div class="row" style="justify-content:space-between">
      <b>${esc(job.label)}</b>
      <span><span class="pill ${esc(job.status)}">${esc(job.status)}</span>
        <span class="hint">${job.elapsed_seconds}s</span></span>
    </div>
    ${job.error ? `<div class="error">${esc(job.error)}</div>` : ""}
    ${results ? `<table><tr><th>Design</th><th class="num">Imprint</th><th>Verdict</th></tr>
      ${results.map((r) => `<tr><td>${esc(r.label)}</td>
        <td class="num">${r.imprint.toExponential(3)}</td>
        <td><span class="pill ${esc(r.verdict)}">${esc(r.verdict)}</span></td></tr>`).join("")}</table>` : ""}
    ${job.lines.length ? `<pre class="log">${esc(job.lines.join("\n"))}</pre>` : ""}
  </div>`;
}

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
