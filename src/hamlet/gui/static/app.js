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
    const { models } = await api("/api/models");
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
  const header = spec.coupling_mode === "single_range"
    ? "<tr><th>coupling</th><th class='num'>from [meV]</th><th class='num'>to [meV]</th><th></th></tr>"
    : "<tr><th>coupling</th><th class='num'>from [meV]</th><th class='num'>to [meV]</th><th></th></tr>";
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
    b.addEventListener("click", () => { b.closest("tr").remove(); }));
}

function readImpurities() {
  return [...el("f-impurities").querySelectorAll("tbody tr")].map((tr) => ({
    site: Number(tr.querySelector(".imp-site").value),
    spin: tr.querySelector(".imp-spin").value,
    transverse_mev: Number(tr.querySelector(".imp-transverse").value),
    axial_mev: Number(tr.querySelector(".imp-axial").value),
  }));
}

function renderModelOptions() {
  const spec = modelSpec();
  const box = el("f-model-options");
  if (!spec.options.length) {
    box.innerHTML = `<div class="box hint">${esc(spec.title)} has no
      hyperparameters exposed here; the defaults are used.</div>`;
    return;
  }
  box.innerHTML = `<div class="box"><p class="hint">Leave these alone unless you
    have a reason. The defaults are what the published models used.</p>` +
    spec.options.map((o) => `<div class="row">
      <label>${esc(o.label)}
        <input type="number" data-option="${esc(o.name)}" value="${o.default}"
          step="${o.type === "integer" ? 1 : "any"}" style="width:9em">
      </label>
      <span class="hint">default ${o.default}</span>
    </div>`).join("") + "</div>";
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
}

function readForm() {
  const spec = systemSpec();
  const rows = [...el("f-couplings").querySelectorAll("input[data-range]")];
  const ranges = spec.couplings.map((_, i) => [
    Number(rows.find((r) => +r.dataset.range === i && r.dataset.edge === "low").value),
    Number(rows.find((r) => +r.dataset.range === i && r.dataset.edge === "high").value),
  ]);
  const options = {};
  el("f-model-options").querySelectorAll("input[data-option]").forEach((i) => {
    options[i.dataset.option] = Number(i.value);
  });
  const form = {
    name: el("f-name").value.trim() || spec.title,
    system_type: chosenSystem,
    n_sites: Number(el("f-n-sites").value),
    n_samples: Number(el("f-n-samples").value),
    coupling_ranges_mev: ranges,
    bias_range_mev: [Number(el("f-bias-lo").value), Number(el("f-bias-hi").value)],
    bias_points: Number(el("f-bias-points").value),
    broadening_mev: Number(el("f-broadening").value),
    observable: el("f-observable").value,
    cutoff_mev: Number(el("f-cutoff").value),
    output_points: Number(el("f-output-points").value),
    model: chosenModel,
    preset: el("f-preset").value,
    model_options: options,
  };
  if (spec.supports_impurities) {
    form.impurities = readImpurities();
    form.transverse_field_mev = Number(el("f-field").value);
  }
  return form;
}

function sampleSvg(bias, sites) {
  const w = 900, h = 190, padL = 44, padR = 10, padT = 10, padB = 26;
  const flat = sites.flat().filter(Number.isFinite);
  const lo = Math.min(...flat), hi = Math.max(...flat), span = (hi - lo) || 1;
  const x = (i) => padL + (i / (bias.length - 1)) * (w - padL - padR);
  const y = (v) => padT + (1 - (v - lo) / span) * (h - padT - padB);
  const paths = sites.map((row, index) => {
    const hue = Math.round((index / Math.max(sites.length, 1)) * 300);
    return `<path d="${row.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("")}"
      fill="none" stroke="hsl(${hue} 62% 48%)" stroke-width="1.3"/>`;
  }).join("");
  const ticks = [0, 0.5, 1].map((f) => {
    const i = Math.round(f * (bias.length - 1));
    return `<text x="${x(i).toFixed(1)}" y="${h - 8}" font-size="11" text-anchor="middle"
      fill="currentColor" opacity=".65">${bias[i].toFixed(1)} meV</text>`;
  }).join("");
  return `<svg class="spectra" style="height:190px" viewBox="0 0 ${w} ${h}"
    preserveAspectRatio="none">${paths}${ticks}</svg>`;
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
  busy(out, "Simulating sample chains… this takes about a minute each.");
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
          ${sampleSvg(r.bias_mev, sample.sites)}`).join("") +
          `<p class="hint">
             ${r.showing_all_sites
               ? `One line per site, all ${r.n_sites} of them.`
               : `One line per site, showing sites ${r.evaluated_sites.join(", ")}
                  of ${r.n_sites} — a subset, because cost scales with the
                  number of sites evaluated, not with the bias resolution.`}
             Simulated with ${esc(r.dynamics_mode)}${
               r.dynamics_mode === "DMRG"
                 ? " (approximate; the full run uses the same setting)"
                 : " (exact)"} at a basis size of ${r.hilbert_dimension}.
             If the features sit at the very edge of the window, or look like
             smooth bumps with no structure, adjust the bias range or the
             broadening before running.</p></div>`;
      },
      (job) => { if (job.lines.length) busy(out, job.lines[job.lines.length - 1]); });
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
        <p class="hint">Your answers were saved as a configuration, so this run
          can be repeated or sent to a cluster with:<br>
          <code>hamlet run ${esc(builtConfig.config_path)}</code></p>
      </div>`;
    el("f-run-zone").hidden = (plan.blocking_issues || []).length > 0;
  } catch (e) { showError(out, e); builtConfig = null; }
});

el("f-run").addEventListener("click", async () => {
  if (!builtConfig) return;
  if (!confirm("Generation and training can take hours. Start it now?")) return;
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

el("f-toggle-advanced").addEventListener("click", () => {
  const box = el("f-model-options");
  box.hidden = !box.hidden;
});

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

  selectSystem(options.systems[0].system_type);
}).catch((e) => showError(el("sys-cards"), e));

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
