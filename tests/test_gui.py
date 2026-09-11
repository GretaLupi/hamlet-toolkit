"""The browser interface.

It exists because knowing *which* command a situation calls for is the hard
part of the package, so these tests check the guidance and contract reporting
as much as the plumbing. The API layer is deliberately HTTP-free so it can be
tested directly; a live server covers the routes and the static assets.
"""

import json
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hamlet.gui import api
from hamlet.gui.server import STATIC_ROOT, _Handler

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def measurement_csv(tmp_path_factory):
    """An L=8 measurement over 0-20 meV, as an experimentalist would supply."""
    path = tmp_path_factory.mktemp("gui") / "chain.csv"
    bias = np.linspace(0.0, 20.0, 81)
    rows = []
    for site, centre in enumerate(np.linspace(4.0, 12.0, 8), start=1):
        spectrum = np.cumsum(np.exp(-((bias - centre) ** 2) / 2.0))
        rows.extend(
            {"site": site, "bias_meV": b, "didv_A": v} for b, v in zip(bias, spectrum)
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def server():
    handler = type("TestHandler", (_Handler,), {"registry": api.JobRegistry()})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(base, route):
    try:
        with urllib.request.urlopen(base + route) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def post(base, route, payload):
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def wait_for_job(registry, job_id, timeout=20.0):
    deadline = threading.Event()
    waited = 0.0
    while waited < timeout:
        job = registry.get(job_id)
        if job is not None and job.status != "running":
            return job.to_dict()
        deadline.wait(0.05)
        waited += 0.05
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --- guidance ---------------------------------------------------------------

def test_overview_leads_with_the_users_situation_not_the_command_name():
    """The landing content has to answer "which of these am I doing?"."""
    situations = api.workflow_overview()["situations"]
    assert len(situations) >= 5
    for entry in situations:
        # Each must say what the user has, what happens, and what it costs --
        # the three things missing from a bare list of commands.
        for key in ("id", "have", "does", "needs", "cost"):
            assert entry[key], f"{entry.get('id')} is missing {key}"
        assert entry["have"].lower().startswith("i ")
    assert {"inspect", "advise", "models", "train", "dmi"} <= {e["id"] for e in situations}


# --- models -----------------------------------------------------------------

def test_published_models_report_the_contract_that_governs_reuse():
    payload = api.describe_available_models()
    models = {entry["name"]: entry for entry in payload["models"]}
    assert models, "no published models were found"
    for entry in models.values():
        assert "error" not in entry, entry
        for key in ("system_type", "view", "n_sites", "bias_cutoff_mev"):
            assert entry[key] is not None, f"{entry['name']} does not report {key}"

    # The impurity model's extra fixed conditions must be surfaced, since
    # system_type alone does not identify them and a mismatch is catastrophic.
    impurity = next(entry for name, entry in models.items() if "dmi_impurity" in name)
    assert impurity["fixed_conditions"], "impurity conditions are not shown"
    assert impurity["fixed_conditions"]["impurity_sites"] == [1, 4, 6]


def test_model_card_is_readable_and_cannot_escape_the_model_directory():
    name = "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
    assert "Model card" in api.read_model_card(name)["markdown"]
    for attempt in ("../../pyproject.toml", "/etc/passwd", "nope"):
        with pytest.raises(FileNotFoundError):
            api.read_model_card(attempt)


# --- experiments ------------------------------------------------------------

def test_inspect_reports_the_fields_that_decide_reuse(measurement_csv):
    result = api.inspect_experiment(measurement_csv)
    assert result["n_sites"] == 8
    assert result["n_bias_points"] == 81
    assert result["bias_min_mev"] == pytest.approx(0.0)
    assert result["bias_max_mev"] == pytest.approx(20.0)
    assert result["is_complete"] and result["starts_at_zero"]
    assert result["missing_points"] == 0
    assert len(result["plot"]["sites"]) == 8


def test_inspect_thins_large_maps_for_transport(measurement_csv):
    result = api.inspect_experiment(measurement_csv, max_points=20)
    assert result["n_bias_points"] == 81, "the reported count must be the real one"
    assert len(result["plot"]["bias_mev"]) == 20, "the plotted series must be thinned"


def test_inspect_imports_a_folder_of_per_site_nanonis_files(monkeypatch, tmp_path):
    raw = tmp_path / "raw-chain"
    raw.mkdir()
    header = "Bias calc (V)\tLI Demod 1 X (A)\tLI Demod 2 X (A)"
    for name, scale in (("Chain1_10.dat", 10), ("Chain1_2.dat", 2)):
        (raw / name).write_text(
            "Experiment\tchain\n[DATA]\n" + header + "\n"
            + "\n".join(
                f"{bias}\t{scale * (index + 1)}\t{scale * (index + 1) * 0.1}"
                for index, bias in enumerate((0.01, 0.0, -0.01))
            )
            + "\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(api, "_experiment_root", lambda: tmp_path / "imports")

    result = api.inspect_experiment(raw)

    assert result["input"]["input_kind"] == "sts_folder"
    assert result["input"]["source_file_count"] == 2
    assert result["input"]["source_files"] == ["Chain1_2.dat", "Chain1_10.dat"]
    assert result["input"]["auxiliary_d2idv2"] is True
    assert result["n_sites"] == 2
    assert result["bias_min_mev"] == pytest.approx(-10.0)
    assert result["bias_max_mev"] == pytest.approx(10.0)
    assert set(result["plots"]) == {"didv", "d2idv2"}
    assert result["plots"]["d2idv2"]["role"] == "plot/QC only"
    assert Path(result["path"]).name == "measurement.npz"
    assert Path(result["path"]).exists()
    assert Path(result["input"]["import_report"]).exists()
    # An unchanged folder reuses its canonical import.
    assert api.inspect_experiment(raw)["path"] == result["path"]


def test_folder_import_explains_when_the_export_is_not_recognised(monkeypatch, tmp_path):
    raw = tmp_path / "another-lab"
    raw.mkdir()
    (raw / "site-1.txt").write_text("bias,signal\n0,1\n")
    monkeypatch.setattr(api, "_experiment_root", lambda: tmp_path / "imports")
    with pytest.raises(ValueError, match="Nanonis-style STS"):
        api.inspect_experiment(raw)


def test_advisor_refuses_the_impurity_model_without_declared_conditions(measurement_csv):
    """The interface must not become a way around the condition contract."""
    decision = api.advise_for_experiment(
        measurement_csv,
        20.0,
        system_type="homogeneous_xxz_j1j2j3_dmi_impurity",
        view="global",
    )
    assert not decision["can_use_existing_model"]
    impurity = next(a for a in decision["artifacts"] if "dmi_impurity" in a["path"])
    assert any("has not declared" in reason for reason in impurity["reasons"])


# --- projects and screening -------------------------------------------------

def test_planning_a_project_writes_nothing():
    config = REPO_ROOT / "examples" / "heisenberg_xxz_dmi_impurities_l8.yaml"
    before = set((REPO_ROOT / "examples").rglob("*"))
    plan = api.plan_project(config)
    assert plan["outputs"], "a plan with no outputs is not a plan"
    assert set((REPO_ROOT / "examples").rglob("*")) == before


def test_saving_an_invalid_config_is_refused_before_it_is_written(tmp_path):
    """A user must not be able to save a file the workflow will later reject."""
    path = tmp_path / "config.yaml"
    original = "name: keep me\n"
    path.write_text(original)
    with pytest.raises(ValueError, match="not valid"):
        api.write_config_text(path, "system_type: nonsense\ndataset: {}\n")
    assert path.read_text() == original, "the original was clobbered"
    assert not list(tmp_path.glob("*.checking")), "a temporary file was left behind"


def test_screening_preview_reports_the_free_symmetry_verdict():
    result = api.screening_preview(REPO_ROOT / "examples" / "dmi_screening.yaml")
    assert result["n_candidates"] == 6
    assert result["n_can_break_symmetry"] == 4
    hopeless = [c for c in result["candidates"] if not c["breaks_symmetry"]]
    assert len(hopeless) == 2
    assert all(len(c["impurities"]) == 1 for c in hopeless)


# --- background jobs --------------------------------------------------------

def test_jobs_run_off_the_request_thread_and_capture_their_output():
    registry = api.JobRegistry()

    def work():
        print("line one")
        print("line two")
        return {"value": 42}

    job = registry.submit("test", "a job", work)
    done = wait_for_job(registry, job.job_id)
    assert done["status"] == "finished"
    assert done["lines"] == ["line one", "line two"]
    assert done["result"] == {"value": 42}


def test_a_failing_job_is_reported_not_swallowed():
    registry = api.JobRegistry()

    def work():
        print("before the problem")
        raise RuntimeError("the simulator fell over")

    job = registry.submit("test", "doomed", work)
    done = wait_for_job(registry, job.job_id)
    assert done["status"] == "failed"
    assert "the simulator fell over" in done["error"]
    # Output produced before the failure, and a traceback, must both survive.
    assert "before the problem" in done["lines"]
    assert any("RuntimeError" in line for line in done["lines"])


def test_running_jobs_are_listed_newest_first():
    registry = api.JobRegistry()
    first = registry.submit("test", "older", lambda: None)
    second = registry.submit("test", "newer", lambda: None)
    wait_for_job(registry, first.job_id)
    wait_for_job(registry, second.job_id)
    listed = [job["label"] for job in registry.list()]
    assert listed[0] == "newer"


# --- http layer -------------------------------------------------------------

@pytest.mark.parametrize("route", ["/", "/style.css", "/app.js", "/quotes.js"])
def test_static_assets_are_served(server, route):
    status, body = get(server, route)
    assert status == 200
    assert len(body) > 500


@pytest.mark.parametrize(
    "route", ["/api/overview", "/api/models", "/api/examples", "/api/jobs"]
)
def test_read_only_routes_return_json(server, route):
    status, body = get(server, route)
    assert status == 200
    assert isinstance(json.loads(body), dict)


@pytest.mark.parametrize(
    "route", ["/../pyproject.toml", "/static/../../project.py", "/..%2Fpyproject.toml"]
)
def test_path_traversal_is_refused(server, route):
    """The server reads local files, so escaping the asset root must fail."""
    status, _ = get(server, route)
    assert status == 404


def test_missing_fields_produce_a_useful_error_not_a_crash(server):
    status, payload = post(server, "/api/plan", {})
    assert status == 400
    assert "config_path" in payload["error"]


def test_unknown_routes_are_rejected(server):
    status, payload = post(server, "/api/not-a-thing", {})
    assert status == 404
    assert "error" in payload


def test_index_mentions_every_panel_the_script_drives():
    """A tab whose panel is missing would render as a blank screen."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    for panel in ("start", "data", "reuse", "models", "train", "dmi", "jobs"):
        assert f'data-panel="{panel}"' in html, f"no tab for {panel}"
        assert f'id="panel-{panel}"' in html, f"no panel for {panel}"
    # Every element the script fetches by id must exist in the page.
    for match in sorted(set(re.findall(r'el\("([a-z0-9-]+)"\)', script))):
        assert f'id="{match}"' in html, f"app.js drives #{match}, missing from the page"


def test_a_shakespeare_quote_keeps_the_user_company_while_a_job_runs():
    """The waits here run to hours, so the page offers a line for them."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    quotes = (STATIC_ROOT / "quotes.js").read_text(encoding="utf-8")
    style = (STATIC_ROOT / "style.css").read_text(encoding="utf-8")

    # The bank has to load first, or the first render calls a function that
    # does not exist yet and takes the panel down with it.
    assert html.index('src="/quotes.js"') < html.index('src="/app.js"')
    assert "function waitingQuote" in quotes
    # Every surface that holds the user through a wait, not just one of them:
    # the jobs list, the analysis, and the sample simulation.
    assert script.count("quoteFor(job.job_id, job.elapsed_seconds)") == 3
    assert ".wait-quote" in style


def test_a_missing_quote_cannot_take_the_jobs_page_down():
    """Decoration must not be load-bearing.

    `waitingQuote` lives in its own file. If that file does not arrive -- an
    older install, a stale cache -- calling it throws from inside jobBlock(),
    which propagates to refreshJobs(), whose catch reports the server as
    stopped. A missing quote would then black out the jobs page and claim the
    run had died.
    """
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "function quoteFor" in script
    # The guard, not the raw function, is what the render paths call.
    assert "waitingQuote(job.job_id" not in script
    guard = script[script.index("function quoteFor") :][:600]
    assert 'typeof waitingQuote === "function"' in guard
    assert "catch" in guard


def test_one_job_shows_one_quote_across_every_surface():
    """The surfaces do not share a poll, so they must share the choice.

    The jobs list refreshes every 3s and a job's own page every 2s. Deriving
    the line from the elapsed time sampled at render made the two disagree
    whenever their polls fell either side of a rotation, and the same run
    quoted two different lines on two pages at once.
    """
    quotes = (STATIC_ROOT / "quotes.js").read_text(encoding="utf-8")

    assert "const showing = new Map()" in quotes
    body = quotes[quotes.index("function waitingQuote") :]
    # Rotation is decided on a stored timestamp, not recomputed from elapsed.
    assert "showing.get(key)" in body and "showing.set(key" in body
    assert "state.since" in body
    assert "Math.floor(elapsed / QUOTE_ROTATE_SECONDS)" not in body


def test_the_quote_tiers_stay_in_order():
    """Each tier's pool has to be a prefix of the whole bank.

    The rotation holds an index, and the pool grows as a wait crosses into a
    later tier. That index only keeps pointing at the same line if the gated
    quotes all sit after the ungated ones -- insert one in the middle and a
    long wait would jump to an unrelated line at three minutes.
    """
    quotes = (STATIC_ROOT / "quotes.js").read_text(encoding="utf-8")
    bank = quotes[quotes.index("const SHAKESPEARE_QUOTES") : quotes.index("const showing")]
    gates = []
    for block in re.findall(r"\{(.*?)\}", bank, re.S):
        if 'source: "' not in block:
            continue
        match = re.search(r"after: (\d+)", block)
        gates.append(int(match.group(1)) if match else 0)

    assert gates, "no quotes found"
    assert gates == sorted(gates), (
        "gated quotes must come after ungated ones, or the rotation jumps "
        "when a wait crosses a tier"
    )


def test_a_running_job_always_gets_a_line():
    """No opt-out, and no waiting period: if it is running, it is quoting.

    The preference and its checkbox are gone. What remains has to have no way
    to return an empty string for a running job, or "always" would depend on
    browser storage that can come back empty anyway.
    """
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    quotes = (STATIC_ROOT / "quotes.js").read_text(encoding="utf-8")

    assert 'id="quotes-on"' not in html
    assert "localStorage" not in quotes, "a stored preference can still hide them"
    assert "data-quote-hide" not in quotes
    assert "QUOTE_AFTER_SECONDS" not in quotes, "a threshold is a wait with no quote"
    body = quotes[quotes.index("function waitingQuote") :]
    assert 'return ""' not in body, "waitingQuote must always produce a line"


def test_static_assets_are_served_uncached():
    """A cached index.html against a new bundle is a page from two versions."""
    server_source = (
        Path(__file__).resolve().parents[1] / "src" / "hamlet" / "gui" / "server.py"
    ).read_text(encoding="utf-8")
    assert "no-store" in server_source


def test_the_gpu_device_card_says_when_this_machine_cannot_honour_it():
    """Selecting a GPU on a machine with none must not look like it worked."""
    api_source = (
        Path(__file__).resolve().parents[1] / "src" / "hamlet" / "gui" / "api.py"
    ).read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    assert "unavailable_here" in api_source
    assert "gpu_help_url" in api_source
    assert "unavailable_here" in script
    # Warned, not forbidden: a config built here may be bound for a cluster.
    assert 'id="device-warning"' in html
    assert "will train on the CPU" in script


def test_every_quote_carries_its_attribution():
    """An unattributed line is a misquotation waiting to happen."""
    quotes = (STATIC_ROOT / "quotes.js").read_text(encoding="utf-8")
    bank = quotes[quotes.index("const SHAKESPEARE_QUOTES") : quotes.index("const showing")]
    lines = re.findall(r"line:\s", bank)
    sources = re.findall(r'source: "([^"]+)"', bank)
    assert len(lines) == len(sources), "a quote is missing its source"
    assert len(sources) >= 20
    for source in sources:
        # "Play, act.scene", or the Induction that The Shrew has instead of a
        # first act. A reader who wants to check a line has to be able to.
        assert re.fullmatch(
            r"[A-Za-z' ]+, (?:[IVX]+\.[ivx]+|Induction [ivx]+)", source
        ), source


def test_long_explanations_are_behind_an_info_toggle():
    """Eight paragraphs open at once is a wall, not a form.

    Every hyperparameter now carries an explanation, which is the point --
    but shown unconditionally they bury the fields they describe, and someone
    who already knows what dropout does should not scroll past a paragraph
    saying so.
    """
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    style = (STATIC_ROOT / "style.css").read_text(encoding="utf-8")

    assert "function infoButton" in script
    assert 'class="info-body"' in script or "info-body" in script
    # The rows are rebuilt when the model changes, so the handler has to be
    # delegated rather than bound per button.
    assert 'target.closest("[data-info]")' in script
    assert ".info-body[hidden]" in style

    # The two longest form explanations start collapsed.
    for target in ("f-workers-hint", "f-spin-hint"):
        assert f'id="{target}" hidden' in html, f"{target} is shown unconditionally"
        assert f'data-info="{target}"' in html, f"{target} has no way to open it"


def test_no_unconditional_wall_of_text_is_left_in_the_page():
    """A long block that is not behind a toggle has to earn its place."""
    import re as _re

    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    # Measure only what is actually on screen: a container whose detail is
    # already folded away is not a wall of text, however long its source.
    visible = _re.sub(
        r'<div class="info-body"[^>]*hidden[^>]*>.*?</div>', "", html, flags=_re.S
    )
    offenders = []
    for match in _re.finditer(r"<(p|div)\b([^>]*)>(.*?)</\1>", visible, _re.S):
        attributes, body = match.group(2), match.group(3)
        if "hidden" in attributes or "info-body" in attributes:
            continue
        plain = " ".join(_re.sub(r"<[^>]+>", "", body).split())
        if len(plain) > 340:
            offenders.append(plain[:70])
    assert not offenders, (
        f"long blocks shown unconditionally: {offenders}. Put the detail "
        f"behind an (i) and leave the first sentence visible."
    )


def test_every_network_setting_explains_itself():
    """The audience measures spectra; they do not tune networks for a living.

    A field labelled "Huber delta" with no explanation is a field nobody
    touches, or worse, one somebody changes at random. Each says what it does
    and which way to move it.
    """
    from hamlet.gui import api

    for model in api.describe_builder_options()["models"]:
        if not model["name"].startswith("keras"):
            continue
        for option in model["options"]:
            hint = option.get("hint", "")
            assert hint, f"{model['name']}.{option['name']} has no explanation"
            assert len(hint) > 60, (
                f"{model['name']}.{option['name']} restates its label rather "
                f"than explaining it: {hint!r}"
            )


# --- stopping a run, changing a number, running it again --------------------

def test_the_same_settings_return_to_the_same_run_directory(tmp_path):
    """So a stopped run resumes instead of starting its generation over."""
    from hamlet.gui import api

    first = api.build_project_config(_builder_form(), workspace=tmp_path)
    second = api.build_project_config(_builder_form(), workspace=tmp_path)
    assert first["run_dir"] == second["run_dir"]
    assert first["fingerprint"] == second["fingerprint"]


def test_tweaked_settings_get_their_own_run_directory(tmp_path):
    """A run refuses a directory holding a different configuration.

    Sending every run of a project to one `run/` meant that stopping a job,
    changing a number and starting again hit that refusal and asked the user
    to invent a new name -- for the most ordinary action there is.
    """
    from hamlet.gui import api

    base = api.build_project_config(_builder_form(), workspace=tmp_path)
    tweaks = [
        {"n_samples": 40},
        {"cutoff_mev": 18.0},
        {"n_sites": 10},
        {"broadening_mev": 0.4},
        # A different model, with hyperparameters that belong to it.
        {"model": "random_forest", "model_options": {"n_estimators": 200}},
    ]
    for tweak in tweaks:
        field = ", ".join(tweak)
        tweaked = api.build_project_config(_builder_form(**tweak), workspace=tmp_path)
        assert tweaked["run_dir"] != base["run_dir"], f"{field} reused the directory"
        assert tweaked["config_path"] != base["config_path"]


def test_a_run_directory_with_work_in_it_is_reported_as_continuing(tmp_path):
    """Existing-but-empty is not resuming: the directory is made before the run."""
    from hamlet.gui import api

    built = api.build_project_config(_builder_form(), workspace=tmp_path)
    assert built["resuming"] is False

    run_dir = Path(built["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    assert api.build_project_config(_builder_form(), workspace=tmp_path)["resuming"] is False

    (run_dir / "resolved_project_config.json").write_text("{}", encoding="utf-8")
    again = api.build_project_config(_builder_form(), workspace=tmp_path)
    assert again["resuming"] is True


def test_the_field_order_of_a_form_does_not_change_the_run(tmp_path):
    """The fingerprint names the settings, not the order they arrived in."""
    from hamlet.gui import api

    form = _builder_form()
    shuffled = {key: form[key] for key in reversed(list(form))}
    assert (
        api.build_project_config(form, workspace=tmp_path)["fingerprint"]
        == api.build_project_config(shuffled, workspace=tmp_path)["fingerprint"]
    )


def test_the_page_says_whether_a_run_is_new_or_continuing():
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "builtConfig.resuming" in script
    assert "Continuing the run" in script
    assert "A new run, in its own folder" in script
    # And starting one while another is going says what that costs.
    assert "share the same cores" in script


# --- the cluster, reduced to the two facts a user has ------------------------

def test_the_cluster_form_needs_only_an_address_and_a_batch_system(tmp_path, monkeypatch):
    """Everything else has a default or is a resource request.

    The scheduler profiles behind these names carry a dozen directive
    templates each, but none of that is a decision anyone makes: picking
    "slurm" is the decision and the profile follows from it.
    """
    import yaml

    from hamlet.gui import api

    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    saved = api.build_cluster_config({
        "host": "you@cluster.example.edu",
        "remote_dir": "/scratch/work/you/hamlet",
        "scheduler": "slurm",
    })
    written = yaml.safe_load(Path(saved["path"]).read_text(encoding="utf-8"))
    assert written["host"] == "you@cluster.example.edu"
    assert written["scheduler"] == "slurm"
    # Nothing invented on the user's behalf.
    assert "resources" not in written
    assert "setup" not in written


def test_the_cluster_form_round_trips_what_it_saved(tmp_path, monkeypatch):
    from hamlet.gui import api

    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    api.build_cluster_config({
        "host": "you@cluster.example.edu",
        "remote_dir": "/scratch/work/you/hamlet",
        "scheduler": "pbs",
        "cpus": 16,
        "walltime": "12:00:00",
        "setup": "module load python/3.11\nsource ~/venvs/hamlet/bin/activate",
    })
    form = api.read_cluster_form()["form"]
    assert form["host"] == "you@cluster.example.edu"
    assert form["scheduler"] == "pbs"
    assert form["cpus"] == 16
    assert form["setup"].splitlines() == [
        "module load python/3.11",
        "source ~/venvs/hamlet/bin/activate",
    ]


@pytest.mark.parametrize(
    "form, message",
    [
        ({"remote_dir": "/x", "scheduler": "slurm"}, "ssh to"),
        ({"host": "you@x", "scheduler": "slurm"}, "directory on the cluster"),
        ({"host": "you@x", "remote_dir": "/x", "scheduler": "condor"}, "scheduler must be"),
        ({"host": "a b", "remote_dir": "/x", "scheduler": "slurm"}, "ssh address"),
    ],
)
def test_the_cluster_form_refuses_what_cannot_work(tmp_path, monkeypatch, form, message):
    from hamlet.gui import api

    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    with pytest.raises(ValueError, match=message):
        api.build_cluster_config(form)


def test_a_hand_written_scheduler_block_is_reported_not_misshown(tmp_path, monkeypatch):
    """A dropdown cannot represent a custom block, so it says so."""
    from hamlet.gui import api

    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    api.write_cluster_config(
        "cluster_schema_version: 1\n"
        "host: you@x\n"
        "remote_dir: /x\n"
        "scheduler:\n"
        "  name: custom\n"
        "  submit_command: [qsub]\n"
        "  directive_prefix: '#PBS'\n"
    )
    form = api.read_cluster_form()["form"]
    assert form["custom_scheduler"] is True
    assert form["scheduler"] == "slurm", "a custom block must not be mislabelled"


def test_the_page_no_longer_asks_anyone_to_write_cluster_yaml():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="cl-host"' in html and 'id="cl-scheduler"' in html
    # The raw editor and the profile table are gone from the normal path.
    assert 'id="cluster-text"' not in html
    assert 'id="scheduler-out"' not in html
    assert "cluster-text" not in script
    # And the page states the access requirement up front.
    assert "ssh-copy-id" in html


def test_the_page_says_where_it_writes_things(server):
    """The workspace is not in the same place for everyone.

    Beside a checkout, under the home directory for an installed package, or
    wherever HAMLET_WORKSPACE points. Someone who does not know which case
    they are in cannot find their own results, so the front page says.
    """
    status, body = get(server, "/api/locations")
    assert status == 200
    info = json.loads(body)

    assert info["workspace"]
    assert info["explanation"], "the reason for this location is not explained"
    titles = {item["title"] for item in info["locations"]}
    assert {"Uploads", "Projects", "Analyses"} <= titles
    for item in info["locations"]:
        assert item["purpose"], f"{item['title']} has no description"
        assert item["path"].startswith(info["workspace"])

    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'id="locations-out"' in html
    assert "loadLocations()" in script


def test_every_output_location_is_described(tmp_path, monkeypatch):
    """A folder nobody can explain should not be appearing on disk."""
    from hamlet.gui import api

    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    info = api.describe_output_locations()
    assert info["override"] == str(tmp_path)
    assert "HAMLET_WORKSPACE" in info["explanation"]
    # Nothing has been written yet, which the report states rather than hides.
    assert all(not item["exists"] for item in info["locations"])


def test_parallel_chains_are_offered_in_the_form():
    """Generation is the long stage; the setting that shortens it belongs here."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="f-workers"' in html
    assert "workers:" in script
    # A cleared box must not silently mean "saturate the machine".
    assert 'el("f-workers").value === ""' in script


def test_the_form_writes_parallel_chains_into_the_config(tmp_path):
    """The number has to reach generation, not just sit on the page."""
    import yaml

    from hamlet.gui import api

    built = api.build_project_config(_builder_form(workers=4), workspace=tmp_path)
    config = yaml.safe_load(Path(built["config_path"]).read_text(encoding="utf-8"))
    assert config["dataset"]["generate"]["workers"] == 4

    # One is the default and carries no meaning, so it is left out rather than
    # written into every configuration the form produces.
    plain = api.build_project_config(_builder_form(), workspace=tmp_path)
    config = yaml.safe_load(Path(plain["config_path"]).read_text(encoding="utf-8"))
    assert "workers" not in config["dataset"]["generate"]


def test_a_negative_worker_count_is_refused_by_the_form(tmp_path):
    from hamlet.gui import api

    with pytest.raises(ValueError, match="negative"):
        api.build_project_config(_builder_form(workers=-2), workspace=tmp_path)


def test_local_couplings_are_drawn_as_a_spin_chain_not_generic_bars():
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    style = (STATIC_ROOT / "style.css").read_text(encoding="utf-8")
    assert "function bondChainChart" in script
    assert 'class="chain-site"' in script
    assert 'class="chain-bond' in script
    assert ".bond-value" in style and ".chain-site" in style


def test_bond_strength_is_visible_without_reading_the_numbers():
    """A chain of identical lines hides the result it is drawing.

    Strength is inked as well as thickened, so the weak bonds recede and the
    strong ones carry the eye. Opacity rather than a computed colour, because
    the hue still has to come from CSS to keep meaning sign, and CSS is where
    the light and dark palettes live.
    """
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    start = script.index("function bondChainChart")
    chart = script[start : script.index("function ", start + 20)]

    assert "stroke-opacity:${ink}" in chart
    # Shading spans the couplings present, not zero to the largest. Real
    # chains sit in a narrow band -- 32 to 38 meV is an ordinary result -- and
    # against an absolute scale every bond inks within a tenth of full, so the
    # picture claims "all the same" about a set that varies by a fifth.
    assert "(Math.abs(row.value) - minMagnitude) / magnitudeSpan" in chart
    # A bond must never fade to invisible -- that reads as a broken chain.
    assert "0.3 + 0.7 * strength" in chart
    # Thickness stays absolute, so genuinely equal couplings still look equal.
    assert "Math.abs(row.value) / maxMagnitude" in chart
    # And the key prints the two ends, or full contrast over a 0.1 meV spread
    # would read as a dramatic difference.
    assert "shading spans" in chart
    assert "all bonds equal" in chart


# --- manifests from more than one pipeline ----------------------------------
# Published artifacts come from datasets generated by this package and from
# datasets imported from earlier work. Their manifests record the same facts
# under different keys, so the browser has to read both shapes.

def test_every_published_model_reports_its_model_and_observable():
    """A blank field here means the browser is reading the wrong key."""
    for entry in api.describe_available_models()["models"]:
        assert entry["model"], f"{entry['name']} does not report which model it is"
        assert entry["observable"], f"{entry['name']} does not report its observable"


def test_training_chain_count_is_the_split_total_not_the_per_shard_count():
    """A dataset merged from shards records the per-shard count in its recipe.

    Reading that would advertise a 3000-chain model as trained on 20, so the
    count must come from the split-group totals.
    """
    for entry in api.describe_available_models()["models"]:
        manifest = json.loads(
            (Path(entry["path"]) / "manifest.json").read_text(encoding="utf-8")
        )
        split = manifest.get("metrics", {}).get("split", {})
        expected = sum(
            split[key]
            for key in ("train_groups", "validation_groups", "test_groups")
            if key in split
        )
        if expected:
            assert entry["n_training_chains"] == expected, entry["name"]
            recipe_count = (
                manifest.get("dataset_metadata", {})
                .get("generation_recipe", {})
                .get("n_samples")
            )
            if recipe_count is not None and recipe_count != expected:
                assert entry["n_training_chains"] != recipe_count


def test_validity_ranges_are_shown_even_without_a_sampling_recipe():
    """The imported artifact has no coupling_ranges_mev; the range still shows.

    Falling back to the stored training distribution matters because the range
    is what tells a user whether their couplings are extrapolation.
    """
    models = {m["name"]: m for m in api.describe_available_models()["models"]}
    imported = models["inhomogeneous_heisenberg_l12_keras_mlp_standard_v1"]
    for parameter in imported["parameters"]:
        low, high = parameter["trained_range_mev"]
        assert 29.0 < low < 31.0, parameter
        assert 44.0 < high < 46.0, parameter


# --- starting up -------------------------------------------------------------
# `hamlet gui` failed with a raw bind traceback when a previous interface was
# still holding the port, and the browser was launched on the calling thread
# before serve_forever, so a hanging launcher left the socket listening while
# nothing was ever accepted.

def test_a_busy_default_port_falls_back_instead_of_failing():
    """A second terminal must not fail just because the first one is serving."""
    import socket
    import threading

    from hamlet.gui import server as gui_server

    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    taken = blocker.getsockname()[1]
    blocker.listen(1)

    started = threading.Event()
    chosen: list[int] = []
    real_server = gui_server.ThreadingHTTPServer

    class Recorder(real_server):
        def __init__(self, address, handler):
            super().__init__(address, handler)
            chosen.append(self.server_address[1])
            started.set()

        def serve_forever(self, *args, **kwargs):
            return None

    gui_server.ThreadingHTTPServer = Recorder
    try:
        # port=None means "the default, or the next free one"; point the default
        # at the blocked port so the fallback path is the one exercised.
        original_default = gui_server.DEFAULT_PORT
        gui_server.DEFAULT_PORT = taken
        gui_server.serve(port=None, open_browser=False)
    finally:
        gui_server.DEFAULT_PORT = original_default
        gui_server.ThreadingHTTPServer = real_server
        blocker.close()

    assert chosen, "the server never bound anything"
    assert chosen[0] != taken, "it bound the port that was already taken"
    assert chosen[0] == taken + 1


def test_an_explicitly_chosen_busy_port_is_refused_with_guidance():
    """An explicit port is honoured strictly, and the error has to be actionable.

    The message must not claim what is holding the port: a suspended interface
    keeps its socket bound while answering nothing.
    """
    import socket

    from hamlet.gui.server import serve

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    taken = blocker.getsockname()[1]
    blocker.listen(1)
    try:
        with pytest.raises(SystemExit) as excinfo:
            serve(port=taken, open_browser=False)
    finally:
        blocker.close()
    message = str(excinfo.value)
    assert "not responding" in message
    assert f"--port {taken + 1}" in message
    assert "not a HamLeT interface" not in message


def test_browser_launch_cannot_block_the_server():
    """It runs before serve_forever, so a hanging launcher must not reach it."""
    import threading

    from hamlet.gui import server as gui_server

    entered = threading.Event()
    release = threading.Event()

    def hanging_open(url):
        entered.set()
        release.wait(timeout=10)
        return True

    import webbrowser

    original = webbrowser.open
    webbrowser.open = hanging_open
    try:
        gui_server._open_browser("http://127.0.0.1:1/")
        # If the launch were synchronous, control would not return until the
        # fake browser was released.
        assert entered.wait(timeout=5), "the launcher never ran"
    finally:
        release.set()
        webbrowser.open = original


def test_headless_sessions_are_told_how_to_reach_the_interface(monkeypatch, capsys):
    """On a login node there is no browser; the port-forward hint is the answer."""
    import threading

    from hamlet.gui import server as gui_server

    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    real_server = gui_server.ThreadingHTTPServer

    class Immediate(real_server):
        def serve_forever(self, *args, **kwargs):
            return None

    monkeypatch.setattr(gui_server, "ThreadingHTTPServer", Immediate)
    opened: list[str] = []
    monkeypatch.setattr(gui_server, "_open_browser", lambda url: opened.append(url))

    gui_server.serve(port=0, open_browser=True)
    output = capsys.readouterr().out
    assert not opened, "a browser was launched with no display"
    assert "No display detected" in output
    assert "ssh -N -L" in output
    assert threading.active_count() >= 1


# --- stopping from the browser ----------------------------------------------
# Closing a tab leaves the process running, which is how you end up with an
# interface nobody can see holding a port nobody can reuse.

def test_shutdown_route_stops_the_server_and_reports_lost_work():
    """The reply must arrive before the socket closes, and name running jobs."""
    import threading
    import time

    handler = type("StopHandler", (_Handler,), {"registry": api.JobRegistry()})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    serving = threading.Thread(target=httpd.serve_forever, daemon=True)
    serving.start()

    release = threading.Event()
    handler.registry.submit("test", "a long simulation", lambda: release.wait(30))
    # Let the job reach "running" before asking to stop.
    deadline = time.time() + 5
    while time.time() < deadline and not any(
        job["status"] == "running" for job in handler.registry.list()
    ):
        time.sleep(0.05)

    try:
        status, payload = post(base, "/api/shutdown", {})
        assert status == 200
        assert payload["stopping"] is True
        assert payload["abandoned_jobs"] == ["a long simulation"], (
            "work that will be lost has to be named, not silently dropped"
        )
        serving.join(timeout=10)
        assert not serving.is_alive(), "the serving loop did not exit"
    finally:
        release.set()
        httpd.server_close()

    # The port must actually be free again afterwards.
    import socket

    probe = socket.socket()
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", int(base.rsplit(":", 1)[1])))
    finally:
        probe.close()


def test_the_page_explains_that_closing_the_tab_is_not_enough():
    """The confusion this fixes is a documentation problem as much as a UI one."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert "Stop server" in html
    assert "Closing this tab leaves the server running" in html
    assert 'id="stopped-banner"' in html


# --- the guided builder ------------------------------------------------------
# The form replaced a YAML editor, which was still asking experimentalists to
# write code. Its options are served from the library so they cannot offer a
# choice the library would reject, and its validation has to fire before any
# compute is spent rather than hours into a run.

def test_builder_options_describe_every_supported_system():
    options = api.describe_builder_options()
    systems = {spec["system_type"] for spec in options["systems"]}
    assert {
        "inhomogeneous_heisenberg",
        "homogeneous_heisenberg",
        "homogeneous_xxz_j1j2j3",
        "homogeneous_xxz_j1j2j3_dmi_impurity",
    } == systems
    for spec in options["systems"]:
        for key in ("title", "recovers", "when", "view", "coupling_mode", "defaults"):
            assert spec[key], f"{spec['system_type']} is missing {key}"
        assert spec["couplings"], spec["system_type"]
    # Only the impurity system takes impurities, and it must ship a default
    # arrangement that can actually expose DMI.
    impurity = next(
        s for s in options["systems"]
        if s["system_type"] == "homogeneous_xxz_j1j2j3_dmi_impurity"
    )
    assert impurity["supports_impurities"]
    transverse = [
        item for item in impurity["default_impurities"] if item["transverse_mev"]
    ]
    assert len(transverse) >= 2, "the default arrangement cannot expose DMI"
    assert not any(
        s["supports_impurities"] for s in options["systems"]
        if s["system_type"] != "homogeneous_xxz_j1j2j3_dmi_impurity"
    )


def test_the_unlearnable_dmi_system_is_not_offered_at_all():
    """A form whose purpose is to prevent wasted runs must not offer one.

    An XXZ chain carrying DMI with nothing to break the symmetry trains to
    about zero D_z skill by construction, at every dataset size tried. It was
    offered with a warning; a warning is the wrong instrument for a choice that
    is never right.
    """
    options = api.describe_builder_options()
    offered = {spec["system_type"] for spec in options["systems"]}
    assert "homogeneous_xxz_j1j2j3_dmi" not in offered
    # The impurity system, which is the one that works, is still there.
    assert "homogeneous_xxz_j1j2j3_dmi_impurity" in offered
    with pytest.raises(ValueError, match="unknown system_type"):
        api.build_project_config(
            _builder_form(system_type="homogeneous_xxz_j1j2j3_dmi")
        )


def test_the_unlearnable_dmi_family_is_still_in_the_library():
    """Removing it from the form must not remove the evidence for removing it.

    Reproducing the measurement that D_z is unlearnable there needs the family.
    """
    from hamlet import HomogeneousXXZDMILongRangeFamily

    assert HomogeneousXXZDMILongRangeFamily is not None


def test_models_declare_whether_they_need_tensorflow():
    """The form disables what the environment cannot run."""
    models = {m["name"]: m for m in api.describe_builder_options()["models"]}
    assert models["ridge"]["needs_tensorflow"] is False
    assert models["random_forest"]["needs_tensorflow"] is False
    assert models["keras_mlp"]["needs_tensorflow"] is True
    for spec in models.values():
        for option in spec["options"]:
            # A None default is meaningful -- it is how "whatever the library
            # does" is expressed for a setting like unlimited tree depth -- but
            # a field the form cannot render is not.
            assert option["type"] in {
                "number", "integer", "layers", "boolean", "choice"
            }, spec["name"]
            assert option["label"], spec["name"]
            if option["type"] == "choice":
                assert option["default"] in option["choices"], option["name"]


def _builder_form(**overrides):
    form = {
        "name": "test run",
        "system_type": "homogeneous_xxz_j1j2j3_dmi_impurity",
        "n_sites": 8,
        "n_samples": 20,
        "coupling_ranges_mev": [[2, 6], [-1.5, 1.5], [-1, 1], [2, 6], [0.3, 2.5]],
        "impurities": [
            {"site": 1, "spin": "S=1", "transverse_mev": 2.0, "axial_mev": 0.0},
            {"site": 4, "spin": "S=1", "transverse_mev": 2.0, "axial_mev": 0.0},
        ],
        "transverse_field_mev": 0.0,
        "bias_range_mev": [0, 20],
        "bias_points": 21,
        "broadening_mev": 0.25,
        "observable": "total_spin",
        "observable_weights": [1, 1, 1],
        "cutoff_mev": 20.0,
        "output_points": 21,
        "model": "ridge",
        "preset": "standard",
        "model_options": {"alpha": 0.001},
    }
    form.update(overrides)
    return form


def test_builder_writes_a_valid_project_without_showing_yaml(tmp_path):
    built = api.build_project_config(_builder_form(), workspace=tmp_path)
    config = Path(built["config_path"])
    assert config.exists()
    # The file exists for reproducibility, but the user was never asked to read
    # or edit it -- the plan is what they see.
    plan = api.plan_project(config)
    assert plan["system_type"] == "homogeneous_xxz_j1j2j3_dmi_impurity"
    assert plan["view"] == "global"
    assert plan["generation_chains"] == 20
    assert plan["outputs"]


def test_impurities_and_field_reach_the_generated_configuration(tmp_path):
    built = api.build_project_config(
        _builder_form(transverse_field_mev=0.5), workspace=tmp_path
    )
    import yaml

    generate = yaml.safe_load(Path(built["config_path"]).read_text())["dataset"]["generate"]
    assert [item["site"] for item in generate["impurities"]] == [1, 4]
    assert generate["impurities"][0]["spin"] == "S=1"
    assert generate["transverse_field_mev"] == 0.5


def test_a_single_range_system_does_not_get_per_parameter_ranges(tmp_path):
    """Bond-inhomogeneous chains take one shared range, not one per bond."""
    import yaml

    built = api.build_project_config(
        _builder_form(
            system_type="inhomogeneous_heisenberg",
            n_sites=12,
            coupling_ranges_mev=[[30, 45]],
            bias_range_mev=[0, 100],
            bias_points=200,
            observable="Sz",
            cutoff_mev=50.0,
            output_points=200,
            impurities=[],
        ),
        workspace=tmp_path,
    )
    payload = yaml.safe_load(Path(built["config_path"]).read_text())
    generate = payload["dataset"]["generate"]
    assert generate["coupling_range_mev"] == [30.0, 45.0]
    assert "coupling_ranges_mev" not in generate
    assert "impurities" not in generate
    assert payload["training"]["view"] == "local_bonds"


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"system_type": "inhomogeneous_heisenberg", "n_sites": 2,
          "coupling_ranges_mev": [[30, 45]], "impurities": []}, "three sites"),
        ({"cutoff_mev": 99.0}, "inside the simulated bias window"),
        ({"output_points": 500}, "beyond the resolution actually simulated"),
        ({"impurities": [{"site": 1, "spin": "S=1", "transverse_mev": 2.0},
                         {"site": 99, "spin": "S=1", "transverse_mev": 2.0}]},
         "outside a 8-site chain"),
    ],
)
def test_impossible_settings_are_refused_before_any_compute(tmp_path, overrides, expected):
    """Each of these would otherwise fail only after the generation stage."""
    with pytest.raises(ValueError, match=expected):
        api.build_project_config(_builder_form(**overrides), workspace=tmp_path)


def test_rejected_settings_are_a_client_error_not_a_server_fault(server):
    """The form distinguishes "impossible request" from "the package broke"."""
    status, payload = post(
        server, "/api/build-config", {"form": _builder_form(cutoff_mev=99.0)}
    )
    assert status == 400
    assert "inside the simulated bias window" in payload["error"]


def test_builder_routes_are_reachable_over_http(server):
    payload = json.loads(get(server, "/api/builder-options")[1])
    assert payload["systems"] and payload["models"] and payload["presets"]
    assert "tensorflow_available" in payload


@pytest.mark.integration
def test_sample_preview_returns_plottable_spectra():
    """The check-before-you-commit step, against the real simulator.

    A preview evaluates representative sites rather than all of them, because
    cost scales with the number of correlators computed; the returned trace
    count is that subset, and the page says which sites they were.
    """
    form = _builder_form(bias_points=21)
    result = api.preview_samples(form, n_samples=1)
    assert len(result["bias_mev"]) == 21
    assert result["target_names"][-1] == "D_z_magnitude"
    assert len(result["samples"]) == 1
    sample = result["samples"][0]
    assert result["n_sites"] == 8
    assert result["evaluated_sites"] == list(api.preview_sites(form))
    assert len(sample["sites"]) == len(result["evaluated_sites"]), "one trace per site shown"
    assert len(sample["sites"][0]) == 21
    assert len(sample["couplings_mev"]) == 5
    assert all(np.isfinite(sample["sites"][0]))


# --- models you trained yourself --------------------------------------------
# A model trained through the interface was invisible on the models page and to
# the reuse advisor, which is exactly when you would want it offered back.

def _fake_artifact(directory: Path, *, system_type="homogeneous_heisenberg", preset="standard"):
    """Minimal artifact manifest, enough for discovery and assessment."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({
        "artifact_schema_version": 1,
        "system_type": system_type,
        "view": "global",
        "n_sites": 8,
        "model_name": "ridge",
        "target_names": ["J1", "J2"],
        "training_preset": {"name": preset},
        "preprocessing": {"bias_cutoff_mev": 60.0, "bias_min_mev": 0.0, "output_points": 40},
        "metrics": {
            "validation": {"ensemble": {"mae": 1.0}},
            "test": {"ensemble": {"mae": 1.1}},
            "split": {"train_groups": 40, "validation_groups": 12, "test_groups": 8},
        },
        "dataset_metadata": {"observable": "Sz"},
    }), encoding="utf-8")
    return directory


def test_models_you_trained_appear_alongside_the_published_ones(monkeypatch, tmp_path):
    workspace = tmp_path / "gui-projects"
    _fake_artifact(workspace / "my chain" / "run" / "artifact")
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)

    models = api.describe_available_models()["models"]
    origins = {m["origin"] for m in models}
    assert "yours" in origins, "a trained model was not discovered"
    assert "published" in origins, "the published models disappeared"

    mine = next(m for m in models if m["origin"] == "yours")
    # Labelled by the project the user named, not the literal "artifact" directory.
    assert mine["label"] == "my chain"
    assert mine["system_type"] == "homogeneous_heisenberg"
    assert mine["n_training_chains"] == 60


def test_a_project_manifest_that_is_not_an_artifact_is_ignored(monkeypatch, tmp_path):
    """Project directories hold other manifests, such as generation recipes."""
    workspace = tmp_path / "gui-projects"
    recipe = workspace / "my chain"
    recipe.mkdir(parents=True)
    (recipe / "manifest.json").write_text(
        json.dumps({"generation_schema_version": 1, "recipe": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    assert all(m["origin"] != "yours" for m in api.describe_available_models()["models"])


def test_your_own_model_card_is_readable(monkeypatch, tmp_path):
    workspace = tmp_path / "gui-projects"
    directory = _fake_artifact(workspace / "my chain" / "run" / "artifact")
    (directory / "MODEL_CARD.md").write_text("# Model card — mine\n", encoding="utf-8")
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    assert "mine" in api.read_model_card("my chain")["markdown"]


def test_the_advisor_considers_models_you_trained(monkeypatch, tmp_path, measurement_csv):
    """Searching only the published models made your own work unreachable."""
    workspace = tmp_path / "gui-projects"
    _fake_artifact(workspace / "my chain" / "run" / "artifact")
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)

    decision = api.advise_for_experiment(
        measurement_csv, 20.0, system_type="homogeneous_heisenberg", view="global"
    )
    considered = [a["path"] for a in decision["artifacts"]]
    assert any("gui-projects" in path for path in considered), (
        "the advisor did not even look at the model the user trained"
    )


# --- preview cost ------------------------------------------------------------
# Simulation cost is linear in the number of correlators evaluated, which is
# per site, so previewing every site is the largest avoidable expense. Bias
# points are nearly free by comparison, and process-level parallelism measured
# slower than sequential because one simulation already uses several cores.

def test_preview_evaluates_a_subset_of_sites_including_the_impurities():
    form = {
        "n_sites": 8,
        "impurities": [{"site": 1}, {"site": 4}],
    }
    sites = api.preview_sites(form, max_sites=4)
    assert len(sites) == 4
    assert 0 in sites and 7 in sites, "the ends of an open chain matter most"
    assert 1 in sites and 4 in sites, "the impurity sites are the point of the preview"
    assert sites == tuple(sorted(sites))


def test_preview_shows_every_site_of_a_short_chain():
    assert api.preview_sites({"n_sites": 3, "impurities": []}, max_sites=4) == (0, 1, 2)


def test_preview_site_count_never_exceeds_the_cap():
    form = {"n_sites": 20, "impurities": [{"site": s} for s in range(10)]}
    assert len(api.preview_sites(form, max_sites=4)) == 4


def test_dynamics_mode_is_chosen_by_basis_size_not_site_count():
    """Eight sites is 256 states, or 3456 with three spin-1 impurities."""
    from hamlet.simulation.dmrgpy import hilbert_dimension, recommended_dynamics_mode
    from hamlet.systems import HomogeneousXXZDMIImpurityChain, SiteImpurity

    def chain(n_sites, impurity_sites):
        return HomogeneousXXZDMIImpurityChain(
            n_sites, [5.0, 0.0, 0.0, 5.0, 1.0],
            impurities=tuple(
                SiteImpurity(s, "S=1", transverse_mev=2.0) for s in impurity_sites
            ),
        )

    assert hilbert_dimension(chain(8, ())) == 256
    assert recommended_dynamics_mode(chain(8, ())) == "ED"
    assert hilbert_dimension(chain(8, (1, 4, 6))) == 2 ** 5 * 3 ** 3
    # Long chains must not be attempted exactly.
    assert recommended_dynamics_mode(chain(14, (1, 4, 6))) == "DMRG"


def test_evaluate_sites_is_validated_against_the_chain():
    from hamlet.simulation import DmrgpySimulator

    with pytest.raises(ValueError, match="cannot be empty"):
        DmrgpySimulator(evaluate_sites=())
    with pytest.raises(ValueError, match="must not repeat"):
        DmrgpySimulator(evaluate_sites=(1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        DmrgpySimulator(evaluate_sites=(-1,))


def test_training_without_an_experiment_needs_a_manual_cutoff(tmp_path):
    """Augmentation is calibrated against a measurement; with none there is
    nothing to calibrate, so the cutoff has to be stated outright."""
    import yaml

    from hamlet.project import HamiltonianLearningProject

    built = api.build_project_config(
        _builder_form(system_type="homogeneous_heisenberg",
                      coupling_ranges_mev=[[30, 45], [0, 10]],
                      bias_range_mev=[0, 100], bias_points=60, observable="Sz",
                      cutoff_mev=60.0, output_points=40, impurities=[]),
        workspace=tmp_path,
    )
    payload = yaml.safe_load(Path(built["config_path"]).read_text())
    # The builder always sets it, which is what makes the path usable at all.
    assert payload["training"]["manual_cutoff_mev"] == 60.0
    project = HamiltonianLearningProject.from_config(built["config_path"])
    assert project.config.experiment_csv is None
    assert hasattr(project, "prepare_training_data_without_experiment")


# --- choosing a file ---------------------------------------------------------
# Typing an absolute path was the interface's last piece of command-line
# thinking. Two mechanisms replace it, because the file can be in two different
# places, and both have to be safe about what they touch.

def test_a_dropped_file_is_stored_and_its_path_returned(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    stored = api.save_upload("chain.csv", b"site,bias_meV,didv_A\n1,0,0\n")
    path = Path(stored["path"])
    assert path.exists()
    assert path.name.endswith("chain.csv")
    assert stored["size_bytes"] == len(b"site,bias_meV,didv_A\n1,0,0\n")
    # Kept rather than read and discarded: every later manifest records this
    # path, and a path that stopped existing would make those records useless.
    assert path.read_bytes().startswith(b"site,")


@pytest.mark.parametrize(
    "given, expected",
    [
        ("chain.csv", "chain.csv"),
        ("../../../etc/passwd", "passwd"),
        ("/absolute/path/data.npz", "data.npz"),
        ("weird name (2).CSV", "weird_name_2.CSV"),
        ("..", "measurement"),
        ("", "measurement"),
        ("...", "measurement"),
        ("a/b/../c.dat", "c.dat"),
        # The extension survives intact -- the loader dispatches on it, and an
        # .npz turned into _npz would be read as a CSV and fail confusingly.
        ("measurement.tar.gz", "measurement_tar.gz"),
        (".hidden", "measurement"),
    ],
)
def test_an_upload_name_cannot_carry_a_path(given, expected):
    """The name comes from the browser, so it is not to be trusted as a path."""
    assert api._safe_upload_name(given) == expected


def test_an_upload_lands_inside_the_uploads_directory_whatever_it_is_called(
    monkeypatch, tmp_path
):
    root = tmp_path / "uploads"
    monkeypatch.setattr(api, "_uploads_root", lambda: root)
    stored = api.save_upload("../../../etc/passwd", b"x")
    written = Path(stored["path"])
    assert written.parent.resolve() == root.resolve()
    assert written.exists() and written.read_bytes() == b"x"
    # Nothing outside the uploads directory was created.
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == [written.name]


def test_two_uploads_of_the_same_name_do_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    first = api.save_upload("chain.csv", b"one")
    second = api.save_upload("chain.csv", b"two")
    assert first["path"] != second["path"]
    assert Path(first["path"]).read_bytes() == b"one"


def test_browser_folder_upload_keeps_site_files_together(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    first = api.save_folder_upload("session-1", "my chain", "site_2.dat", b"two")
    second = api.save_folder_upload("session-1", "my chain", "site_10.dat", b"ten")
    assert first["folder_path"] == second["folder_path"]
    folder = Path(first["folder_path"])
    assert {item.name for item in folder.iterdir()} == {"site_2.dat", "site_10.dat"}
    assert Path(first["path"]).read_bytes() == b"two"


def test_folder_upload_names_cannot_escape_the_session_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    stored = api.save_folder_upload("../../session", "../chain", "../../site.dat", b"x")
    written = Path(stored["path"])
    assert (tmp_path / "uploads" / "folders") in written.parents
    assert written.name == "site.dat"


@pytest.mark.parametrize(
    "data, expected",
    [(b"", "empty"), (b"x" * (api.MAX_UPLOAD_BYTES + 1), "accepts up to")],
)
def test_useless_uploads_are_refused(monkeypatch, tmp_path, data, expected):
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    with pytest.raises(ValueError, match=expected):
        api.save_upload("chain.csv", data)


def test_browsing_lists_directories_and_readable_files_only(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "chain.csv").write_text("a")
    (tmp_path / "measurement.npz").write_text("a")
    (tmp_path / "notes.docx").write_text("a")
    (tmp_path / ".hidden").write_text("a")
    listing = api.browse_directory(tmp_path)
    names = {entry["name"] for entry in listing["entries"]}
    assert names == {"sub", "chain.csv", "measurement.npz"}
    # Counted rather than silently dropped, so an apparently empty directory is
    # distinguishable from one full of the wrong kind of file.
    assert listing["n_hidden_other_files"] == 1
    assert listing["parent"] == str(tmp_path.parent)
    assert {entry["kind"] for entry in listing["entries"]} == {"directory", "file"}


def test_browsing_a_file_shows_the_directory_it_lives_in(tmp_path):
    target = tmp_path / "chain.csv"
    target.write_text("a")
    assert api.browse_directory(target)["path"] == str(tmp_path.resolve())


def test_browser_marks_a_folder_that_contains_raw_sts_files(tmp_path):
    (tmp_path / "site-1.dat").write_text("raw")
    (tmp_path / "site-2.txt").write_text("raw")
    listing = api.browse_directory(tmp_path)
    assert listing["selectable_as_measurement"] is True
    assert listing["raw_sts_file_count"] == 2


def test_browsing_somewhere_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.browse_directory(tmp_path / "nope")


def test_only_files_the_interface_produced_can_be_served_back(monkeypatch, tmp_path):
    """Serving a file to the browser is a different act from reading one."""
    workspace = tmp_path / "gui-projects"
    (workspace / "run").mkdir(parents=True)
    report = workspace / "run" / "report.html"
    report.write_text("<b>hi</b>")
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    monkeypatch.setattr(api, "_uploads_root", lambda: tmp_path / "uploads")
    monkeypatch.setattr(api, "_screening_root", lambda: tmp_path / "screenings")
    monkeypatch.setattr(api, "_analysis_root", lambda: tmp_path / "analyses")
    monkeypatch.setattr(api, "_published_root", lambda: tmp_path / "published")
    assert api.resolve_readable_file(report) == report.resolve()

    outside = tmp_path / "secret.csv"
    outside.write_text("nope")
    with pytest.raises(PermissionError):
        api.resolve_readable_file(outside)
    with pytest.raises(PermissionError):
        api.resolve_readable_file("/etc/passwd")


def test_upload_and_browse_are_reachable_over_http(server, tmp_path):
    request = urllib.request.Request(
        server + "/api/upload?name=chain.csv",
        data=b"site,bias_meV,didv_A\n1,0,0\n",
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        stored = json.loads(response.read())
    assert Path(stored["path"]).exists()
    Path(stored["path"]).unlink()

    folder_request = urllib.request.Request(
        server + "/api/upload-folder?session=test-session&folder=chain&name=site-1.dat",
        data=b"[DATA]\nBias calc (V)\tLI Demod 1 X (A)\n0\t1\n",
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(folder_request) as response:
        folder_stored = json.loads(response.read())
    uploaded = Path(folder_stored["path"])
    assert uploaded.exists()
    assert uploaded.parent == Path(folder_stored["folder_path"])
    uploaded.unlink()
    uploaded.parent.rmdir()

    status, payload = get(server, "/api/browse")
    assert status == 200
    assert "entries" in json.loads(payload)


def test_serving_a_file_outside_the_workspace_is_refused(server):
    status, _ = get(server, "/api/file?path=/etc/passwd")
    assert status == 403


def test_the_page_offers_a_file_chooser_rather_than_a_path_box():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "dragover" in script and "/api/upload" in script
    assert "/api/upload-folder" in script
    assert "/api/browse" in script
    for prefix in ("data", "reuse", "an"):
        assert f'id="{prefix}-choose-folder"' in html
        assert re.search(
            rf'id="{prefix}-folder-upload"[^>]*webkitdirectory[^>]*multiple', html
        )
    # The remaining text inputs holding paths are hidden plumbing, not asks.
    for field in ("data-path", "reuse-path", "an-path"):
        assert re.search(rf'id="{field}"[^>]*hidden', html), field


def test_every_file_field_has_the_elements_its_helper_drives():
    """The helper builds ids from a prefix, which the id sweep cannot see."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    prefixes = re.findall(r'attachFileField\(\{\s*prefix:\s*"([a-z0-9-]+)"', script)
    assert prefixes, "no file fields are attached"
    suffixes = sorted(set(re.findall(r'el\(`\$\{prefix\}-([a-z0-9-]+)`\)', script)))
    assert suffixes, "the helper does not build any ids from its prefix"
    for prefix in prefixes:
        for suffix in suffixes:
            assert f'id="{prefix}-{suffix}"' in html, f"missing #{prefix}-{suffix}"


# --- designing a DMI sample without writing YAML -----------------------------

def _screening_form(**overrides):
    form = json.loads(json.dumps(api.describe_screening_options()["defaults"]))
    form.update(overrides)
    return form


def test_the_design_form_produces_a_runnable_screening(tmp_path):
    built = api.build_screening_config(_screening_form(), workspace=tmp_path)
    config = Path(built["config_path"])
    assert config.exists()
    # It has to be the same file the command line reads, or the design decision
    # is not reproducible outside the browser.
    from hamlet.dmi_design import load_screening_config

    designs, protocol = load_screening_config(config)
    assert len(designs) == built["n_candidates"]
    assert protocol.observable == "total_spin"


def test_the_form_reports_the_free_symmetry_verdict_immediately(tmp_path):
    """The single most useful thing a screen can say, and it costs nothing."""
    built = api.build_screening_config(_screening_form(), workspace=tmp_path)
    verdicts = {c["label"]: c["breaks_symmetry"] for c in built["candidates"]}
    assert verdicts["one impurity"] is False, "one impurity cannot break the symmetry"
    assert verdicts["two impurities"] is True
    assert verdicts["three impurities"] is True
    assert built["n_can_break_symmetry"] == 3


def test_the_form_carries_the_calibration_that_makes_an_imprint_mean_anything():
    options = api.describe_screening_options()
    assert options["calibration"], "an imprint without its calibration is a bare number"
    assert any("0.54" in entry["note"] for entry in options["calibration"])
    assert {v["name"] for v in options["verdicts"]} == {
        "hidden", "too weak", "marginal", "promising", "strong"
    }


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"candidates": []}, "at least one arrangement"),
        (
            {"candidates": [{"sites": [99], "transverse_mev": 2.0}]},
            "outside a 8-site chain",
        ),
        (
            {"candidates": [{"sites": [1, 1], "transverse_mev": 2.0}]},
            "listed twice",
        ),
        (
            {"candidates": [{"sites": [], "transverse_mev": 0.0}]},
            "nothing in it can break the symmetry",
        ),
    ],
)
def test_impossible_designs_are_refused_with_the_reason(tmp_path, overrides, expected):
    with pytest.raises(ValueError, match=expected):
        api.build_screening_config(_screening_form(**overrides), workspace=tmp_path)


def test_dmi_larger_than_the_exchange_scale_is_refused(tmp_path):
    """The pair trades D_z against exchange at fixed hypotenuse, so it cannot."""
    form = _screening_form()
    form["chain"] = {**form["chain"], "j_eff_mev": 2.0, "d_z_mev": 5.0}
    with pytest.raises(ValueError, match="no larger than the exchange scale"):
        api.build_screening_config(form, workspace=tmp_path)


def test_the_dmi_page_asks_for_a_design_not_a_configuration_path():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "/api/build-screening" in script
    assert "screening YAML" not in html, "the page still asks for a YAML path"
    for field in ("d-n-sites", "d-jeff", "d-dz", "d-candidates"):
        assert f'id="{field}"' in html, field


def test_screening_routes_are_reachable_over_http(server):
    status, payload = get(server, "/api/screening-options")
    assert status == 200 and "defaults" in json.loads(payload)
    status, payload = post(server, "/api/build-screening", {"form": _screening_form()})
    assert status == 200, payload
    assert payload["n_can_break_symmetry"] == 3


# --- designing the network ---------------------------------------------------

def test_the_mlp_architecture_is_editable_and_reaches_the_run(tmp_path):
    import yaml

    built = api.build_project_config(
        _builder_form(
            model="keras_mlp",
            model_options={
                "hidden_units": [256, 128],
                "dropout": 0.1,
                "activation": "gelu",
                "batch_normalization": False,
            },
        ),
        workspace=tmp_path,
    )
    options = yaml.safe_load(Path(built["config_path"]).read_text())["training"]["model_options"]
    assert options["hidden_units"] == [256, 128]
    assert options["activation"] == "gelu"
    assert options["batch_normalization"] is False
    from hamlet.project import ProjectConfig

    assert ProjectConfig.from_file(built["config_path"]).model_options["dropout"] == 0.1


def test_layer_widths_are_accepted_as_a_list_or_as_typed_text():
    assert api.coerce_model_options("keras_mlp", {"hidden_units": "512, 256, 128"})[
        "hidden_units"
    ] == [512, 256, 128]
    assert api.coerce_model_options("keras_mlp", {"hidden_units": [64]})[
        "hidden_units"
    ] == [64]


@pytest.mark.parametrize(
    "model, options, expected",
    [
        ("keras_mlp", {"hidden_units": []}, "at least one layer"),
        ("keras_mlp", {"hidden_units": [0]}, "at least one unit"),
        ("keras_mlp", {"hidden_units": [999999]}, "beyond anything"),
        ("keras_mlp", {"hidden_units": [16] * 30}, "beyond the 12"),
        ("keras_mlp", {"hidden_units": ["wide"]}, "not a whole number"),
        ("keras_mlp", {"dropout": 1.0}, "above the maximum"),
        ("keras_mlp", {"activation": "sigmoidish"}, "is not one of"),
        ("keras_mlp", {"learning_rate": -1}, "below the minimum"),
        ("ridge", {"n_estimators": 10}, "no hyperparameter called"),
    ],
)
def test_unusable_hyperparameters_are_refused_before_any_compute(model, options, expected):
    """Each of these would otherwise fail after generation, hours in."""
    with pytest.raises(ValueError, match=expected):
        api.coerce_model_options(model, options)


def test_a_blank_hyperparameter_means_the_library_default():
    """Clearing one field must not send an empty value down the stack."""
    assert api.coerce_model_options(
        "random_forest", {"max_depth": "", "n_estimators": 200}
    ) == {"n_estimators": 200}


def test_the_library_gets_the_last_word_on_a_hyperparameter_set():
    """Per-field limits cannot catch a rule that spans fields.

    The library states those rules once, in the configuration dataclass, and
    constructing it is free -- unlike building the network, which happens hours
    into a run.
    """
    with pytest.raises(ValueError, match="not usable"):
        api._validate_with_the_library("keras_mlp", {"hidden_units": []})
    with pytest.raises(ValueError, match="not usable"):
        api._validate_with_the_library("keras_cnn", {"kernel_size": 0})
    # A set the library accepts passes straight through.
    api._validate_with_the_library("keras_mlp", {"hidden_units": [32], "dropout": 0.1})


def test_the_page_offers_a_layer_editor():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "renderLayers" in script and "layer-add" in script
    assert 'id="f-model-options"' in html


# --- searching for hyperparameters -------------------------------------------

def test_a_search_can_be_asked_for_and_reaches_the_configuration(tmp_path):
    import yaml

    from hamlet.project import ProjectConfig

    built = api.build_project_config(
        _builder_form(tuning={"enabled": True, "n_trials": 8, "timeout_minutes": 30}),
        workspace=tmp_path,
    )
    tuning = yaml.safe_load(Path(built["config_path"]).read_text())["training"]["tuning"]
    assert tuning["n_trials"] == 8
    assert tuning["timeout_seconds"] == 1800.0
    # A trial is a short run, not a full one: a search that cost as much as the
    # training it precedes would never be worth starting.
    assert tuning["preset"] == "quick"
    config = ProjectConfig.from_file(built["config_path"])
    assert config.tuning.n_trials == 8


def test_no_search_is_configured_unless_it_was_asked_for(tmp_path):
    import yaml

    built = api.build_project_config(_builder_form(), workspace=tmp_path)
    training = yaml.safe_load(Path(built["config_path"]).read_text())["training"]
    assert "tuning" not in training


@pytest.mark.parametrize("n_trials", [0, 10_000])
def test_an_impossible_number_of_trials_is_refused(tmp_path, n_trials):
    with pytest.raises(ValueError, match="between 1 and"):
        api.build_project_config(
            _builder_form(tuning={"enabled": True, "n_trials": n_trials}),
            workspace=tmp_path,
        )


def test_the_form_says_which_backend_a_search_would_use():
    """"Optuna or not" changes how good the search is, so it must be visible."""
    tuning = api.describe_builder_options()["tuning"]
    assert tuning["backend"] in {"optuna", "random search"}
    assert tuning["optuna_available"] in {True, False}
    assert set(tuning["tunable_models"]) >= {"keras_mlp", "ridge", "random_forest"}
    assert tuning["searched"]["keras_mlp"], "nothing is said about what varies"
    assert "validation" in tuning["notes"]


def test_the_plan_announces_a_search_and_the_file_it_writes(tmp_path):
    built = api.build_project_config(
        _builder_form(tuning={"enabled": True, "n_trials": 5}), workspace=tmp_path
    )
    plan = api.plan_project(built["config_path"])
    assert any("search" in stage for stage in plan["stages"])
    assert any(item["path"].endswith("tuning.json") for item in plan["outputs"])
    assert any("validation split only" in note for note in plan["notes"])


# --- applying a model to a measurement ---------------------------------------
# The step the interface could not do: you could train a model and then had to
# leave for the command line to use it.

def test_an_analysis_copies_every_contract_term_from_the_model(tmp_path):
    import yaml

    name = "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    built = api.build_analysis_config(measurement, name, workspace=tmp_path / "out")
    payload = yaml.safe_load(Path(built["config_path"]).read_text())

    manifest = json.loads((api._published_root() / name / "manifest.json").read_text())
    assert payload["system_type"] == manifest["system_type"]
    assert payload["training"]["view"] == manifest["view"]
    # The cutoff is the model's, not a field the user could disagree with: the
    # weights are specific to it and are never substituted.
    assert payload["training"]["manual_cutoff_mev"] == manifest["preprocessing"][
        "bias_cutoff_mev"
    ]
    assert built["cutoff_mev"] == manifest["preprocessing"]["bias_cutoff_mev"]
    assert payload["artifact"].endswith(name)


def test_two_analyses_of_the_same_name_do_not_collide(tmp_path):
    """A run refuses to overwrite its own outputs, so each gets its own place."""
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    name = "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
    first = api.build_analysis_config(
        measurement, name, name="same", workspace=tmp_path / "out"
    )
    second = api.build_analysis_config(
        measurement, name, name="same", workspace=tmp_path / "out"
    )
    assert first["run_dir"] != second["run_dir"] or first["config_path"] != second[
        "config_path"
    ]


def test_analysing_with_a_model_that_does_not_exist_says_which_do(tmp_path):
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    with pytest.raises(FileNotFoundError, match="known models are"):
        api.build_analysis_config(measurement, "not-a-model", workspace=tmp_path)


def test_analysing_a_measurement_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such measurement"):
        api.build_analysis_config(
            tmp_path / "nope.csv",
            "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1",
            workspace=tmp_path,
        )


def test_the_interface_covers_the_whole_pipeline():
    """Every stage of the workflow has to be reachable without a terminal."""
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    for panel in ("start", "data", "reuse", "models", "analyse", "train", "dmi", "jobs"):
        assert f'data-panel="{panel}"' in html, f"no tab for {panel}"
        assert f'id="panel-{panel}"' in html, f"no panel for {panel}"
    for route in ("/api/inspect", "/api/advise", "/api/build-config",
                  "/api/run-project", "/api/build-analysis", "/api/run-analysis",
                  "/api/build-screening", "/api/run-screening"):
        assert route in script, f"the page never calls {route}"
    # The outputs of a run are opened from the page, not hunted for on disk.
    assert "/api/file?path=" in script
    assert "report.html" in script


def test_a_model_missing_a_contract_term_cannot_be_applied(tmp_path):
    """Guessing the system or the cutoff is the mistake the contract prevents."""
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(
        json.dumps({"artifact_schema_version": 1, "view": "global",
                    "preprocessing": {"bias_cutoff_mev": 20.0}})
    )
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    with pytest.raises(ValueError, match="system_type"):
        api.build_analysis_config(measurement, str(artifact), workspace=tmp_path / "out")


def test_a_directory_that_is_not_a_model_is_refused(tmp_path):
    """A path is accepted on its own terms, so it has to be checked on them."""
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    empty = tmp_path / "not-a-model"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        api.build_analysis_config(measurement, str(empty), workspace=tmp_path / "out")

    recipe = tmp_path / "recipe"
    recipe.mkdir()
    (recipe / "manifest.json").write_text(json.dumps({"generation_recipe": {}}))
    with pytest.raises(FileNotFoundError, match="artifact_schema_version"):
        api.build_analysis_config(measurement, str(recipe), workspace=tmp_path / "out")


def test_a_model_outside_the_workspace_can_still_be_applied(tmp_path):
    """A model trained on a cluster should not have to be moved to be used."""
    source = api._published_root() / (
        "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
    )
    elsewhere = tmp_path / "from-the-cluster"
    elsewhere.mkdir()
    (elsewhere / "manifest.json").write_text((source / "manifest.json").read_text())
    measurement = tmp_path / "chain.csv"
    measurement.write_text("site,bias_meV,didv_A\n1,0,0\n")
    built = api.build_analysis_config(
        measurement, str(elsewhere), workspace=tmp_path / "out"
    )
    assert built["artifact_path"] == str(elsewhere)


# --- running the analysis, end to end ----------------------------------------
# The stage the interface exists to complete. Everything above builds the
# configuration; these check that running it actually produces the answers, and
# that the job plumbing carries them back to the page.

@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    """A real ridge artifact and a measurement it fits, trained in about a second.

    Built rather than borrowed from the shipped model bank: those are large, and a
    published model that stopped matching this fixture's measurement would make
    an unrelated test fail for the wrong reason.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_guided_training import make_training_dataset

    from hamlet.project import HamiltonianLearningProject, ProjectConfig

    root = tmp_path_factory.mktemp("artifact")
    raw = make_training_dataset(n_samples=40)
    dataset = root / "dataset.npz"
    raw.save(dataset)
    config = ProjectConfig(
        name="fixture model",
        experiment_csv=None,
        output_dir=root / "run",
        dataset_path=dataset,
        cutoffs_mev=(50.0,),
        manual_cutoff_mev=50.0,
        output_points=30,
        view="local_bonds",
        model="ridge",
        preset="quick",
        verbose=0,
    )
    outcome = HamiltonianLearningProject(config).run()

    measurement = root / "measurement.csv"
    rows = []
    for site, spectrum in enumerate(raw.spectra[3], start=1):
        rows.extend(
            {"site": site, "bias_meV": float(b), "didv_A": float(v)}
            for b, v in zip(raw.bias_mev, spectrum)
        )
    pd.DataFrame(rows).to_csv(measurement, index=False)
    return {"artifact": outcome.artifact_path, "measurement": measurement, "root": root}


def test_running_an_analysis_writes_every_answer_file(trained_artifact, tmp_path):
    built = api.build_analysis_config(
        trained_artifact["measurement"],
        str(trained_artifact["artifact"]),
        name="end to end",
        allow_development_artifacts=True,
        workspace=tmp_path,
    )
    result = api.run_analysis(built["config_path"])

    assert result["kind"] == "analysis"
    assert result["status"]
    for key in ("report_html", "summary_png", "couplings_csv", "report_json"):
        assert Path(result[key]).exists(), key
    # The report is self-contained, which is what makes it shareable.
    html = Path(result["report_html"]).read_text(encoding="utf-8")
    assert "end to end" in html
    assert "Generated by HamLeT" in html
    # The couplings come back small enough to render on the page.
    table = result["couplings"]
    assert table["columns"][0] == "bond"
    assert len(table["rows"]) == table["n_rows"]
    assert float(table["rows"][0][3]) > 0, "no coupling was inferred"


def test_a_development_model_is_refused_unless_that_is_asked_for(
    trained_artifact, tmp_path
):
    """The quick preset marks an artifact development-only, and it stays so."""
    built = api.build_analysis_config(
        trained_artifact["measurement"],
        str(trained_artifact["artifact"]),
        name="refused",
        workspace=tmp_path,
    )
    with pytest.raises(RuntimeError, match="preflight decision"):
        api.run_analysis(built["config_path"])
    # And the refusal is written down rather than only raised.
    decision = list(Path(built["run_dir"]).rglob("workflow_decision.json"))
    assert decision, "the preflight decision was not saved"
    reasons = json.loads(decision[0].read_text())["artifact_assessments"]
    assert any("development-only" in r for item in reasons for r in item["reasons"])


def test_the_analysis_job_carries_its_output_back_to_the_page(
    trained_artifact, tmp_path
):
    """Analysis is a job, so the page must be able to poll it to completion."""
    registry = api.JobRegistry()
    built = api.build_analysis_config(
        trained_artifact["measurement"],
        str(trained_artifact["artifact"]),
        name="as a job",
        allow_development_artifacts=True,
        workspace=tmp_path,
    )
    job = registry.submit(
        "analysis", "analyse", lambda: api.run_analysis(built["config_path"])
    )
    finished = wait_for_job(registry, job.job_id, timeout=120.0)
    assert finished["status"] == "finished", finished["error"]
    assert finished["result"]["kind"] == "analysis"
    assert any("checking the model" in line for line in finished["lines"])


def test_the_whole_pipeline_runs_over_http(server, trained_artifact):
    """Build the analysis and run it exactly as the page does, over the wire."""
    status, built = post(server, "/api/build-analysis", {
        "path": str(trained_artifact["measurement"]),
        "model": str(trained_artifact["artifact"]),
        "name": "over http",
        "allow_development_artifacts": True,
    })
    assert status == 200, built
    assert built["cutoff_mev"] == 50.0

    status, job = post(server, "/api/run-analysis", {"config_path": built["config_path"]})
    assert status == 200, job
    assert job["kind"] == "analysis"

    for _ in range(600):
        status, polled = get(server, f"/api/job?id={job['job_id']}")
        polled = json.loads(polled)
        if polled["status"] != "running":
            break
        threading.Event().wait(0.2)
    assert polled["status"] == "finished", polled.get("error")

    # And the files it wrote are reachable from the page, which is the point of
    # running it there rather than in a terminal.
    status, body = get(server, "/api/file?path=" + polled["result"]["report_html"])
    assert status == 200
    assert b"Generated by HamLeT" in body


def test_a_training_run_started_from_the_page_reports_its_model(server, tmp_path):
    """The train page's Run button, followed to completion on a tiny project."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_guided_training import make_training_dataset

    import yaml

    make_training_dataset(n_samples=40).save(tmp_path / "dataset.npz")
    config = tmp_path / "project.yaml"
    config.write_text(yaml.safe_dump({
        "config_schema_version": 1,
        "name": "page run",
        "output_dir": str(tmp_path / "run"),
        "dataset": {"format": "portable", "path": str(tmp_path / "dataset.npz")},
        "training": {"cutoffs_mev": [50.0], "manual_cutoff_mev": 50.0,
                     "output_points": 30, "view": "local_bonds",
                     "model": "ridge", "preset": "quick",
                     "tuning": {"n_trials": 2}},
    }, sort_keys=False))

    status, job = post(server, "/api/run-project", {"config_path": str(config)})
    assert status == 200, job
    for _ in range(900):
        polled = json.loads(get(server, f"/api/job?id={job['job_id']}")[1])
        if polled["status"] != "running":
            break
        threading.Event().wait(0.2)
    assert polled["status"] == "finished", polled.get("error")

    result = polled["result"]
    assert result["kind"] == "training"
    assert result["validation_mae_mev"] >= 0
    assert result["test_mae_mev"] >= 0
    # A search was configured, so its outcome has to reach the page too --
    # otherwise the user cannot tell a tuned run from an untuned one.
    assert result["tuning"]["backend"] in {"optuna", "random"}
    assert "kept_defaults" in result["tuning"]
    assert Path(result["tuning"]["report_path"]).exists()


def test_the_coupling_table_is_trimmed_for_the_page(tmp_path):
    """A long chain must not send hundreds of rows into the browser."""
    path = tmp_path / "couplings.csv"
    rows = ["bond,coupling"] + [f"{i},{30 + i}" for i in range(200)]
    path.write_text("\n".join(rows) + "\n")
    table = api._read_couplings(path, max_rows=5)
    assert table["columns"] == ["bond", "coupling"]
    assert len(table["rows"]) == 5
    assert table["n_rows"] == 200
    assert table["truncated"] is True


def test_a_missing_coupling_table_is_empty_not_an_error(tmp_path):
    assert api._read_couplings(tmp_path / "nope.csv") == {"columns": [], "rows": []}
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert api._read_couplings(empty) == {"columns": [], "rows": []}


# --- where the interface writes ----------------------------------------------

def test_an_installed_package_does_not_write_into_site_packages(monkeypatch, tmp_path):
    """REPO_ROOT is inside site-packages for a wheel install, where pip can
    delete it. A user's measurements do not belong there."""
    monkeypatch.delenv("HAMLET_WORKSPACE", raising=False)
    monkeypatch.setattr(api, "_is_source_checkout", lambda: False)
    monkeypatch.setattr(api.Path, "home", staticmethod(lambda: tmp_path))
    assert api._workspace_base() == tmp_path / ".hamlet" / "workspace"


def test_a_source_checkout_keeps_its_output_beside_the_project(monkeypatch):
    monkeypatch.delenv("HAMLET_WORKSPACE", raising=False)
    monkeypatch.setattr(api, "_is_source_checkout", lambda: True)
    assert api._workspace_base() == api.REPO_ROOT / "results"


def test_the_workspace_can_be_pointed_somewhere_else(monkeypatch, tmp_path):
    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path / "elsewhere"))
    assert api._workspace_base() == (tmp_path / "elsewhere").resolve()
    assert api._uploads_root() == (tmp_path / "elsewhere").resolve() / "gui-uploads"
    assert api._workspace_root() == (tmp_path / "elsewhere").resolve() / "gui-projects"


def test_every_directory_the_interface_writes_to_is_readable_back(monkeypatch, tmp_path):
    """A file it produced but refuses to serve would be a dead link on the page."""
    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    roots = {str(root) for root in api._readable_roots()}
    for produced in (api._workspace_root(), api._uploads_root(),
                     api._screening_root(), api._analysis_root()):
        assert str(produced) in roots, produced


def test_figures_are_drawn_without_a_gui_backend(trained_artifact, tmp_path):
    """Jobs run on worker threads, where an interactive backend is documented
    as likely to fail. The interface never shows a figure, only saves one."""
    import warnings

    import matplotlib

    api.use_headless_plotting()
    assert matplotlib.get_backend().lower() == "agg"

    built = api.build_analysis_config(
        trained_artifact["measurement"],
        str(trained_artifact["artifact"]),
        name="headless",
        allow_development_artifacts=True,
        workspace=tmp_path,
    )
    registry = api.JobRegistry()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        job = registry.submit(
            "analysis", "analyse", lambda: api.run_analysis(built["config_path"])
        )
        finished = wait_for_job(registry, job.job_id, timeout=120.0)
    assert finished["status"] == "finished", finished["error"]
    assert not [
        str(w.message) for w in caught if "GUI outside of the main thread" in str(w.message)
    ]
    assert Path(finished["result"]["summary_png"]).exists()


def test_selecting_the_headless_backend_is_safe_to_repeat():
    import matplotlib

    api.use_headless_plotting()
    api.use_headless_plotting()
    assert matplotlib.get_backend().lower() == "agg"


# --- the chain, drawn --------------------------------------------------------
# Impurity positions are zero-based, must be distinct, and whether an
# arrangement can expose DMI depends on where they sit. A row of numbers hides
# all of that, so both pages that choose sites draw the chain instead.

def test_both_pages_that_choose_sites_draw_a_chain():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    # One shared component, so the two pages cannot drift apart.
    assert "function chainSvg(" in script
    assert "function renderChain(" in script
    assert 'id="f-chain"' in html, "the training page has nowhere to draw a chain"
    assert 'class="chain cand-chain"' in script, "candidates have no chain"
    # Clicking a site is the point; typing an index is the fallback.
    assert "onToggle" in script
    assert "click a site" in html.lower()


def test_the_chain_is_clickable_and_reachable_by_keyboard():
    """It is the primary control on the DMI page, not decoration over one."""
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'role="button"' in script and 'tabindex="0"' in script
    assert '"keydown"' in script
    assert 'event.key === "Enter"' in script


def test_a_site_off_the_end_of_the_chain_is_reported_not_dropped():
    """Shortening a chain must not silently change the design."""
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "function offChainWarning(" in script
    assert "lie" in script and "outside a" in script
    # Both chains use it, so neither can quietly discard a site.
    assert script.count("offChainWarning(") >= 3


def test_the_chain_follows_the_chain_length():
    """A diagram of a chain that is no longer configured is worse than none."""
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert 'el("f-n-sites").addEventListener("input", drawImpurityChain)' in script
    assert 'el("d-n-sites").addEventListener("input"' in script


def test_a_new_impurity_copies_the_systems_own_default():
    """Clicking a site must produce one that can actually expose DMI.

    An invented S=1 with no transverse anisotropy is inert: it would look
    placed and change nothing.
    """
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "function defaultImpurity(" in script
    assert "default_impurities" in script
    spec = next(
        s for s in api.describe_builder_options()["systems"]
        if s["system_type"] == "homogeneous_xxz_j1j2j3_dmi_impurity"
    )
    assert spec["default_impurities"][0]["transverse_mev"] > 0, (
        "the template the diagram copies cannot break the symmetry"
    )


def test_the_design_page_states_the_symmetry_rule_as_sites_are_chosen():
    """The free verdict, restated while the design is being drawn."""
    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "function candidateVerdict(" in script
    assert "One impurity cannot break the symmetry" in script
    assert "distinct >= 2" in script


def test_the_chain_geometry_keeps_long_chains_legible():
    """A 20-site chain has to fit without the balls growing into each other."""
    import re

    script = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    geometry = re.search(
        r"const CHAIN_GEOMETRY = \{ radius: (\d+), padX: (\d+), padY: (\d+), "
        r"longChain: (\d+) \}",
        script,
    )
    assert geometry, "the geometry is not stated in one place"
    radius, pad_x, pad_y, long_chain = (int(g) for g in geometry.groups())
    long_radius = int(re.search(r"const radius = long \? (\d+)", script).group(1))
    long_gap = int(re.search(r"const gap = long \? (\d+) : (\d+)", script).group(1))
    short_gap = int(re.search(r"const gap = long \? (\d+) : (\d+)", script).group(2))
    # Both shrink together, so the sticks stay visible rather than vanishing
    # between balls that have grown to meet.
    assert long_gap - 2 * long_radius > 0, "long chains have no visible bonds"
    assert short_gap - 2 * radius > 0, "short chains have no visible bonds"
    assert long_radius >= 10, "the circles must stay large enough to click"
    # The site index sits below the ball, inside the padding.
    assert pad_y > radius, "the index label falls outside the drawing"
    assert pad_x > 0
    assert long_chain >= 12, "chains this short do not need the tighter spacing"


# --- the logo ----------------------------------------------------------------

@pytest.mark.parametrize("name", ["hamlet-logo.png", "hamlet-icon.png"])
def test_the_logo_ships_inside_the_package(name):
    """Served from disk, so it has to live where the server can reach it.

    `assets/` is outside the package and absent from a wheel, so the copy under
    `static/` is the one that matters.
    """
    packaged = STATIC_ROOT / name
    assert packaged.is_file(), f"{name} is missing from the interface's assets"
    original = REPO_ROOT / "assets" / "logos" / name
    assert packaged.read_bytes() == original.read_bytes(), (
        f"{name} has drifted from assets/logos/{name}"
    )


def test_the_page_shows_the_logo_and_sets_a_tab_icon():
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert 'rel="icon"' in html and 'href="/hamlet-icon.png"' in html
    assert 'src="/hamlet-logo.png"' in html
    # A logo is not a caption: a reader who cannot see it still needs the name.
    assert 'alt="HamLeT' in html
    assert "<title>HamLeT</title>" in html

    # Exactly one icon is offered. When several match, browsers take the last
    # usable one, and `media` on a favicon is honoured by some and ignored by
    # others -- so any ordering that is right in one browser puts the wrong
    # art in another's tab strip.
    import re as _re

    icons = _re.findall(r'<link rel="icon"[^>]*>', html)
    assert len(icons) == 1, f"more than one icon link: {icons}"


def test_the_tab_icon_fills_the_tab():
    """A framed, non-square mark is letterboxed and then mostly frame.

    The tab gives it 16 pixels. Whatever is spent on a border is taken from
    the letter and the hand, which are the mark.
    """
    from PIL import Image
    import numpy as np

    icon = Image.open(STATIC_ROOT / "hamlet-icon.png").convert("RGBA")
    width, height = icon.size
    assert width == height, f"a non-square icon is letterboxed into the tab: {icon.size}"
    assert width >= 128, "too small to downscale cleanly to a tab icon"

    opaque = np.array(icon)[..., 3] > 8
    rows, cols = np.nonzero(opaque)
    covered = (cols.max() - cols.min() + 1) * (rows.max() - rows.min() + 1)
    fraction = covered / (width * height)
    assert fraction > 0.6, (
        f"the mark covers only {fraction:.0%} of the tile; at 16 pixels that "
        f"is mostly empty space"
    )


def test_the_logo_survives_a_dark_background():
    """It is black line art on transparency, invisible on a dark ground."""
    css = (STATIC_ROOT / "style.css").read_text(encoding="utf-8")
    dark = css[css.index("@media (prefers-color-scheme: dark)"):]
    assert "#brand-logo { filter: invert(1); }" in dark


@pytest.mark.parametrize(
    "route, content_type, magic",
    [
        ("/hamlet-icon.png", "image/png", b"\x89PNG"),
        ("/hamlet-logo.png", "image/png", b"\x89PNG"),
    ],
)
def test_the_logo_is_served_as_an_image(server, route, content_type, magic):
    """Handed over as an octet-stream, a browser quietly declines the favicon."""
    request = urllib.request.Request(server + route)
    with urllib.request.urlopen(request) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == content_type
        assert response.read().startswith(magic)


# --- the rest of the wire ----------------------------------------------------
# Every route the page calls, exercised over HTTP rather than only in-process,
# because the failures these catch live in the routing, not the operations.

def test_an_oversized_upload_is_refused_without_reading_it(server):
    """The point of a size limit is not to spend the memory finding out."""
    request = urllib.request.Request(
        server + "/api/upload?name=huge.csv",
        data=b"x" * 32,
        headers={
            "Content-Type": "application/octet-stream",
            # Declared, not sent: the server must decide from the header.
            "Content-Length": str(api.MAX_UPLOAD_BYTES + 1),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            status, payload = response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        status, payload = exc.code, json.loads(exc.read())
    except OSError:
        # Some clients abort once the server answers early, which is itself
        # evidence the body was never consumed.
        return
    assert status == 413
    assert "accepts up to" in payload["error"]


def test_an_empty_upload_is_a_client_error(server):
    request = urllib.request.Request(
        server + "/api/upload?name=nothing.csv",
        data=b"",
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    assert status == 400


@pytest.mark.parametrize(
    "suffix, content_type",
    [
        (".html", "text/html; charset=utf-8"),
        (".csv", "text/csv; charset=utf-8"),
        (".json", "application/json"),
        (".png", "image/png"),
    ],
)
def test_produced_files_are_served_with_a_usable_type(
    server, monkeypatch, tmp_path, suffix, content_type
):
    """A report handed over as an octet-stream downloads instead of opening."""
    workspace = tmp_path / "gui-projects"
    workspace.mkdir(parents=True)
    produced = workspace / f"result{suffix}"
    produced.write_bytes(b"x")
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    request = urllib.request.Request(
        server + "/api/file?path=" + urllib.parse.quote(str(produced))
    )
    with urllib.request.urlopen(request) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == content_type


def test_serving_a_file_that_does_not_exist_is_a_404(server, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_workspace_root", lambda: tmp_path)
    status, _ = get(server, "/api/file?path=" + str(tmp_path / "gone.html"))
    assert status == 404


def test_the_symmetry_check_is_reachable_over_http(server):
    config = REPO_ROOT / "examples" / "dmi_screening.yaml"
    status, payload = post(server, "/api/screening-preview", {"config_path": str(config)})
    assert status == 200
    assert payload["n_can_break_symmetry"] == 4


def test_the_model_card_route_answers_and_refuses(server):
    name = "homogeneous_xxz_j1j2j3_dmi_impurity_l8_ridge_standard_v1"
    status, body = get(server, f"/api/model-card?name={name}")
    assert status == 200
    assert "Model card" in json.loads(body)["markdown"]
    status, _ = get(server, "/api/model-card?name=../../pyproject.toml")
    assert status == 404


def test_reading_and_saving_a_configuration_round_trips(server, tmp_path):
    """The escape hatch for a configuration the form did not write."""
    import yaml

    path = tmp_path / "project.yaml"
    path.write_text(yaml.safe_dump({
        "config_schema_version": 1,
        "name": "hand written",
        "output_dir": str(tmp_path / "run"),
        "artifact": str(tmp_path / "artifact"),
        "training": {"manual_cutoff_mev": 50.0, "cutoffs_mev": [50.0]},
    }, sort_keys=False))
    status, body = get(server, "/api/config?path=" + str(path))
    assert status == 200
    text = json.loads(body)["text"]
    assert "hand written" in text

    status, payload = post(server, "/api/save-config", {
        "path": str(path), "text": text.replace("hand written", "renamed")})
    assert status == 200 and payload["saved"]
    assert "renamed" in path.read_text()


def test_a_preview_is_submitted_as_a_job_not_held_open(server):
    """One chain is about a minute, far too long to hold a request open."""
    status, payload = post(server, "/api/preview-samples", {
        "form": _builder_form(bias_points=5), "n_samples": 1})
    assert status == 200
    assert payload["kind"] == "preview"
    assert payload["status"] == "running"
    assert payload["job_id"]


def test_the_examples_route_lists_the_shipped_configurations(server):
    status, body = get(server, "/api/examples")
    assert status == 200
    names = {entry["name"] for entry in json.loads(body)["configs"]}
    assert "dmi_screening.yaml" in names
    for entry in json.loads(body)["configs"]:
        assert Path(entry["path"]).exists()


def test_a_long_directory_listing_is_truncated_and_says_so(tmp_path):
    for index in range(30):
        (tmp_path / f"chain-{index:03d}.csv").write_text("x")
    listing = api.browse_directory(tmp_path, max_entries=10)
    assert len(listing["entries"]) == 10
    assert listing["truncated"] is True
    assert api.browse_directory(tmp_path)["truncated"] is False


def test_browsing_offers_shortcuts_that_exist(monkeypatch, tmp_path):
    """A shortcut to a directory that is not there is a dead button."""
    monkeypatch.setenv("HAMLET_WORKSPACE", str(tmp_path))
    (tmp_path / "gui-uploads").mkdir(parents=True)
    listing = api.browse_directory(tmp_path)
    labels = {item["label"] for item in listing["shortcuts"]}
    assert "Uploads" in labels
    assert "Your runs" not in labels, "a directory that does not exist was offered"
    for item in listing["shortcuts"]:
        assert Path(item["path"]).exists()


def test_a_model_is_resolved_by_the_name_the_page_shows(monkeypatch, tmp_path):
    """The page sends back the label it displayed, not a path."""
    workspace = tmp_path / "gui-projects"
    artifact = workspace / "my chain" / "run" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text(json.dumps({"artifact_schema_version": 1}))
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    monkeypatch.setattr(api, "_published_root", lambda: tmp_path / "none")
    assert api._artifact_for("my chain") == artifact
    assert api._artifact_for(str(artifact)) == artifact


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({"enabled": True, "n_trials": "twelve"}, "whole number"),
        ({"enabled": True, "n_trials": 5, "timeout_minutes": 0}, "must be positive"),
        ({"enabled": True, "n_trials": 5, "timeout_minutes": -3}, "must be positive"),
    ],
)
def test_impossible_search_settings_are_refused(settings, expected):
    with pytest.raises(ValueError, match=expected):
        api._tuning_payload({"tuning": settings}, "ridge")


def test_a_search_is_only_configured_when_it_was_asked_for():
    assert api._tuning_payload({}, "ridge") is None
    assert api._tuning_payload({"tuning": {"enabled": False, "n_trials": 9}}, "ridge") is None
    assert api._tuning_payload({"tuning": {"enabled": True}}, "ridge")["n_trials"] == 20


def test_a_search_with_no_time_limit_records_none():
    payload = api._tuning_payload(
        {"tuning": {"enabled": True, "n_trials": 3, "timeout_minutes": None}}, "ridge"
    )
    assert "timeout_seconds" not in payload


# --- job plumbing ------------------------------------------------------------

def test_a_chatty_job_cannot_grow_without_bound():
    """A multi-hour generation prints per chain; holding all of it is a leak."""
    registry = api.JobRegistry()

    def noisy():
        for index in range(3000):
            print(f"line {index}")
        return "done"

    job = registry.submit("test", "noisy", noisy)
    finished = wait_for_job(registry, job.job_id, timeout=60.0)
    assert finished["status"] == "finished"
    assert len(finished["lines"]) <= 2000, "the output buffer is unbounded"
    # The newest output is what a user is watching, so that is what survives.
    assert finished["lines"][-1] == "line 2999"


def test_partial_output_is_visible_while_a_job_runs():
    """A progress line only helps if it arrives before the job ends."""
    import threading as _threading

    registry = api.JobRegistry()
    release = _threading.Event()

    def slow():
        print("first step")
        release.wait(10)
        return "done"

    job = registry.submit("test", "slow", slow)
    for _ in range(200):
        if registry.get(job.job_id).lines:
            break
        _threading.Event().wait(0.02)
    assert registry.get(job.job_id).lines == ["first step"]
    assert registry.get(job.job_id).status == "running"
    release.set()
    assert wait_for_job(registry, job.job_id)["status"] == "finished"


def test_a_job_reports_how_long_it_has_been_going():
    registry = api.JobRegistry()
    job = registry.submit("test", "instant", lambda: None)
    finished = wait_for_job(registry, job.job_id)
    assert finished["elapsed_seconds"] >= 0
    assert finished["finished_at"] >= finished["started_at"]


def test_a_model_you_trained_is_labelled_by_the_name_you_gave_it(
    monkeypatch, tmp_path
):
    """The artifact directory is called "artifact"; the project above it carries
    the name the user typed, which is the one worth showing."""
    workspace = tmp_path / "gui-projects"
    artifact = workspace / "my first chain" / "run" / "artifact"
    artifact.mkdir(parents=True)
    monkeypatch.setattr(api, "_workspace_root", lambda: workspace)
    assert api._workspace_label(artifact) == "my first chain"
    # A directory outside the workspace keeps its own name rather than raising.
    assert api._workspace_label(tmp_path / "elsewhere") == "elsewhere"
