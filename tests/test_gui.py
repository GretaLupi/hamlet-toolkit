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
    payload = api.describe_published_models()
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

@pytest.mark.parametrize("route", ["/", "/style.css", "/app.js"])
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


# --- manifests from more than one pipeline ----------------------------------
# Published artifacts come from datasets generated by this package and from
# datasets imported from earlier work. Their manifests record the same facts
# under different keys, so the browser has to read both shapes.

def test_every_published_model_reports_its_model_and_observable():
    """A blank field here means the browser is reading the wrong key."""
    for entry in api.describe_published_models()["models"]:
        assert entry["model"], f"{entry['name']} does not report which model it is"
        assert entry["observable"], f"{entry['name']} does not report its observable"


def test_training_chain_count_is_the_split_total_not_the_per_shard_count():
    """A dataset merged from shards records the per-shard count in its recipe.

    Reading that would advertise a 3000-chain model as trained on 20, so the
    count must come from the split-group totals.
    """
    for entry in api.describe_published_models()["models"]:
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
    models = {m["name"]: m for m in api.describe_published_models()["models"]}
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
