"""Localhost HTTP server for the browser interface.

Standard library only, deliberately: the audience installs this to analyse
measurements, not to maintain a web stack, and a GUI is not worth a dependency
that can break the science path. It binds to 127.0.0.1 because the operations
behind it read local files and start local compute.
"""

from __future__ import annotations

import errno
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import api

STATIC_ROOT = Path(__file__).resolve().parent / "static"


class _Handler(BaseHTTPRequestHandler):
    server_version = "hamlet-gui"
    registry: api.JobRegistry

    def log_message(self, format: str, *args: Any) -> None:
        # The default handler logs every poll, which buries real messages.
        return None

    # --- helpers ---
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: BaseException, status: int = 400) -> None:
        self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        raw = self.rfile.read(length)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _serve_static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        target = (STATIC_ROOT / name).resolve()
        # Refuse anything that escapes the static directory.
        if STATIC_ROOT not in target.parents or not target.is_file():
            self.send_error(404, "not found")
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            # Without this a browser is handed the favicon as an octet-stream
            # and quietly declines to use it, which looks like a missing file.
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", content_types.get(target.suffix, "application/octet-stream")
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --- routes ---
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route == "/api/overview":
                self._send_json(api.workflow_overview())
            elif route == "/api/models":
                self._send_json(api.describe_available_models())
            elif route == "/api/model-card":
                self._send_json(api.read_model_card(query["name"][0]))
            elif route == "/api/builder-options":
                self._send_json(api.describe_builder_options())
            elif route == "/api/screening-options":
                self._send_json(api.describe_screening_options())
            elif route == "/api/compute":
                self._send_json(api.describe_compute_options())
            elif route == "/api/cluster-config":
                self._send_json(api.read_cluster_config())
            elif route == "/api/browse":
                self._send_json(
                    api.browse_directory(
                        query["path"][0] if query.get("path") else None
                    )
                )
            elif route == "/api/file":
                self._serve_result_file(query["path"][0])
            elif route == "/api/examples":
                self._send_json(api.list_example_configs())
            elif route == "/api/config":
                self._send_json(api.read_config_text(query["path"][0]))
            elif route == "/api/jobs":
                self._send_json({"jobs": self.registry.list()})
            elif route == "/api/job":
                job = self.registry.get(query["id"][0])
                if job is None:
                    self._send_json({"error": "no such job"}, status=404)
                else:
                    self._send_json(job.to_dict())
            else:
                self._serve_static(route)
        except FileNotFoundError as exc:
            self._send_error_json(exc, status=404)
        except KeyError as exc:
            self._send_error_json(ValueError(f"missing query parameter {exc}"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            self._send_error_json(exc, status=500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in {"/api/upload", "/api/upload-folder"}:
                # Handled before the body is read as JSON: this one carries raw
                # file bytes, and base64 in a JSON envelope would inflate a
                # measurement by a third for no benefit.
                self._receive_upload(
                    parse_qs(parsed.query), folder=route == "/api/upload-folder"
                )
                return
            payload = self._read_json()
            if route == "/api/inspect":
                self._send_json(api.inspect_experiment(payload["path"]))
            elif route == "/api/advise":
                self._send_json(
                    api.advise_for_experiment(
                        payload["path"],
                        float(payload["cutoff_mev"]),
                        artifact_roots=payload.get("artifact_roots"),
                        system_type=payload.get("system_type") or None,
                        view=payload.get("view") or None,
                    )
                )
            elif route == "/api/plan":
                seconds = payload.get("seconds_per_chain")
                self._send_json(
                    api.plan_project(
                        payload["config_path"],
                        seconds_per_chain=float(seconds) if seconds else None,
                    )
                )
            elif route == "/api/build-config":
                self._send_json(api.build_project_config(payload["form"]))
            elif route == "/api/preview-samples":
                self._send_json(self._submit_preview(payload).to_dict())
            elif route == "/api/save-config":
                self._send_json(
                    api.write_config_text(payload["path"], payload["text"])
                )
            elif route == "/api/build-screening":
                self._send_json(api.build_screening_config(payload["form"]))
            elif route == "/api/screening-preview":
                self._send_json(api.screening_preview(payload["config_path"]))
            elif route == "/api/run-screening":
                self._send_json(self._submit_screening(payload).to_dict())
            elif route == "/api/build-analysis":
                self._send_json(
                    api.build_analysis_config(
                        payload["path"],
                        payload["model"],
                        name=payload.get("name") or None,
                        allow_development_artifacts=bool(
                            payload.get("allow_development_artifacts", False)
                        ),
                    )
                )
            elif route == "/api/run-analysis":
                self._send_json(self._submit_analysis(payload).to_dict())
            elif route == "/api/run-project":
                self._send_json(self._submit_project(payload).to_dict())
            elif route == "/api/cancel-job":
                self._send_json(self.registry.cancel(payload["job_id"]))
            elif route == "/api/save-cluster-config":
                self._send_json(api.write_cluster_config(payload["text"]))
            elif route == "/api/check-cluster":
                self._send_json(api.check_cluster())
            elif route == "/api/cluster-script":
                self._send_json(api.cluster_script(payload["config_path"]))
            elif route == "/api/submit-to-cluster":
                self._send_json(self._submit_to_cluster(payload).to_dict())
            elif route == "/api/cluster-status":
                self._send_json(api.cluster_job_status(payload["job_id"]))
            elif route == "/api/cancel-cluster-job":
                self._send_json(api.cancel_cluster_job(payload["job_id"]))
            elif route == "/api/fetch-from-cluster":
                self._send_json(self._submit_fetch(payload).to_dict())
            elif route == "/api/shutdown":
                self._shutdown()
            else:
                self._send_json({"error": "no such route"}, status=404)
        except KeyError as exc:
            self._send_error_json(ValueError(f"missing field {exc}"))
        except ValueError as exc:
            # Rejected settings are the caller's problem, not a server fault.
            # The guided form leans on that distinction to tell "you asked for
            # something impossible" apart from "the package broke".
            self._send_error_json(exc, status=400)
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            self._send_error_json(exc, status=500)

    def _receive_upload(
        self, query: dict[str, list[str]], *, folder: bool = False
    ) -> None:
        """Accept one dropped file and reply with the path it was stored at."""
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._send_error_json(ValueError("no file content was sent"))
            return
        if length > api.MAX_UPLOAD_BYTES:
            # Refused without reading the body: the point of a size limit is not
            # to spend memory finding out it was exceeded.
            self._send_error_json(
                ValueError(
                    f"that file is {length / 1e6:.0f} MB; the interface accepts "
                    f"up to {api.MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
                ),
                status=413,
            )
            return
        data = self.rfile.read(length)
        name = query.get("name", ["measurement"])[0]
        try:
            if folder:
                self._send_json(
                    api.save_folder_upload(
                        query.get("session", [""])[0],
                        query.get("folder", ["experiment"])[0],
                        name,
                        data,
                    )
                )
            else:
                self._send_json(api.save_upload(name, data))
        except ValueError as exc:
            self._send_error_json(exc, status=400)

    def _serve_result_file(self, path: str) -> None:
        """Serve a file this interface produced, so results open in the browser.

        A report is an HTML file whose whole purpose is to be looked at; making
        the user find it on disk to do that would undo the point of running the
        analysis from a page.
        """
        try:
            target = api.resolve_readable_file(path)
        except PermissionError as exc:
            self._send_error_json(exc, status=403)
            return
        content_types = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json",
            ".csv": "text/csv; charset=utf-8",
            ".png": "image/png",
            ".md": "text/plain; charset=utf-8",
            ".yaml": "text/plain; charset=utf-8",
            ".yml": "text/plain; charset=utf-8",
            ".txt": "text/plain; charset=utf-8",
        }
        body = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type",
            content_types.get(target.suffix.lower(), "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _shutdown(self) -> None:
        """Stop the server at the browser's request.

        Closing a tab leaves the process running, which is a real way to end up
        with an interface nobody can see and a port nobody can reuse. Any job
        still running is reported so the answer is informed rather than a
        surprise; shutdown itself is deferred to another thread because
        ``shutdown()`` blocks until the serving loop exits, and that loop is
        the one handling this request.
        """
        import threading

        running = [job for job in self.registry.list() if job["status"] == "running"]
        self._send_json(
            {
                "stopping": True,
                "abandoned_jobs": [job["label"] for job in running],
            }
        )
        threading.Thread(
            target=self.server.shutdown, name="hamlet-gui-shutdown", daemon=True
        ).start()

    # --- job submission ---
    def _submit_preview(self, payload: dict[str, Any]) -> api.GuiJob:
        """Simulate a couple of sample chains so the settings can be eyeballed.

        A job rather than a direct call: one chain is around a minute, which is
        far too long to hold a request open.
        """
        form = payload["form"]
        n_samples = int(payload.get("n_samples", 2))

        def work(token: Any) -> Any:
            return api.preview_samples(form, n_samples=n_samples, should_stop=token)

        return self.registry.submit("preview", "sample simulation", work)

    def _submit_screening(self, payload: dict[str, Any]) -> api.GuiJob:
        config_path = payload["config_path"]
        verify = bool(payload.get("verify_symmetric", False))

        def work(token: Any) -> Any:
            from ..cancellation import check_cancelled
            from ..dmi_design import (
                format_screening_table,
                load_screening_config,
                screen_dmi_designs,
            )
            from ..simulation import DmrgpySimulator

            designs, protocol = load_screening_config(config_path)
            print(f"screening {len(designs)} candidate design(s)")

            def report(item: Any) -> None:
                print(f"  {item.design.name}: {item.imprint:.3e} ({item.verdict})")
                # After reporting, so a stopped screen still shows the verdict
                # of the candidate it had just finished.
                check_cancelled(token, "the screening")

            results = screen_dmi_designs(
                designs,
                DmrgpySimulator(dynamics_mode="ED"),
                protocol,
                skip_symmetric=not verify,
                progress=report,
            )
            print()
            print(format_screening_table(results))
            return [
                {
                    "label": item.design.name,
                    "imprint": item.imprint,
                    "verdict": item.verdict,
                    "breaks_symmetry": item.predicted_to_break_symmetry,
                }
                for item in results
            ]

        return self.registry.submit(
            "screening", f"screen {Path(config_path).name}", work
        )

    def _submit_to_cluster(self, payload: dict[str, Any]) -> api.GuiJob:
        """Staging is a file copy that can take minutes, so it is a job."""
        config_path = payload["config_path"]

        def work() -> Any:
            return api.submit_to_cluster(config_path)

        return self.registry.submit(
            "cluster", f"send {Path(config_path).parent.name} to the cluster", work
        )

    def _submit_fetch(self, payload: dict[str, Any]) -> api.GuiJob:
        project_dir = payload["project_dir"]

        def work() -> Any:
            return api.fetch_from_cluster(project_dir)

        return self.registry.submit(
            "fetch", f"fetch {Path(project_dir).name} from the cluster", work
        )

    def _submit_analysis(self, payload: dict[str, Any]) -> api.GuiJob:
        config_path = payload["config_path"]

        def work(token: Any) -> Any:
            return api.run_analysis(config_path, should_stop=token)

        return self.registry.submit(
            "analysis", f"analyse {Path(config_path).parent.name}", work
        )

    def _submit_project(self, payload: dict[str, Any]) -> api.GuiJob:
        """Run a project exactly as `hamlet run` would.

        The stages are the library's, not the interface's: a run started here
        and the same configuration run from the command line have to do the
        same thing, or the path printed on the plan is a lie.
        """
        config_path = payload["config_path"]

        def work(token: Any) -> Any:
            from ..project import HamiltonianLearningProject

            api.use_headless_plotting()
            project = HamiltonianLearningProject.from_config(config_path)
            outcome = project.run(
                progress=lambda done, total: print(f"  {done} of {total} chains"),
                should_stop=token,
            )
            result: dict[str, Any] = {
                "kind": "analysis" if outcome.report_path else "training",
                "artifact_path": str(outcome.artifact_path),
                "status": str(outcome.status),
                "cutoff_mev": outcome.selected_cutoff_mev,
            }
            if outcome.report_path is not None:
                result["report_html"] = str(outcome.report_path)
                result["couplings_csv"] = str(outcome.analysis_dir / "couplings.csv")
                result["summary_png"] = str(outcome.analysis_dir / "summary.png")
            elif project.training_run is not None:
                metrics = project.training_run.metrics
                result["validation_mae_mev"] = metrics["validation"]["ensemble"]["mae"]
                result["test_mae_mev"] = metrics["test"]["ensemble"]["mae"]
            if project.tuning_report is not None:
                result["tuning"] = {
                    "backend": project.tuning_report.backend,
                    "kept_defaults": project.tuning_report.kept_defaults,
                    "best_options": project.tuning_report.best_options,
                    "improvement_mev": project.tuning_report.improvement_mev,
                    "report_path": str(project.config.output_dir / "tuning.json"),
                }
            print(f"done; artifact written to {outcome.artifact_path}")
            return result

        return self.registry.submit(
            "project", f"run {Path(config_path).name}", work
        )


DEFAULT_PORT = 8765
# How many ports past the default to try before giving up. Small on purpose: if
# this many are taken, something is wrong that a wider scan would only hide.
PORT_SEARCH_LIMIT = 12


def _interface_already_running(host: str, port: int, timeout: float = 0.5) -> bool:
    """Whether a HamLeT interface is already answering on this address.

    Running ``hamlet gui`` twice is an easy thing to do, and the useful reply is
    "it is already open, here is the link" rather than a bind error.
    """
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/overview", timeout=timeout
        ) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return False
    return isinstance(payload, dict) and "situations" in payload


def _open_browser(url: str) -> None:
    """Launch a browser without ever blocking the caller.

    ``webbrowser.open`` can block for a long time, or hang outright, when it
    resolves to a console browser or a launcher that waits on the terminal. It
    runs before ``serve_forever``, so a hang there leaves the socket bound and
    listening while nothing is ever accepted -- the server looks started and
    answers nothing. A daemon thread keeps that failure mode impossible.
    """
    import threading
    import webbrowser

    def launch() -> None:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - the printed URL still works
            pass

    threading.Thread(target=launch, name="hamlet-gui-browser", daemon=True).start()


def _has_display() -> bool:
    import os

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def serve(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    open_browser: bool = True,
) -> None:
    """Run the interface until interrupted.

    ``port`` of ``None`` means "the default, or the next free one after it",
    which keeps a second terminal from failing over a port collision. An
    explicit port is honoured strictly, because someone who chose a port
    usually needs that one.
    """
    # Jobs run on worker threads and every figure they make is written to a
    # file, so the process is settled onto a non-interactive backend up front
    # rather than discovering the problem inside an hours-long run.
    api.use_headless_plotting()

    requested = port
    first = DEFAULT_PORT if port is None else int(port)
    candidates = (
        range(first, first + PORT_SEARCH_LIMIT) if requested is None else (first,)
    )

    handler = type("Handler", (_Handler,), {"registry": api.JobRegistry()})
    httpd = None
    for candidate in candidates:
        try:
            httpd = ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            if _interface_already_running(host, candidate):
                url = f"http://{host}:{candidate}/"
                print(f"A HamLeT interface is already running on {url}")
                print("Opening that one instead of starting a second.")
                if open_browser:
                    _open_browser(url)
                return
            continue
        else:
            break

    if httpd is None:
        if requested is not None:
            # Deliberately does not claim what is holding the port. A suspended
            # `hamlet gui` (Ctrl+Z) keeps its socket bound while answering
            # nothing, so "not a HamLeT interface" would be actively wrong.
            raise SystemExit(
                f"Port {first} on {host} is in use by a process that is not "
                f"responding.\n"
                f"If you started an interface earlier and suspended it, bring it "
                f"back with `fg`, or list suspended jobs with `jobs`.\n"
                f"Otherwise start on another port:\n"
                f"    hamlet gui --port {first + 1}"
            )
        raise SystemExit(
            f"Ports {first} to {first + PORT_SEARCH_LIMIT - 1} on {host} are all "
            f"in use.\nPick one explicitly with: hamlet gui --port <number>"
        )

    with httpd:
        actual = httpd.server_address[1]
        url = f"http://{host}:{actual}/"
        if requested is None and actual != DEFAULT_PORT:
            print(f"Port {DEFAULT_PORT} was busy, so this is on {actual} instead.")
        print(f"HamLeT interface on {url}")
        print("This is a local server; nothing leaves your machine.")
        print("To stop it: Ctrl+C here, or the Stop button on the page.")
        print("Closing the browser tab does not stop it.")
        if open_browser and _has_display():
            _open_browser(url)
        elif open_browser:
            # Common on a cluster login node, where there is no browser to open.
            print()
            print("No display detected, so no browser was opened. If you are")
            print("connected over SSH, forward the port and open the URL locally:")
            print(f"    ssh -N -L {actual}:{host}:{actual} <this-host>")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
