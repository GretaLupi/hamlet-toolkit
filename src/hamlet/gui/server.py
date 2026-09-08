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
                self._send_json(api.describe_published_models())
            elif route == "/api/model-card":
                self._send_json(api.read_model_card(query["name"][0]))
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
        route = urlparse(self.path).path
        try:
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
            elif route == "/api/save-config":
                self._send_json(
                    api.write_config_text(payload["path"], payload["text"])
                )
            elif route == "/api/screening-preview":
                self._send_json(api.screening_preview(payload["config_path"]))
            elif route == "/api/run-screening":
                self._send_json(self._submit_screening(payload).to_dict())
            elif route == "/api/run-project":
                self._send_json(self._submit_project(payload).to_dict())
            else:
                self._send_json({"error": "no such route"}, status=404)
        except KeyError as exc:
            self._send_error_json(ValueError(f"missing field {exc}"))
        except Exception as exc:  # noqa: BLE001 - surfaced to the browser
            self._send_error_json(exc, status=500)

    # --- job submission ---
    def _submit_screening(self, payload: dict[str, Any]) -> api.GuiJob:
        config_path = payload["config_path"]
        verify = bool(payload.get("verify_symmetric", False))

        def work() -> Any:
            from ..dmi_design import (
                format_screening_table,
                load_screening_config,
                screen_dmi_designs,
            )
            from ..simulation import DmrgpySimulator

            designs, protocol = load_screening_config(config_path)
            print(f"screening {len(designs)} candidate design(s)")
            results = screen_dmi_designs(
                designs,
                DmrgpySimulator(dynamics_mode="ED"),
                protocol,
                skip_symmetric=not verify,
                progress=lambda item: print(
                    f"  {item.design.name}: {item.imprint:.3e} ({item.verdict})"
                ),
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

    def _submit_project(self, payload: dict[str, Any]) -> api.GuiJob:
        config_path = payload["config_path"]

        def work() -> Any:
            from ..project import HamiltonianLearningProject

            project = HamiltonianLearningProject.from_config(config_path)
            outcome = project.run()
            return {"summary": str(outcome)[:2000]}

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
        print("Press Ctrl+C to stop.")
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
