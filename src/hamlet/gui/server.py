"""Localhost HTTP server for the browser interface.

Standard library only, deliberately: the audience installs this to analyse
measurements, not to maintain a web stack, and a GUI is not worth a dependency
that can break the science path. It binds to 127.0.0.1 because the operations
behind it read local files and start local compute.
"""

from __future__ import annotations

import json
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


def serve(
    *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    """Run the interface until interrupted."""
    handler = type("Handler", (_Handler,), {"registry": api.JobRegistry()})
    with ThreadingHTTPServer((host, port), handler) as httpd:
        url = f"http://{host}:{port}/"
        print(f"HamLeT interface on {url}")
        print("This is a local server; nothing leaves your machine.")
        print("Press Ctrl+C to stop.")
        if open_browser:
            import webbrowser

            # A failure to find a browser must not take the server down; the
            # printed URL is still usable.
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001
                pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
