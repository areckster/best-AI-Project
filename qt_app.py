"""Qt desktop shell for the chat application.

This module launches the FastAPI backend (if it is not already running)
and embeds the existing web UI inside a Qt WebEngine view so that the
desktop experience matches the HTML client feature-for-feature.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from typing import Optional

from PyQt6.QtCore import QTimer, QUrl, Qt
from PyQt6.QtWidgets import QApplication
from PyQt6.QtWebEngineWidgets import QWebEngineView


APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
APP_URL = f"http://{APP_HOST}:{APP_PORT}/"


def _ensure_webengine_env() -> None:
    """Prepare environment variables required by Qt WebEngine."""

    # Running as root inside containers requires the Chromium sandbox to be
    # disabled. Keep existing user-provided flags intact.
    flags = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").strip()
    extra = "--no-sandbox"
    if extra not in flags.split():
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = f"{flags} {extra}".strip()

    # Share the OpenGL context so the embedded browser initialises cleanly.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)


def _is_server_up() -> bool:
    """Return True if the backend responds to the health endpoint."""

    try:
        with urllib.request.urlopen(f"{APP_URL}api/health", timeout=1.5):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _wait_for_server(timeout: float = 15.0) -> bool:
    """Poll the backend until it becomes reachable or ``timeout`` expires."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_server_up():
            return True
        time.sleep(0.2)
    return False


def _start_server() -> Optional[subprocess.Popen]:
    """Start the FastAPI backend via uvicorn and return the process handle."""

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        APP_HOST,
        "--port",
        str(APP_PORT),
        "--workers",
        "1",
        "--loop",
        "uvloop",
        "--http",
        "httptools",
    ]

    env = os.environ.copy()
    proc = subprocess.Popen(cmd, env=env)
    if _wait_for_server():
        return proc

    # Server failed to start within the timeout; clean up and return None.
    with suppress(ProcessLookupError):
        proc.terminate()
    return None


def main() -> int:
    _ensure_webengine_env()

    server_proc: Optional[subprocess.Popen] = None
    started_server = False

    if not _is_server_up():
        server_proc = _start_server()
        started_server = server_proc is not None
        if not started_server:
            print("Failed to launch backend server.", file=sys.stderr)
            return 1

    app = QApplication(sys.argv)

    view = QWebEngineView()
    view.setWindowTitle("Chat Client")
    view.resize(1280, 800)
    view.load(QUrl(APP_URL))
    view.show()

    def _cleanup() -> None:
        if started_server and server_proc and server_proc.poll() is None:
            with suppress(ProcessLookupError):
                server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    server_proc.kill()

    app.aboutToQuit.connect(_cleanup)  # type: ignore[arg-type]

    # Periodically refresh the view in case the server takes a little longer
    # to boot on first launch.
    if not _is_server_up():
        def _retry_load() -> None:
            if _is_server_up():
                view.load(QUrl(APP_URL))

        timer = QTimer(view)
        timer.setInterval(500)
        timer.timeout.connect(_retry_load)  # type: ignore[arg-type]
        timer.start()

    exit_code = app.exec()
    _cleanup()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
