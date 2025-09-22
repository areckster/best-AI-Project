"""Qt desktop client that mirrors the behaviour of the HTML chat UI.

The goal of this module is to offer a first-class desktop experience without
embedding the web front-end inside a `QWebEngineView`.  The widgets here
reimplement the chat transcript, thinking/telemetry panes, session management,
document search, attachments, and settings management that the browser client
provides.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from markdown_it import MarkdownIt
from PyQt6.QtCore import QEvent, QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QCloseEvent, QTextOption
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_HOST = os.environ.get("APP_HOST", "127.0.0.1")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
REASONING_OFF_MODEL = os.environ.get("REASONING_OFF_MODEL", "gemma3:4b-it-qat")

SESSIONS_FILE = Path.home() / ".best_ai_sessions.json"


def _is_server_up() -> bool:
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{APP_URL}/api/health")
            return r.status_code == 200
    except httpx.HTTPError:
        return False


def _wait_for_server(timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_server_up():
            return True
        time.sleep(0.2)
    return False


def _start_server() -> Optional[subprocess.Popen]:
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

    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    return None


class MarkdownRenderer:
    """Thin wrapper that mirrors the markdown-it configuration from the web UI."""

    def __init__(self) -> None:
        self.md = (
            MarkdownIt("commonmark", {"breaks": True, "html": False})
            .enable("table")
            .enable("strikethrough")
        )

    def to_html(self, text: str) -> str:
        rendered = self.md.render(text or "")
        # Ensure code blocks retain a copy button affordance similar to web UI.
        return rendered.replace(
            "<pre><code",
            '<div class="codewrap"><pre><code',
        ).replace("</code></pre>", "</code></pre></div>")


markdown_renderer = MarkdownRenderer()


@dataclass
class Attachment:
    name: str
    path: Optional[str] = None
    text: Optional[str] = None
    mime: Optional[str] = None
    size: int = 0

    @property
    def is_text(self) -> bool:
        return bool(self.text)


class MessageWidget(QWidget):
    """Widget representing a single user/assistant bubble."""

    def __init__(self, role: str, content: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.role = role
        self.setObjectName("message")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        meta = QLabel("You" if role == "user" else "Assistant")
        meta.setProperty("role", role)
        meta.setStyleSheet(
            "QLabel[role='user'] { font-weight: 600; color: #9ec5ff; }"
            "QLabel[role='assistant'] { font-weight: 600; color: #a9b9cc; }"
        )
        layout.addWidget(meta)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.browser.setStyleSheet(
            "QTextBrowser {"
            " background-color: rgba(14,22,34,0.55);"
            " border: 1px solid rgba(158,197,255,0.08);"
            " border-radius: 12px;"
            " padding: 10px;"
            " color: #e6edf6;"
            " font-size: 14px;"
            " }"
            "QTextBrowser QWidget { color: #e6edf6; }"
        )
        layout.addWidget(self.browser)

        self.set_content(content)

    def set_content(self, content: str) -> None:
        raw = content or ""
        html = markdown_renderer.to_html(raw)
        self.browser.setHtml(html)
        self.browser.setProperty("raw", raw)

    def append_content(self, content: str) -> None:
        raw = (self.browser.property("raw") or "") + content
        self.set_content(raw)


class ThinkingPanel(QGroupBox):
    """Displays reasoning/tool-call events for an assistant reply."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("Thinking", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setFlat(True)
        self.setStyleSheet(
            "QGroupBox { border: 1px solid rgba(158,197,255,0.12);"
            " border-radius: 10px; margin-top: 12px; padding: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;"
            " padding: 0 4px; color: #9ec5ff; }"
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        self.setLayout(layout)

        self.summary_label = QLabel("Idle")
        self.summary_label.setStyleSheet("color: #a9b9cc; font-size: 12px;")
        layout.addWidget(self.summary_label)

        self.log_view = QTextBrowser()
        self.log_view.setStyleSheet(
            "QTextBrowser { background: rgba(14,22,34,0.6); border: none; color: #e6edf6; }"
        )
        self.log_view.setOpenExternalLinks(True)
        layout.addWidget(self.log_view)

        self._log_lines: List[str] = []
        self._thinking_buffer: List[str] = []

    def append_log(self, line: str) -> None:
        self._log_lines.append(line)
        html = "<br/>".join(self._log_lines)
        self.log_view.setHtml(html)

    def append_thinking(self, text: str) -> None:
        if not text.strip():
            return
        self._thinking_buffer.append(text)
        joined = " ".join(self._thinking_buffer)[-2000:]
        self.summary_label.setText(joined.strip())

    def flush_thinking(self) -> None:
        if not self._thinking_buffer:
            return
        chunk = " ".join(self._thinking_buffer)
        self.append_log(f"<em>{chunk}</em>")
        self._thinking_buffer = []
        self.summary_label.setText("Idle")


@dataclass
class ThinkingContext:
    panel: ThinkingPanel
    buffer: str = ""
    in_think: bool = False
    sawThinkOpen: bool = False
    sawThinkClose: bool = False

    def consume_delta(self, delta: str) -> str:
        out = []
        i = 0
        while i < len(delta):
            if not self.in_think:
                open_idx = delta.find("<think>", i)
                close_idx = delta.find("</think>", i)
                if open_idx == -1 and close_idx == -1:
                    out.append(delta[i:])
                    break
                if open_idx != -1 and (close_idx == -1 or open_idx < close_idx):
                    out.append(delta[i:open_idx])
                    self.in_think = True
                    self.sawThinkOpen = True
                    i = open_idx + len("<think>")
                    continue
                if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                    i = close_idx + len("</think>")
                    continue
            else:
                close_idx = delta.find("</think>", i)
                if close_idx == -1:
                    chunk = delta[i:]
                    self.panel.append_thinking(chunk)
                    self.buffer += chunk
                    break
                chunk = delta[i:close_idx]
                if chunk:
                    self.panel.append_thinking(chunk)
                    self.buffer += chunk
                self.panel.flush_thinking()
                self.in_think = False
                self.sawThinkClose = True
                i = close_idx + len("</think>")
                continue
            i += 1
        return "".join(out)


class SessionStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sessions: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self.sessions = []
            return
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            self.sessions = []
            return
        if isinstance(data, list):
            self.sessions = data
        else:
            self.sessions = []

    def save(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self.sessions, fh, indent=2)
        except Exception:
            pass

    def add(self, title: str, history: List[Dict[str, Any]]) -> None:
        self.sessions.append({
            "id": int(time.time() * 1000),
            "title": title or "Untitled chat",
            "history": history,
            "ts": time.time(),
        })
        self.save()

    def delete(self, idx: int) -> None:
        if 0 <= idx < len(self.sessions):
            del self.sessions[idx]
            self.save()


class ChatWorker(QThread):
    delta = pyqtSignal(dict)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, payload: Dict[str, Any], parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.payload = payload

    def run(self) -> None:
        try:
            with httpx.Client(base_url=APP_URL, timeout=None) as client:
                with client.stream("POST", "/api/chat/stream", json=self.payload) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    for chunk in resp.iter_text():
                        if not chunk:
                            continue
                        buffer += chunk
                        while "\n\n" in buffer:
                            part, buffer = buffer.split("\n\n", 1)
                            line = part.strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data:
                                continue
                            try:
                                event = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if event.get("type") == "done":
                                self.finished.emit(event)
                            else:
                                self.delta.emit(event)
        except Exception as exc:  # pragma: no cover - network errors
            self.failed.emit(str(exc))


class VisionWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        prompt: str,
        history: List[Dict[str, Any]],
        image: Attachment,
        reasoning: bool,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.prompt = prompt
        self.history = history
        self.image = image
        self.reasoning = reasoning

    def run(self) -> None:
        if not self.image.path:
            self.failed.emit("Image attachment missing")
            return
        data = {
            "prompt": self.prompt,
            "model": REASONING_OFF_MODEL,
            "reasoning": "on" if self.reasoning else "off",
            "history": json.dumps(self.history),
        }
        try:
            with open(self.image.path, "rb") as fh:
                files = {"image": (self.image.name, fh, self.image.mime or "application/octet-stream")}
                with httpx.Client(base_url=APP_URL, timeout=120.0) as client:
                    resp = client.post("/api/gemma3", data=data, files=files)
                    resp.raise_for_status()
                    self.finished.emit(resp.json())
        except Exception as exc:
            self.failed.emit(str(exc))


class HealthWorker(QThread):
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            with httpx.Client(base_url=APP_URL, timeout=5.0) as client:
                resp = client.get("/api/health")
                resp.raise_for_status()
                self.finished.emit(resp.json())
        except Exception as exc:
            self.failed.emit(str(exc))


class DocSearchWorker(QThread):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, query: str, rerank: bool, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.query = query
        self.rerank = rerank

    def run(self) -> None:
        payload = {"query": self.query, "k": 8, "rerank": self.rerank}
        try:
            with httpx.Client(base_url=APP_URL, timeout=30.0) as client:
                resp = client.post("/api/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results") if isinstance(data, dict) else []
                if isinstance(results, list):
                    self.finished.emit(results)
                else:
                    self.finished.emit([])
        except Exception as exc:
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    def __init__(self, settings: Dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings

        layout = QVBoxLayout(self)

        self.model_tag = QLineEdit(settings.get("model_tag", ""))
        layout.addWidget(QLabel("Model tag"))
        layout.addWidget(self.model_tag)

        self.dynamic_ctx = QCheckBox("Dynamic context")
        self.dynamic_ctx.setChecked(settings.get("dynamic_ctx", True))
        layout.addWidget(self.dynamic_ctx)

        ctx_layout = QHBoxLayout()
        layout.addLayout(ctx_layout)
        ctx_layout.addWidget(QLabel("Max context"))
        self.max_ctx = QSpinBox()
        self.max_ctx.setMaximum(200000)
        self.max_ctx.setValue(int(settings.get("max_ctx", 40000)))
        ctx_layout.addWidget(self.max_ctx)

        ctx_layout2 = QHBoxLayout()
        layout.addLayout(ctx_layout2)
        ctx_layout2.addWidget(QLabel("Static context"))
        self.num_ctx = QSpinBox()
        self.num_ctx.setMaximum(200000)
        self.num_ctx.setValue(int(settings.get("num_ctx", 8192)))
        ctx_layout2.addWidget(self.num_ctx)

        layout.addWidget(QLabel("System prompt"))
        self.system_prompt = QTextEdit(settings.get("system", ""))
        self.system_prompt.setMinimumHeight(120)
        layout.addWidget(self.system_prompt)

        adv = QGroupBox("Advanced")
        adv_layout = QVBoxLayout(adv)
        self.temperature = QLineEdit(str(settings.get("temperature", 0.9)))
        self.top_p = QLineEdit(str(settings.get("top_p", 0.9)))
        self.top_k = QLineEdit(str(settings.get("top_k", 100)))
        self.num_predict = QLineEdit(str(settings.get("num_predict", "")))
        self.seed = QLineEdit(str(settings.get("seed", "")))
        adv_layout.addWidget(QLabel("Temperature"))
        adv_layout.addWidget(self.temperature)
        adv_layout.addWidget(QLabel("top_p"))
        adv_layout.addWidget(self.top_p)
        adv_layout.addWidget(QLabel("top_k"))
        adv_layout.addWidget(self.top_k)
        adv_layout.addWidget(QLabel("Max output tokens"))
        adv_layout.addWidget(self.num_predict)
        adv_layout.addWidget(QLabel("Seed"))
        adv_layout.addWidget(self.seed)
        layout.addWidget(adv)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> Dict[str, Any]:
        return {
            "model_tag": self.model_tag.text().strip(),
            "dynamic_ctx": self.dynamic_ctx.isChecked(),
            "max_ctx": self.max_ctx.value(),
            "num_ctx": self.num_ctx.value(),
            "system": self.system_prompt.toPlainText(),
            "temperature": float(self.temperature.text() or 0.9),
            "top_p": float(self.top_p.text() or 0.9),
            "top_k": int(self.top_k.text() or 100),
            "num_predict": self.num_predict.text().strip(),
            "seed": self.seed.text().strip(),
        }


class DocSearchDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Search Docs")
        self.resize(640, 520)
        layout = QVBoxLayout(self)

        query_layout = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Search docs…")
        query_layout.addWidget(self.query_edit)
        self.search_btn = QPushButton("Search")
        query_layout.addWidget(self.search_btn)
        layout.addLayout(query_layout)

        self.rerank = QCheckBox("Re-rank results")
        layout.addWidget(self.rerank)

        self.results = QListWidget()
        layout.addWidget(self.results)

        self.preview = QTextBrowser()
        self.preview.setMinimumHeight(160)
        layout.addWidget(self.preview)

        self.results.itemSelectionChanged.connect(self._update_preview)

    def populate(self, results: List[Dict[str, Any]]) -> None:
        self.results.clear()
        for res in results:
            title = res.get("title") or res.get("doc_id") or "Result"
            host = res.get("source") or "doc"
            item = QListWidgetItem(f"{title} — {host}")
            item.setData(Qt.ItemDataRole.UserRole, res)
            self.results.addItem(item)

    def _update_preview(self) -> None:
        items = self.results.selectedItems()
        if not items:
            self.preview.clear()
            return
        data = items[0].data(Qt.ItemDataRole.UserRole) or {}
        text = str(data.get("preview") or "")[:3000]
        meta = []
        if data.get("source"):
            meta.append(data["source"])
        if data.get("uri"):
            meta.append(data["uri"])
        header = " • ".join(meta)
        html = f"<h3>{data.get('title') or data.get('doc_id') or 'Document'}</h3>"
        if header:
            html += f"<p><em>{header}</em></p>"
        html += f"<pre>{text}</pre>"
        self.preview.setHtml(html)


class ChatMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Best AI Project – Desktop")
        self.resize(1280, 860)

        self.history: List[Dict[str, Any]] = []
        self.attachments: List[Attachment] = []
        self.image_attachment: Optional[Attachment] = None
        self.sessions = SessionStore(SESSIONS_FILE)
        self.settings = {
            "model_tag": "qwen3:4b-thinking-2507-q4_K_M",
            "dynamic_ctx": True,
            "max_ctx": 40000,
            "num_ctx": 8192,
            "system": "",
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 100,
            "num_predict": "",
            "seed": "",
        }
        self.reasoning_enabled = True

        self.streaming = False
        self.thinking: Optional[ThinkingContext] = None

        self._build_ui()
        QTimer.singleShot(100, self._refresh_health)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(8)
        main_layout.addLayout(header)

        self.ctx_label = QLabel("ctx: auto")
        header.addWidget(self.ctx_label)

        header.addStretch(1)

        self.telemetry_label = QLabel("tkn: —/— • — ms")
        header.addWidget(self.telemetry_label)

        self.doc_search_box = QLineEdit()
        self.doc_search_box.setPlaceholderText("Search docs…")
        self.doc_search_box.returnPressed.connect(self._open_doc_search)
        header.addWidget(self.doc_search_box)

        doc_btn = QPushButton("Search")
        doc_btn.clicked.connect(self._open_doc_search)
        header.addWidget(doc_btn)

        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.clicked.connect(self._new_chat)
        header.addWidget(self.new_chat_btn)

        # Chat transcript area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.addStretch(1)
        self.chat_scroll.setWidget(self.chat_container)
        main_layout.addWidget(self.chat_scroll, 1)

        # Composer
        composer = QHBoxLayout()
        composer.setSpacing(6)
        main_layout.addLayout(composer)

        left = QVBoxLayout()
        composer.addLayout(left, 1)

        self.attachment_label = QLabel("Attachments: none")
        left.addWidget(self.attachment_label)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Ask anything…")
        self.prompt_edit.installEventFilter(self)
        left.addWidget(self.prompt_edit, 1)

        right = QVBoxLayout()
        composer.addLayout(right)

        attach_btn = QPushButton("Attach files…")
        attach_btn.clicked.connect(self._pick_files)
        right.addWidget(attach_btn)

        image_btn = QPushButton("Attach image…")
        image_btn.clicked.connect(self._pick_image)
        right.addWidget(image_btn)

        self.reason_btn = QPushButton("Thinking On")
        self.reason_btn.setCheckable(True)
        self.reason_btn.setChecked(True)
        self.reason_btn.clicked.connect(self._toggle_reasoning)
        right.addWidget(self.reason_btn)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._send_message)
        right.addWidget(send_btn)

        right.addStretch(1)

        # Sidebar for sessions / health
        dock = QWidget()
        dock_layout = QVBoxLayout(dock)
        dock_layout.setContentsMargins(12, 12, 12, 12)
        dock_layout.setSpacing(8)

        self.session_list = QListWidget()
        dock_layout.addWidget(QLabel("Saved chats"))
        dock_layout.addWidget(self.session_list, 1)
        self.session_list.itemActivated.connect(self._load_session)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Archive current")
        save_btn.clicked.connect(self._archive_session)
        btn_row.addWidget(save_btn)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._delete_session)
        btn_row.addWidget(del_btn)
        dock_layout.addLayout(btn_row)

        self.health_label = QLabel("Checking backend…")
        dock_layout.addWidget(self.health_label)

        settings_btn = QPushButton("Settings…")
        settings_btn.clicked.connect(self._open_settings)
        dock_layout.addWidget(settings_btn)

        dock_layout.addStretch(1)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)
        left_wrapper = QWidget()
        left_wrapper.setLayout(main_layout)
        splitter.addWidget(left_wrapper)
        splitter.addWidget(dock)
        splitter.setSizes([900, 320])

        super().setCentralWidget(splitter)

        self._refresh_sessions()

        # Menu for exporting chat
        export_act = QAction("Export chat", self)
        export_act.triggered.connect(self._export_chat)
        self.menuBar().addAction(export_act)

    # ----------------------------------------------------------------- utils --
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        if obj is self.prompt_edit and event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Return and not event.modifiers():
                self._send_message()
                return True
        return super().eventFilter(obj, event)

    def _add_message(self, role: str, content: str) -> MessageWidget:
        widget = MessageWidget(role, content)
        item = QWidget()
        lay = QVBoxLayout(item)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(widget)
        if role != "user":
            thinking = ThinkingPanel()
            lay.addWidget(thinking)
            widget.thinking_panel = thinking  # type: ignore[attr-defined]
        index = self.chat_layout.count() - 1
        self.chat_layout.insertWidget(index, item)
        QTimer.singleShot(20, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))
        return widget

    def _update_attachments_label(self) -> None:
        parts = [att.name for att in self.attachments]
        if self.image_attachment:
            parts.append(f"Vision: {self.image_attachment.name}")
        self.attachment_label.setText("Attachments: " + (", ".join(parts) if parts else "none"))

    def _toggle_reasoning(self) -> None:
        self.reasoning_enabled = self.reason_btn.isChecked()
        self.reason_btn.setText("Thinking On" if self.reasoning_enabled else "Thinking Off")

    def _pick_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files")
        for path in paths:
            if not path:
                continue
            stat = os.stat(path)
            att = Attachment(name=os.path.basename(path), path=path, size=stat.st_size)
            if att.name.lower().endswith((".txt", ".md", ".json", ".csv")):
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        att.text = fh.read()
                        att.mime = "text/plain"
                except Exception:
                    pass
            self.attachments.append(att)
        self._update_attachments_label()

    def _pick_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            filter="Images (*.png *.jpg *.jpeg *.webp)",
        )
        if not path:
            return
        stat = os.stat(path)
        self.image_attachment = Attachment(
            name=os.path.basename(path),
            path=path,
            size=stat.st_size,
            mime="image/jpeg" if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg") else "image/png",
        )
        self._update_attachments_label()

    def _compose_user_message(self, prompt: str) -> Tuple[str, str]:
        display_parts: List[str] = []
        history_parts: List[str] = []
        if prompt:
            display_parts.append(prompt)
            history_parts.append(prompt)
        for att in self.attachments:
            if att.is_text and att.text:
                block = f"**File: {att.name}**\n```\n{att.text}\n```"
                display_parts.append(block)
                history_parts.append(block)
            else:
                note = f"(Attached file: {att.name})"
                display_parts.append(note)
                history_parts.append(note)
        if self.image_attachment:
            display_parts.append(f"![{self.image_attachment.name}](attachment)")
            size_kb = max(1, int(self.image_attachment.size / 1024))
            history_parts.append(
                f"[Image uploaded: {self.image_attachment.name} • {self.image_attachment.mime} • {size_kb}kb]"
            )
        return "\n\n".join(display_parts), "\n\n".join(history_parts)

    def _ingest_attachments(self) -> str:
        if not self.attachments:
            return ""
        import io

        files: List[Tuple[str, Tuple[str, Any, str]]] = []
        handles: List[Any] = []
        for att in self.attachments:
            if att.path:
                fh = open(att.path, "rb")
                handles.append(fh)
                files.append(
                    (
                        "files",
                        (
                            att.name,
                            fh,
                            att.mime or "application/octet-stream",
                        ),
                    )
                )
            elif att.text is not None:
                buf = io.BytesIO(att.text.encode("utf-8"))
                handles.append(buf)
                files.append(("files", (att.name or "attachment.txt", buf, "text/plain")))
        if not files:
            return ""
        try:
            with httpx.Client(base_url=APP_URL, timeout=120.0) as client:
                resp = client.post("/api/upload", files=files)
                resp.raise_for_status()
                data = resp.json()
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass
        entries = []
        for item in data.get("files", []):
            if item and item.get("ok"):
                entries.append(item)
        if not entries:
            return ""
        refs = ", ".join((entry.get("title") or entry.get("name") or "file") for entry in entries)
        file_paths = [
            {
                "title": entry.get("title") or entry.get("name") or "",
                "path": entry.get("path") or "",
            }
            for entry in entries
            if entry.get("path")
        ]
        if self.reasoning_enabled:
            note = (
                f"Attached files ingested: {refs}. Use search_docs first for targeted facts (filters.source='file'). "
                "For quick orientation, you may call summarize_file with a specific {path}. "
                f"Files: {json.dumps(file_paths)}"
            )
        else:
            note = (
                f"Attached files ingested: {refs}. Relevant excerpts may be provided below for grounding. "
                f"Files: {json.dumps(file_paths)}"
            )
        return note

    def _send_message(self) -> None:
        if self.streaming:
            QMessageBox.warning(self, "Streaming", "Wait for the current reply to finish.")
            return

        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt and not self.attachments and not self.image_attachment:
            return

        display, history = self._compose_user_message(prompt)
        ingested_note = ""
        if self.attachments:
            try:
                ingested_note = self._ingest_attachments()
            except Exception as exc:
                QMessageBox.warning(self, "Upload failed", str(exc))
                return

        if display:
            self._add_message("user", display or "(submitted)")
        self.history.append({"role": "user", "content": history or display})

        image_att = self.image_attachment

        self.prompt_edit.clear()
        self.attachments = []
        self.image_attachment = None
        self._update_attachments_label()

        if image_att:
            self._send_vision_message(prompt, image_att)
            return

        payload_messages = self.history.copy()
        if ingested_note:
            payload_messages.insert(0, {"role": "system", "content": ingested_note})
        system_prompt = self.settings.get("system") or ""
        if system_prompt:
            payload_messages.insert(0, {"role": "system", "content": system_prompt})

        options = {
            "temperature": self.settings.get("temperature", 0.9),
            "top_p": self.settings.get("top_p", 0.9),
            "top_k": self.settings.get("top_k", 100),
            "num_predict": self.settings.get("num_predict") or None,
            "seed": self.settings.get("seed") or None,
            "dynamic_ctx": self.settings.get("dynamic_ctx", True),
            "max_ctx": self.settings.get("max_ctx", 40000),
            "num_ctx": self.settings.get("num_ctx", 8192),
        }

        payload = {
            "messages": payload_messages,
            "settings": options,
            "system": system_prompt,
            "tools": self.reasoning_enabled,
        }

        model_tag = self.settings.get("model_tag")
        if model_tag:
            payload["model"] = model_tag

        self.streaming = True
        assistant_widget = self._add_message("assistant", "")
        thinking_panel: ThinkingPanel = getattr(assistant_widget, "thinking_panel")
        self.thinking = ThinkingContext(panel=thinking_panel)

        worker = ChatWorker(payload)
        worker.delta.connect(lambda evt: self._handle_delta(evt, assistant_widget))
        worker.finished.connect(lambda evt: self._handle_done(evt, assistant_widget, worker))
        worker.failed.connect(lambda err: self._handle_error(err, assistant_widget, worker))
        worker.start()
        self.worker = worker  # keep reference

    def _handle_delta(self, event: Dict[str, Any], widget: MessageWidget) -> None:
        etype = event.get("type")
        if etype == "delta":
            delta = str(event.get("delta") or "")
            if self.thinking:
                visible = self.thinking.consume_delta(delta)
            else:
                visible = delta
            widget.append_content(visible)
        elif etype == "tool_calls" and self.thinking:
            calls = event.get("tool_calls") or []
            for tc in calls:
                name = tc.get("function", {}).get("name") or tc.get("name") or "tool"
                args = tc.get("function", {}).get("arguments")
                self.thinking.panel.append_log(f"<strong>{name}</strong> {args}")
        elif etype == "tool_result" and self.thinking:
            name = event.get("name") or "tool"
            out = event.get("output")
            snippet = json.dumps(out, indent=2) if out else "(no output)"
            self.thinking.panel.append_log(f"<code>{name}</code><pre>{snippet}</pre>")
        elif etype == "error":
            self.thinking.panel.append_log(f"<span style='color:#f56b82'>Error: {event.get('message')}</span>")

    def _handle_done(self, event: Dict[str, Any], widget: MessageWidget, worker: ChatWorker) -> None:
        self.streaming = False
        if self.thinking:
            self.thinking.panel.flush_thinking()
        raw = widget.browser.property("raw") or ""
        self.history.append({"role": "assistant", "content": raw})
        worker.deleteLater()

    def _handle_error(self, err: str, widget: MessageWidget, worker: ChatWorker) -> None:
        self.streaming = False
        widget.append_content(f"\n**[error]** {err}")
        worker.deleteLater()

    def _send_vision_message(self, prompt: str, image: Attachment) -> None:
        assistant_widget = self._add_message("assistant", "Analyzing image…")
        self.streaming = True

        worker = VisionWorker(prompt, self.history.copy(), image, self.reasoning_enabled)
        worker.finished.connect(lambda data: self._handle_vision_done(data, assistant_widget, worker))
        worker.failed.connect(lambda err: self._handle_vision_error(err, assistant_widget, worker))
        worker.start()
        self.worker = worker

    def _handle_vision_done(self, data: Dict[str, Any], widget: MessageWidget, worker: VisionWorker) -> None:
        answer = (
            data.get("response")
            or data.get("reply")
            or data.get("message")
            or data.get("output")
            or ""
        )
        widget.set_content(answer or "(no response)")
        self.history.append({"role": "assistant", "content": answer})
        self.streaming = False
        worker.deleteLater()

    def _handle_vision_error(self, err: str, widget: MessageWidget, worker: VisionWorker) -> None:
        widget.set_content(f"**[error]** {err}")
        self.streaming = False
        worker.deleteLater()

    def _refresh_health(self) -> None:
        worker = HealthWorker()
        worker.finished.connect(self._set_health)
        worker.failed.connect(lambda err: self._set_health(None, err))
        worker.start()
        self._health_worker = worker

    def _set_health(self, payload: Optional[Dict[str, Any]], error: Optional[str] = None) -> None:
        if payload:
            ok = payload.get("ok")
            model = payload.get("model") or "unknown"
            if ok:
                self.health_label.setText(f"Backend ✓ • {model}")
            else:
                self.health_label.setText("Backend unreachable")
        else:
            self.health_label.setText(f"Backend error: {error}")

    def _open_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings.update(dlg.values())

    def _open_doc_search(self) -> None:
        query = self.doc_search_box.text().strip()
        if not query:
            return
        dlg = DocSearchDialog(self)
        dlg.query_edit.setText(query)

        worker_ref: Dict[str, DocSearchWorker] = {}

        def run_search() -> None:
            q = dlg.query_edit.text().strip()
            if not q:
                return
            worker = DocSearchWorker(q, dlg.rerank.isChecked())
            worker.finished.connect(lambda results: dlg.populate(results))
            worker.failed.connect(lambda err: QMessageBox.warning(self, "Search failed", err))
            worker.start()
            worker_ref["worker"] = worker

        dlg.search_btn.clicked.connect(run_search)
        dlg.query_edit.returnPressed.connect(run_search)
        dlg.rerank.stateChanged.connect(lambda _state: run_search())

        run_search()
        dlg.exec()
        if worker_ref.get("worker"):
            worker_ref["worker"].deleteLater()

    def _archive_session(self) -> None:
        if not self.history:
            QMessageBox.information(self, "Archive", "Nothing to archive yet.")
            return
        title = next((m["content"][:60] for m in self.history if m.get("role") == "user" and m.get("content")), "Untitled chat")
        self.sessions.add(title, self.history)
        self._refresh_sessions()

    def _refresh_sessions(self) -> None:
        self.session_list.clear()
        for sess in self.sessions.sessions:
            item = QListWidgetItem(sess.get("title") or "Untitled chat")
            item.setData(Qt.ItemDataRole.UserRole, sess)
            self.session_list.addItem(item)

    def _load_session(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        self.history = []
        for msg in data.get("history") or []:
            role = msg.get("role")
            content = msg.get("content") or ""
            self._add_message(role or "assistant", content)
            self.history.append({"role": role, "content": content})

    def _delete_session(self) -> None:
        row = self.session_list.currentRow()
        if row < 0:
            return
        self.sessions.delete(row)
        self._refresh_sessions()

    def _export_chat(self) -> None:
        if not self.history:
            QMessageBox.information(self, "Export", "No messages to export yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export chat",
            "chat_export.json",
            "JSON (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"history": self.history}, fh, indent=2)
        except Exception as exc:
            QMessageBox.warning(self, "Export failed", str(exc))

    def _new_chat(self) -> None:
        self.history = []
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.streaming:
            reply = QMessageBox.question(
                self,
                "Streaming",
                "A response is still streaming. Quit anyway?",
                QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        super().closeEvent(event)


def main() -> int:
    started_server = False
    server_proc: Optional[subprocess.Popen] = None
    if not _is_server_up():
        import contextlib

        server_proc = _start_server()
        started_server = server_proc is not None
        if not started_server:
            print("Failed to launch backend server", file=sys.stderr)
            return 1

    app = QApplication(sys.argv)

    win = ChatMainWindow()
    win.show()

    def _cleanup() -> None:
        if started_server and server_proc and server_proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    server_proc.kill()

    app.aboutToQuit.connect(_cleanup)  # type: ignore[arg-type]

    code = app.exec()
    _cleanup()
    return code


if __name__ == "__main__":
    raise SystemExit(main())

