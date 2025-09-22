#!/usr/bin/env python3
"""
macos_app.py — Native macOS UI for the chat server

This app provides a native shell with two modes:
- Native mode: Cocoa UI for chatting, settings, model management, attachments, export.
- Web mode (Exact Parity): Embeds the existing index.html via WKWebView for 100% feature parity.

Prerequisites (macOS):
- Python 3.9+
- PyObjC: pip install pyobjc
- WebKit: included with macOS

Run: python macos_app.py
"""

import os
import sys
import json
import threading
from typing import List, Dict, Any

import asyncio
import httpx

from Cocoa import (
    NSApplication, NSApp, NSWindow, NSWindowStyleMask, NSBackingStoreBuffered,
    NSObject, NSMakeRect, NSTitledWindowMask, NSClosableWindowMask, NSResizableWindowMask,
    NSView, NSScrollView, NSTextView, NSButton, NSBezelStyleRounded, NSSegmentedControl,
    NSSegmentedCell, NSSegmentedControlSegmentStyle, NSTextField, NSProgressIndicator,
    NSVisualEffectView, NSAppearanceNameVibrantDark, NSAppearance,
    NSLayoutConstraint, NSLayoutFormatAlignAllCenterY, NSLayoutConstraintOrientationHorizontal,
    NSLayoutRelationEqual, NSAlert, NSAlertStyleInformational, NSSavePanel, NSOpenPanel,
    NSColor, NSMenu, NSMenuItem, NSTextAlignmentRight
)
from Foundation import NSURL, NSObject
from WebKit import WKWebView, WKWebViewConfiguration


APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.getenv("APP_BACKEND", "http://127.0.0.1:8000")


def main_thread(func):
    """Decorator to ensure UI updates are on the main thread."""
    from PyObjCTools import AppHelper

    def wrapper(*args, **kwargs):
        AppHelper.callAfter(func, *args, **kwargs)
    return wrapper


class ChatClient:
    """Thin async client for server endpoints, with thread helpers."""
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self._loop = asyncio.new_event_loop()
        self._thr = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thr.start()
        self._client = httpx.AsyncClient(http2=True, timeout=None)

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _post_stream(self, path: str, json_body: Dict[str, Any]):
        url = f"{self.base}{path}"
        async with self._client.stream("POST", url, json=json_body) as resp:
            async for chunk in resp.aiter_lines():
                if not chunk:
                    continue
                if not chunk.startswith("data:"):
                    continue
                payload = chunk[5:].strip()
                if not payload:
                    continue
                try:
                    evt = json.loads(payload)
                    yield evt
                except Exception:
                    continue

    def chat_stream(self, messages: List[Dict[str, str]], settings: Dict[str, Any], system: str, on_event):
        async def _run_stream():
            body = {"messages": messages, "settings": settings, "system": system}
            try:
                async for evt in self._post_stream("/api/chat/stream", body):
                    on_event(evt)
            except Exception as e:
                on_event({"type": "error", "message": f"{type(e).__name__}: {e}"})

        self._run(_run_stream())

    def models(self, on_result):
        async def _get():
            try:
                r = await self._client.get(f"{self.base}/api/models", timeout=10.0)
                on_result(r.json())
            except Exception as e:
                on_result({"error": str(e)})
        self._run(_get())

    def set_model(self, tag: str, on_result):
        async def _post():
            try:
                r = await self._client.post(f"{self.base}/api/models/set", json={"model": tag})
                on_result(r.json())
            except Exception as e:
                on_result({"error": str(e)})
        self._run(_post())

    def health(self, on_result):
        async def _get():
            try:
                r = await self._client.get(f"{self.base}/api/health", timeout=5.0)
                on_result(r.json())
            except Exception as e:
                on_result({"ok": False, "error": str(e)})
        self._run(_get())


class NativeChatView(NSView):
    """Native chat UI: messages list, composer, telemetry, settings access."""
    def initWithClient_(self, client: ChatClient):
        self = super().init()
        if self is None:
            return None
        self.client = client
        self.history: List[Dict[str, str]] = []
        self.streaming = False
        self.settings: Dict[str, Any] = {
            "dynamic_ctx": True,
            "max_ctx": 40000,
            "num_ctx": 8192,
            "temperature": 0.9,
            "top_p": 0.9,
            "top_k": 100,
            "num_predict": "",
            "seed": "",
        }
        self.system = ""
        self.attachments: List[Dict[str, Any]] = []
        # telemetry
        self._started_at = 0.0
        self._first_byte = 0.0
        self._last_token_est = 0
        self._build()
        return self

    def _build(self):
        self.setTranslatesAutoresizingMaskIntoConstraints_(False)

        # Background blur
        self.bg = NSVisualEffectView.alloc().init()
        self.bg.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.bg.setBlendingMode_(0)
        self.bg.setMaterial_(0)
        self.addSubview_(self.bg)

        # Messages scroll + text view
        self.scroll = NSScrollView.alloc().init()
        self.scroll.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.scroll.setHasVerticalScroller_(True)
        self.text = NSTextView.alloc().init()
        self.text.setEditable_(False)
        self.text.setRichText_(True)
        self.text.setDrawsBackground_(False)
        self.scroll.setDocumentView_(self.text)
        self.addSubview_(self.scroll)

        # Composer controls
        self.prompt = NSTextView.alloc().init()
        self.prompt.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.prompt.setRichText_(False)
        self.prompt.setFont_(self.text.font())
        self.prompt.setString_("")

        self.sendBtn = NSButton.alloc().init()
        self.sendBtn.setTitle_("Send")
        self.sendBtn.setBezelStyle_(NSBezelStyleRounded)
        self.sendBtn.setTarget_(self)
        self.sendBtn.setAction_(b"send:")

        self.attachBtn = NSButton.alloc().init()
        self.attachBtn.setTitle_("Attach…")
        self.attachBtn.setBezelStyle_(NSBezelStyleRounded)
        self.attachBtn.setTarget_(self)
        self.attachBtn.setAction_(b"attach:")

        self.telemetry = NSTextField.alloc().init()
        self.telemetry.setEditable_(False)
        self.telemetry.setBezeled_(False)
        self.telemetry.setDrawsBackground_(False)
        self.telemetry.setAlignment_(NSTextAlignmentRight)
        self.telemetry.setStringValue_("tkn: —/— • — ms")

        self.spinner = NSProgressIndicator.alloc().init()
        self.spinner.setStyle_(1)  # spinning
        self.spinner.setDisplayedWhenStopped_(False)

        # Add subviews
        self.addSubview_(self.prompt)
        self.addSubview_(self.sendBtn)
        self.addSubview_(self.attachBtn)
        self.addSubview_(self.telemetry)
        self.addSubview_(self.spinner)

        # AutoLayout
        views = locals()
        for v in (self.bg, self.scroll, self.prompt, self.sendBtn, self.attachBtn, self.telemetry, self.spinner):
            v.setTranslatesAutoresizingMaskIntoConstraints_(False)

        self.addConstraints_([
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.bg, 0, 0, self, 0, 1.0, 0.0),
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.bg, 3, 0, self, 3, 1.0, 0.0),
        ])

        # Layout metrics
        pad = 12.0
        def pin(a, attr_a, b, attr_b, c=0.0):
            self.addConstraint_(
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    a, attr_a, 0, b, attr_b, 1.0, c)
            )

        # Pin scroll area
        pin(self.scroll, 1, self, 1, -120)  # bottom above composer
        pin(self.scroll, 2, self, 2, pad)   # left
        pin(self.scroll, 3, self, 3, -pad)  # right
        pin(self.scroll, 0, self, 0, pad)   # top

        # Prompt area
        pin(self.prompt, 2, self, 2, pad)
        pin(self.prompt, 3, self, 3, -160)
        pin(self.prompt, 1, self, 1, -60)
        pin(self.prompt, 0, self, 0, -100)  # height ~40px

        # Buttons
        pin(self.sendBtn, 3, self, 3, -pad)
        pin(self.sendBtn, 1, self, 1, -60)
        pin(self.attachBtn, 3, self, 3, -100)
        pin(self.attachBtn, 1, self, 1, -60)

        # Telemetry + spinner
        pin(self.telemetry, 3, self, 3, -pad)
        pin(self.telemetry, 1, self, 1, -pad)
        pin(self.spinner, 3, self, 3, -pad)
        pin(self.spinner, 1, self, 1, -90)

        self._append_system("Welcome. Ask anything…")

    def _append(self, role: str, content: str):
        from AppKit import NSAttributedString
        prefix = "You: " if role == "user" else ("Assistant: " if role == "assistant" else "System: ")
        body = content.strip() + "\n\n"
        text = f"{prefix}{body}"
        attr = NSAttributedString.alloc().initWithString_(text)
        self.text.textStorage().appendAttributedString_(attr)
        self.text.scrollToEndOfDocument_(None)

    def _append_system(self, content: str):
        self._append("system", content)

    def _append_user(self, content: str):
        self._append("user", content)

    def _append_assistant_delta(self, delta: str):
        if not delta:
            return
        from AppKit import NSAttributedString
        attr = NSAttributedString.alloc().initWithString_(delta)
        self.text.textStorage().appendAttributedString_(attr)
        self.text.scrollToEndOfDocument_(None)

        # token estimate & update telemetry
        self._last_token_est = max(self._last_token_est, int(len(delta) / 4))
        self._update_telemetry()

    def _append_assistant_done(self):
        from AppKit import NSAttributedString
        end = NSAttributedString.alloc().initWithString_("\n\n")
        self.text.textStorage().appendAttributedString_(end)
        self.text.scrollToEndOfDocument_(None)

    def _build_user_payload(self, prompt: str) -> str:
        user = prompt
        for att in self.attachments:
            if att.get("type") == "text" and att.get("text"):
                user += f"\n\n**File: {att.get('name','file')}**\n```\n{att['text']}\n```\n"
            elif att.get("type") == "image" and att.get("url"):
                user += f"\n\n![{att.get('name','image')}]({att['url']})\n"
            else:
                user += f"\n\n(Attached file: {att.get('name','file')})\n"
        return user

    def send_(self, _):
        if self.streaming:
            # No cancel in this simple version
            return
        prompt = str(self.prompt.string()).strip()
        if not prompt and not self.attachments:
            return
        user_text = self._build_user_payload(prompt)
        self.history.append({"role": "user", "content": user_text})
        self._append_user(user_text)
        self.prompt.setString_("")
        self.streaming = True
        self.spinner.startAnimation_(None)
        # telemetry reset
        import time
        self._started_at = time.perf_counter()
        self._first_byte = 0.0
        self._last_token_est = 0
        self._update_telemetry(in_tokens=int(len(user_text)/4))
        # prepare assistant buffer for this turn
        self._assistant_buf = ""
        self._start_stream()

    def attach_(self, _):
        panel = NSOpenPanel.openPanel()
        panel.setAllowsMultipleSelection_(True)
        if panel.runModal():
            for url in panel.URLs():
                path = url.path()
                try:
                    name = os.path.basename(path)
                    with open(path, 'rb') as f:
                        data = f.read()
                    # Heuristic: treat small files as text
                    try:
                        text = data.decode('utf-8')
                        self.attachments.append({"type": "text", "name": name, "text": text})
                    except Exception:
                        self.attachments.append({"type": "binary", "name": name})
                except Exception as e:
                    self._append_system(f"Failed to attach: {e}")

    def _on_event(self, evt: Dict[str, Any]):
        typ = evt.get("type")
        if typ == "delta":
            delta = evt.get("delta", "")
            if self._first_byte == 0.0:
                import time
                self._first_byte = time.perf_counter()
                self._update_telemetry()
            # Stream and collect assistant output
            self._append_assistant_delta(delta)
            try:
                self._assistant_buf += delta
            except Exception:
                self._assistant_buf = delta
        elif typ == "tool_calls":
            # Optionally reflect ongoing tool usage
            self._append_system("[tools] running…")
        elif typ == "tool_result":
            name = evt.get("name")
            self._append_system(f"[tool:{name}] done")
        elif typ == "done":
            self._append_assistant_done()
            self.streaming = False
            self.spinner.stopAnimation_(None)
            # record assistant reply in history
            if getattr(self, "_assistant_buf", ""):
                self.history.append({"role": "assistant", "content": self._assistant_buf})
        elif typ == "error":
            self._append_system(f"Error: {evt.get('message')}")
            self.streaming = False
            self.spinner.stopAnimation_(None)

    def _start_stream(self):
        settings = dict(self.settings)
        system = self.system
        self.client.chat_stream(self.history, settings, system, self._on_event)

    def _update_telemetry(self, in_tokens: int | None = None):
        import time
        latency_ms = "—"
        if self._first_byte:
            latency_ms = f"{int(1000*(self._first_byte - self._started_at))} ms"
        out = self._last_token_est or 0
        if in_tokens is None:
            # estimate from last user msg if any
            in_tokens = 0
            if self.history:
                in_tokens = int(len(self.history[-1].get('content',''))/4)
        self.telemetry.setStringValue_(f"tkn: {in_tokens}/{out} • {latency_ms}")


class SettingsPanel(NSWindow):
    """Simple settings window with model management and health."""
    def initWithClient_native_(self, client: ChatClient, native: NativeChatView):
        self = super().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 520, 560), NSWindowStyleMask.titled | NSWindowStyleMask.closable, NSBackingStoreBuffered, False
        )
        if self is None:
            return None
        self.setTitle_("Settings")
        self.client = client
        self.native = native

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 520, 560))
        root.setTranslatesAutoresizingMaskIntoConstraints_(False)
        self.setContentView_(root)

        # Fields
        def label(text):
            l = NSTextField.alloc().init()
            l.setEditable_(False); l.setBezeled_(False); l.setDrawsBackground_(False)
            l.setStringValue_(text)
            return l

        self.modelField = NSTextField.alloc().init()
        self.modelField.setStringValue_("")
        self.useBtn = NSButton.alloc().init(); self.useBtn.setTitle_("Use Model"); self.useBtn.setTarget_(self); self.useBtn.setAction_(b"applyModel:")
        self.refreshBtn = NSButton.alloc().init(); self.refreshBtn.setTitle_("Installed…"); self.refreshBtn.setTarget_(self); self.refreshBtn.setAction_(b"refreshModels:")
        self.modelsList = NSTextView.alloc().init(); self.modelsList.setEditable_(False)

        self.systemField = NSTextView.alloc().init(); self.systemField.setEditable_(True)

        # Basic params
        def number_field(val):
            f = NSTextField.alloc().init(); f.setStringValue_(str(val)); return f
        self.dynamicField = NSButton.alloc().init(); self.dynamicField.setTitle_("Dynamic Context"); self.dynamicField.setButtonType_(3); self.dynamicField.setState_(1)
        self.maxCtxField = number_field(40000)
        self.numCtxField = number_field(8192)
        self.tempField = number_field(0.9)
        self.topPField = number_field(0.9)
        self.topKField = number_field(100)
        self.numPredictField = NSTextField.alloc().init(); self.numPredictField.setStringValue_("")
        self.seedField = NSTextField.alloc().init(); self.seedField.setStringValue_("")
        self.applySettingsBtn = NSButton.alloc().init(); self.applySettingsBtn.setTitle_("Apply Settings"); self.applySettingsBtn.setTarget_(self); self.applySettingsBtn.setAction_(b"applySettings:")

        self.healthField = NSTextField.alloc().init(); self.healthField.setEditable_(False); self.healthField.setBezeled_(False); self.healthField.setDrawsBackground_(False); self.healthField.setStringValue_("Checking backend…")

        # Layout (simple vertical stack)
        y = 520
        def add(view, h=24, pad=8):
            nonlocal y
            y -= (h + pad)
            view.setFrame_(NSMakeRect(16, y, 488, h))
            root.addSubview_(view)

        add(label("Model tag:"))
        add(self.modelField)
        row = NSView.alloc().initWithFrame_(NSMakeRect(16, y-26, 488, 26)); root.addSubview_(row); y -= (26 + 8)
        self.useBtn.setFrame_(NSMakeRect(0, 0, 120, 26)); row.addSubview_(self.useBtn)
        self.refreshBtn.setFrame_(NSMakeRect(128, 0, 120, 26)); row.addSubview_(self.refreshBtn)
        add(label("Installed models:"))
        self.modelsList.enclosingScrollView() or None
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, y-120, 488, 120)); y -= (120 + 8)
        scroll.setHasVerticalScroller_(True); scroll.setDocumentView_(self.modelsList); root.addSubview_(scroll)

        add(label("System prompt:"), h=18)
        spScroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(16, y-100, 488, 100)); y -= (100 + 8)
        spScroll.setHasVerticalScroller_(True); spScroll.setDocumentView_(self.systemField); root.addSubview_(spScroll)

        add(self.dynamicField)
        add(label("Max Context")); add(self.maxCtxField)
        add(label("Static Context (num_ctx)")); add(self.numCtxField)
        add(label("Temperature")); add(self.tempField)
        add(label("top_p")); add(self.topPField)
        add(label("top_k")); add(self.topKField)
        add(label("num_predict (optional)")); add(self.numPredictField)
        add(label("seed (optional)")); add(self.seedField)
        add(self.applySettingsBtn)
        add(self.healthField)

        # Prime with native values
        self._sync_from_native()
        self._check_health()

        return self

    def _sync_from_native(self):
        s = self.native.settings
        self.dynamicField.setState_(1 if s.get('dynamic_ctx') else 0)
        self.maxCtxField.setStringValue_(str(s.get('max_ctx', 40000)))
        self.numCtxField.setStringValue_(str(s.get('num_ctx', 8192)))
        self.tempField.setStringValue_(str(s.get('temperature', 0.9)))
        self.topPField.setStringValue_(str(s.get('top_p', 0.9)))
        self.topKField.setStringValue_(str(s.get('top_k', 100)))
        self.numPredictField.setStringValue_(str(s.get('num_predict', '')))
        self.seedField.setStringValue_(str(s.get('seed', '')))
        self.systemField.setString_(self.native.system or "")

    def applySettings_(self, _):
        self.native.settings = {
            "dynamic_ctx": bool(self.dynamicField.state()),
            "max_ctx": int(self.maxCtxField.stringValue() or '40000'),
            "num_ctx": int(self.numCtxField.stringValue() or '8192'),
            "temperature": float(self.tempField.stringValue() or '0.9'),
            "top_p": float(self.topPField.stringValue() or '0.9'),
            "top_k": int(self.topKField.stringValue() or '100'),
            "num_predict": (self.numPredictField.stringValue() or '').strip(),
            "seed": (self.seedField.stringValue() or '').strip(),
        }
        self.native.system = str(self.systemField.string() or "")

    def refreshModels_(self, _):
        def on_result(obj):
            txt = json.dumps(obj, indent=2)
            self.modelsList.setString_(txt)
        self.client.models(on_result)

    def applyModel_(self, _):
        tag = self.modelField.stringValue()
        if not tag:
            return
        def on_result(obj):
            NSAlert.alertWithMessageText_defaultButton_alternateButton_otherButton_informativeTextWithFormat_(
                "Model", "OK", None, None, json.dumps(obj)
            ).runModal()
        self.client.set_model(tag, on_result)

    def _check_health(self):
        def on_health(h):
            if h.get('ok'):
                self.healthField.setStringValue_(f"Backend OK • model: {h.get('model')}")
            else:
                self.healthField.setStringValue_(f"Backend DOWN: {h.get('error','unknown')}")
        self.client.health(on_health)


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.client = ChatClient(BACKEND)

        style = (
            NSWindowStyleMask.titled | NSWindowStyleMask.closable | NSWindowStyleMask.resizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200.0, 200.0, 900.0, 640.0), style, NSBackingStoreBuffered, False
        )
        self.window.setTitle_("Chat (macOS)")

        # Mode selector
        self.mode = 0  # 0=native, 1=web

        self.native = NativeChatView.alloc().initWithClient_(self.client)
        self.web = WKWebView.alloc().initWithFrame_configuration_(NSMakeRect(0, 0, 900, 640), WKWebViewConfiguration.alloc().init())
        self.web.setHidden_(True)
        # Load local index.html for parity
        local = os.path.join(APP_DIR, 'index.html')
        if os.path.exists(local):
            self.web.loadFileURL_allowingReadAccessToURL_(NSURL.fileURLWithPath_(local), NSURL.fileURLWithPath_(APP_DIR))

        root = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 900, 640))
        root.addSubview_(self.native)
        root.addSubview_(self.web)
        for v in (self.native, self.web):
            v.setTranslatesAutoresizingMaskIntoConstraints_(False)
            # Pin to edges
            for (attr_a, attr_b) in ((0,0),(1,1),(2,2),(3,3)):
                root.addConstraint_(
                    NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                        v, attr_a, 0, root, attr_b, 1.0, 0.0)
                )

        self.window.setContentView_(root)

        # Toolbar-ish controls
        self.segment = NSSegmentedControl.segmentedControlWithLabels_trackingMode_target_action_([
            "Native", "Web (Parity)"
        ], 1, self, b"toggleMode:")
        self.segment.setSelected_forSegment_(True, 0)
        self.window.setTitleVisibility_(1)
        self.window.setTitlebarAppearsTransparent_(True)
        self.window.standardWindowButton_(0).setHidden_(False)
        # Place segmented control + buttons
        root.addSubview_(self.segment)
        self.segment.setTranslatesAutoresizingMaskIntoConstraints_(False)
        root.addConstraint_(
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.segment, 0, 0, root, 0, 1.0, 10.0))
        root.addConstraint_(
            NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                self.segment, 2, 0, root, 2, 1.0, 10.0))

        # Buttons: New, Export, Settings
        self.newBtn = NSButton.alloc().init(); self.newBtn.setTitle_("New Chat"); self.newBtn.setTarget_(self); self.newBtn.setAction_(b"newChat:")
        self.exportBtn = NSButton.alloc().init(); self.exportBtn.setTitle_("Export…"); self.exportBtn.setTarget_(self); self.exportBtn.setAction_(b"exportChat:")
        self.settingsBtn = NSButton.alloc().init(); self.settingsBtn.setTitle_("Settings…"); self.settingsBtn.setTarget_(self); self.settingsBtn.setAction_(b"openSettings:")
        for b, off in ((self.newBtn, 120), (self.exportBtn, 220), (self.settingsBtn, 320)):
            root.addSubview_(b)
            b.setTranslatesAutoresizingMaskIntoConstraints_(False)
            root.addConstraint_(
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    b, 0, 0, root, 0, 1.0, 10.0))
            root.addConstraint_(
                NSLayoutConstraint.constraintWithItem_attribute_relatedBy_toItem_attribute_multiplier_constant_(
                    b, 2, 0, root, 2, 1.0, off))

        self.window.makeKeyAndOrderFront_(None)

    def toggleMode_(self, sender):
        idx = sender.selectedSegment()
        self.mode = idx
        if idx == 0:
            self.native.setHidden_(False)
            self.web.setHidden_(True)
        else:
            self.native.setHidden_(True)
            self.web.setHidden_(False)

    def newChat_(self, _):
        self.native.history = []
        self.native.text.setString_("")
        self.native._append_system("New chat.")

    def exportChat_(self, _):
        panel = NSSavePanel.savePanel()
        panel.setAllowedFileTypes_(["md", "txt"])
        panel.setNameFieldStringValue_("chat.md")
        if panel.runModal():
            path = panel.URL().path()
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    for m in self.native.history:
                        f.write(f"## {m['role']}\n\n{m['content']}\n\n")
                NSAlert.alertWithMessageText_defaultButton_alternateButton_otherButton_informativeTextWithFormat_(
                    "Export", "OK", None, None, f"Saved to {path}"
                ).runModal()
            except Exception as e:
                NSAlert.alertWithMessageText_defaultButton_alternateButton_otherButton_informativeTextWithFormat_(
                    "Export", "OK", None, None, f"Failed: {e}"
                ).runModal()

    def openSettings_(self, _):
        if not hasattr(self, 'settingsWin') or self.settingsWin is None:
            self.settingsWin = SettingsPanel.alloc().initWithClient_native_(self.client, self.native)
        self.settingsWin.center()
        self.settingsWin.makeKeyAndOrderFront_(None)


if __name__ == '__main__':
    app = NSApplication.sharedApplication()
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    import PyObjCTools.AppHelper as AppHelper
    AppHelper.runEventLoop()
