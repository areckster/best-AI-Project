#!/usr/bin/env python3
import asyncio
import json
import os
import sys
from typing import Any, Dict

import httpx


API_URL = os.getenv("CHAT_API", "http://127.0.0.1:8000/api/chat/stream")


async def stream_chat(prompt: str, system: str = "", developer: str = "", settings: Dict[str, Any] | None = None) -> None:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "system": system,
        "developer": developer,
        "settings": settings or {},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", API_URL, json=payload) as resp:
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code}: {await resp.aread()}")
                return
            buffer = b""
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                buffer += chunk
                while b"\n\n" in buffer:
                    raw, buffer = buffer.split(b"\n\n", 1)
                    # Each SSE record comes as lines; we only care about `data:` ones
                    for line in raw.split(b"\n"):
                        if not line.startswith(b"data: "):
                            continue
                        data = line[len(b"data: "):]
                        try:
                            evt = json.loads(data)
                        except Exception:
                            continue
                        etype = evt.get("type")
                        if etype == "delta":
                            sys.stdout.write(evt.get("delta", ""))
                            sys.stdout.flush()
                        elif etype == "tool_calls":
                            print("\n[tool_calls]", json.dumps(evt.get("tool_calls", []), indent=2))
                        elif etype == "tool_result":
                            print("\n[tool_result]", evt.get("name"), json.dumps(evt.get("output"), indent=2)[:1200])
                        elif etype == "gate_warning":
                            print("\n[gate_warning]", evt.get("message"))
                        elif etype == "error":
                            print("\n[error]", evt.get("message"))
                        elif etype == "done":
                            usage = evt.get("usage")
                            if usage:
                                print(f"\n\n[done] usage={usage}")
                        else:
                            # Unknown events for debugging
                            print("\n[event]", evt)


def main():
    if len(sys.argv) < 2:
        print("Usage: scripts/cli_chat.py 'your prompt here' [settings-json]")
        print("Example: scripts/cli_chat.py 'Expected price iPhone 17 Air' '{\"mock_model\": true}'")
        return
    prompt = sys.argv[1]
    settings: Dict[str, Any] | None = None
    if len(sys.argv) >= 3:
        try:
            settings = json.loads(sys.argv[2])
        except Exception as e:
            print(f"Invalid settings JSON: {e}")
            return
    asyncio.run(stream_chat(prompt, settings=settings))


if __name__ == "__main__":
    main()
