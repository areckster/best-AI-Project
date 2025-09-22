# server.py
import os
import math
import base64
import asyncio
import datetime as _dt
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
import hashlib
import mimetypes
import pathlib

import httpx
import orjson
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi import Body

from tools import (
    web_search,
    open_url,
    eval_expr,
    execute,
    read_file,
    write_file,
    terminal_open,
    terminal_run,
    terminal_terminate,
    notes_write,
    notes_list,
    notes_read,
    user_prefs_write,
    user_prefs_list,
    user_prefs_read,
)
from docstore import init_store, get_store

APP_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(APP_DIR, "tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# Configure via env if you want
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("MODEL", "qwen3:4b-thinking-2507-q4_K_M") #previously huihui_ai/qwen3-abliterated:4b
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "8192"))
USER_MAX_CTX = int(os.getenv("USER_MAX_CTX", "40000"))
REASONING_OFF_MODEL = os.getenv("REASONING_OFF_MODEL", "gemma3:4b-it-qat")
REASON_SUMMARY_MODEL = os.getenv("REASON_SUMMARY_MODEL", "llama3.2:1b")
REASON_SUMMARY_THINK_CTX = int(os.getenv("REASON_SUMMARY_THINK_CTX", "1024"))
CURRENT_DATE_HUMAN = _dt.datetime.now(_dt.timezone.utc).astimezone().strftime("%B %d, %Y")
OFFICIAL_HOSTS = {
    h.strip().lower()
    for h in (os.getenv("OFFICIAL_HOSTS") or "apple.com, support.apple.com, developer.apple.com, docs.oracle.com, docs.microsoft.com, support.google.com, help.twitter.com, help.instagram.com, aws.amazon.com, docs.aws.amazon.com")
    if h.strip()
}


def _normalize_model_tag(tag: str) -> str:
    t = (tag or "").strip().lower()
    if not t:
        return ""
    if "/" in t:
        t = t.split("/")[-1]
    return t


def _is_qwen3_or_gemma4b(tag: str) -> bool:
    """Heuristic match for qwen3 and gemma 4B variants.

    We normalize the tag and look for:
      - any qwen3* model
      - gemma3:4b* or a literal 'gemma4b' token
    """
    t = _normalize_model_tag(tag)
    if not t:
        return False
    if "qwen3" in t:
        return True
    # Accept either explicit 'gemma4b' or gemma3 4b family strings
    if "gemma4b" in t:
        return True
    if "gemma3" in t and ":4b" in t:
        return True
    return False


_default_non_tool_models = {
    _normalize_model_tag(REASONING_OFF_MODEL),
    "gemma3:4b",
    "gemma3:4b-it-qat",
}
_env_non_tool = {
    _normalize_model_tag(part)
    for part in os.getenv("NON_TOOL_MODELS", "").split(",")
    if part.strip()
}
NON_TOOL_MODELS = {m for m in (_default_non_tool_models | _env_non_tool) if m}


def model_supports_tools(tag: str) -> bool:
    return _normalize_model_tag(tag) not in NON_TOOL_MODELS

#supercedes system prompt (somehow)
DEFAULT_DEVELOPER_PROMPT = (
    "You are a pragmatic, tool-using assistant focused on accurate, actionable results.\n\n"
    "MISSION: Deliver a complete, satisfying answer. If a tool helps, use it without hesitation.\n\n"
    f"TIME: Treat today as {CURRENT_DATE_HUMAN}. Make it clear when information is unverified or projected beyond that date.\n\n"
    "MAY: • Launch tools proactively; expand scope; reframe problems; improvise.\n"
    "MUST NOT: • Stall; defer to weak wording; hold back out of politeness.\n\n"
    "TOOLKIT: web_search; open_url; search_docs; summarize_file; read_whole_file; assistant; eval_expr/execute; terminal_*; read_file/write_file; notes_*/user_prefs_*.\n\n"
    "DOCS USAGE:\n"
    "  • Prefer search_docs for targeted facts and citations (filters.source='file' when using user uploads).\n"
    "  • Use summarize_file(path) to get a quick overview of a specific file when you need orientation.\n"
    "  • Use read_whole_file(path) only when exact wording is required; it is expensive in context.\n"
    "  • Use assistant(instruction, path|doc_id) to delegate large-file tasks (e.g., extract questions), it processes files in chunks and returns focused results.\n\n"
    "SOURCE-FIRST & VERIFICATION:\n"
    "  • After any web_search, call open_url on the best 1–3 results before answering.\n"
    "  • Any numeric claim should be backed by an OPENED source when possible.\n"
    "  • Cite only pages opened in this session; name domain + title/section + date.\n\n"
    "FAIL→DIAGNOSE→HEAL (ALL TOOLS):\n"
    "  • FAILURE if: ok=false, top-level error, nonzero returncode, missing fields, or parse fails.\n"
    "  • On FAILURE (no prose between calls): classify, sanity-check, re-init state, normalize args, retry ≤2 with sensible variation, or pivot tools.\n\n"
    "STRICT PROTOCOL: Parse tool JSON and act. If terminal_run fails due to closed session, terminal_open then re-run. Treat stderr with returncode==0 as a warning.\n\n"
    "OUTPUT STYLE: Write a thorough, well-structured answer with multi-paragraph explanations, focused bullets, and concrete examples when useful. Avoid revealing chain-of-thought.\n"
    "  • Prefer clarity and depth over brevity; add short, labeled sections if it helps.\n\n"
    "FUTURE/UNVERIFIED CONTENT:\n"
    "  • For product rumors, unreleased specs, release dates, or market figures: DO NOT speculate.\n"
    "  • Only state facts after opening an official or authoritative source via open_url.\n"
    "  • If you cannot open an official source that confirms the claim, answer: 'No verified details available as of today.' and proceed with general context without specifics.\n\n"
    "HARD BANS: No invented numbers/timelines. Re-verify contentious claims before citing.\n\n"
    "MANDATES: After web_search, open results before answering. If stuck, try something else (new search, different tool, code, terminal).\n\n"
    "FACTUAL CLAIM GATE: Before stating dates/specs/prices/titles/quotations, ensure at least one successful open_url to an authoritative source in this session about the claim. Otherwise, say you cannot verify.\n\n"
    "REMEMBER: Be decisive, source-grounded, and tool‑wielding.\n"
)


# Tool definitions: encourage follow-up opens after search
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for recent or factual info and return top results. "
                "After identifying relevant URLs, the assistant should invoke open_url to fetch and summarize page contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": (
                "Open a URL and return a concise text extract for summarization. "
                "Pass a smaller max_chars (e.g., 1200) for a quick scan; increase only when needed. "
                "Typically used after web_search identifies relevant links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch"},
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters of extracted text (start small; increase only if needed)",
                        "default": 6000,
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the local docstore using hybrid semantic + keyword retrieval. Returns relevant chunks with titles, sources, and previews."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Max results", "default": 6},
                    "rerank": {"type": "boolean", "description": "Re-rank top results with small model", "default": False},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "tag": {"type": "string"},
                            "doc_id": {"type": "string"}
                        },
                        "description": "Optional filters"
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_related_links",
            "description": (
                "From a given page, find top related links and return short previews ranked by relevance to the current query. "
                "Use when the opened page is insufficient and you need adjacent sources."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The page URL to explore from"},
                    "query": {"type": "string", "description": "What you are looking for (to score links)"},
                    "k": {"type": "integer", "description": "Number of related links to preview", "default": 5}
                },
                "required": ["url", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eval_expr",
            "description": "Evaluate a Python expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "Expression to evaluate"}
                },
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute",
            "description": "Run a Python code snippet and capture stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to run"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_file",
            "description": (
                "Summarize a local uploaded file by filename/path. "
                "Use for quick orientation when you need an overview; prefer search_docs for targeted facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path under server tmp/ to summarize"},
                    "max_chars": {"type": "integer", "description": "Max input characters from file to summarize", "default": 12000}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_whole_file",
            "description": (
                "Read and return the full extracted text of a local uploaded file by filename/path. "
                "Use when you truly need exact wording; large files may be truncated for transport."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path under server tmp/ to read"},
                    "max_chars": {"type": "integer", "description": "Optional soft cap for response size", "default": 100000}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assistant",
            "description": (
                "Delegate a large-text task to a helper that can process long files in chunks. "
                "Use this to extract, summarize, or transform content too big for your context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string", "description": "What the assistant should do (e.g., extract all questions)"},
                    "path": {"type": "string", "description": "Optional absolute path under tmp/ for an uploaded file"},
                    "doc_id": {"type": "string", "description": "Optional doc_id previously ingested in the docstore"},
                    "max_chars": {"type": "integer", "description": "Soft cap of characters to consume", "default": 200000}
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "contents": {"type": "string", "description": "Text to write"}
                },
                "required": ["path", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_open",
            "description": "Open a terminal session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_run",
            "description": "Run a shell command in the terminal session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run"}
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_terminate",
            "description": "Terminate the terminal session.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notes_write",
            "description": "Store a note in memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Note identifier"},
                    "content": {"type": "string", "description": "Note content"}
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notes_list",
            "description": "List all stored note keys.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notes_read",
            "description": "Read a note by key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Note identifier"}
                },
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "user_prefs_write",
            "description": "Store a user preference value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Preference key"},
                    "content": {"type": "string", "description": "Preference value"}
                },
                "required": ["key", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "user_prefs_list",
            "description": "List stored user preference keys.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "user_prefs_read",
            "description": "Read a stored user preference value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Preference key"}
                },
                "required": ["key"],
            },
        },
    },
]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
    app.state.client = httpx.AsyncClient(http2=True, limits=limits)
    # Initialize docstore (SQLite + FAISS) at startup
    init_store()
    app.state.sum_lock = asyncio.Lock()
    yield
    # Shutdown
    await app.state.client.aclose()

app = FastAPI(title="Ollama Chat UI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- File ingestion helpers --------
def _sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()

def _safe_name(name: str) -> str:
    name = os.path.basename(name)
    name = name.replace("..", "_")
    return name

def _guess_ext(filename: str, content_type: Optional[str]) -> str:
    if filename and "." in filename:
        return "." + filename.rsplit(".", 1)[-1].lower()
    if content_type:
        return mimetypes.guess_extension(content_type) or ""
    return ""

def _pdf_text(path: str) -> str:
    """Extract text from a PDF file.

    Priority order:
    1) pypdf (pure-Python wheels available on Py3.13)
    2) PyMuPDF/fitz (if installed)
    """
    # Try pypdf first (no native build required)
    try:
        from pypdf import PdfReader  # type: ignore

        try:
            reader = PdfReader(path)
            parts = []
            for page in reader.pages:
                try:
                    parts.append(page.extract_text() or "")
                except Exception:
                    continue
            txt = "\n".join([p.strip() for p in parts if p and p.strip()])
            if txt:
                return txt
        except Exception:
            pass
    except Exception:
        pass

    # Fallback to PyMuPDF if present
    try:
        import fitz  # type: ignore

        try:
            doc = fitz.open(path)
            parts = []
            for page in doc:
                try:
                    parts.append(page.get_text("text") or "")
                except Exception:
                    continue
            return "\n".join([p.strip() for p in parts if p and p.strip()])
        except Exception:
            return ""
    except Exception:
        return ""

def _ocr_pdf(path: str) -> str:
    # Best-effort OCR using pdf2image + pytesseract
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return ""
    try:
        pages = convert_from_path(path)
        out = []
        for img in pages[:100]:  # cap to prevent extreme OCR times
            try:
                out.append(pytesseract.image_to_string(img) or "")
            except Exception:
                continue
        return "\n".join([x.strip() for x in out if x and x.strip()])
    except Exception:
        return ""

def _ocr_image(path: str) -> str:
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(path)
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""

def _docx_text(path: str) -> str:
    """Extract text from a .docx file using docx2txt (pure-Python)."""
    try:
        import docx2txt  # type: ignore
    except Exception:
        return ""
    try:
        txt = docx2txt.process(path) or ""
        return txt.strip()
    except Exception:
        return ""

def _html_text(path: str) -> str:
    """Extract readable text from a local HTML file with BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        return ""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        soup = BeautifulSoup(raw, "html.parser")
        # Remove scripts/styles/navs for cleaner text
        for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
            try:
                tag.decompose()
            except Exception:
                pass
        return (soup.get_text(" ") or "").strip()
    except Exception:
        return ""

def _rtf_text(path: str) -> str:
    """Extract text from RTF using striprtf (best-effort)."""
    try:
        from striprtf.striprtf import rtf_to_text  # type: ignore
    except Exception:
        return ""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        return (rtf_to_text(raw.decode("utf-8", "replace")) or "").strip()
    except Exception:
        return ""

def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars/token
    return math.ceil(len(text) / 4) if text else 0


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def build_options(settings: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    temperature = float(settings.get("temperature", 0.9))
    top_p = float(settings.get("top_p", 0.9))
    top_k = int(settings.get("top_k", 100))
    seed = settings.get("seed") or None
    num_predict = settings.get("num_predict") or None

    dynamic_ctx = bool(settings.get("dynamic_ctx", True))
    user_max_ctx = int(settings.get("max_ctx", USER_MAX_CTX))
    static_ctx = int(settings.get("num_ctx", DEFAULT_NUM_CTX))
    num_thread = settings.get("num_thread") or None
    num_batch = settings.get("num_batch") or None
    num_gpu = settings.get("num_gpu") or None

    joined = "".join(f"{m.get('role','')}: {m.get('content','')}\n" for m in messages)
    est_tokens = estimate_tokens(joined)

    if dynamic_ctx:
        num_ctx = clamp(int(est_tokens * 1.2), 4096, user_max_ctx)
    else:
        num_ctx = clamp(static_ctx, 2048, user_max_ctx)

    opts: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "num_ctx": num_ctx,
    }
    if seed not in (None, ""):
        try:
            opts["seed"] = int(seed)
        except Exception:
            pass
    if num_predict not in (None, ""):
        try:
            opts["num_predict"] = int(num_predict)
        except Exception:
            pass
    if num_thread not in (None, ""):
        try:
            opts["num_thread"] = int(num_thread)
        except Exception:
            pass
    if num_batch not in (None, ""):
        try:
            opts["num_batch"] = int(num_batch)
        except Exception:
            pass
    if num_gpu not in (None, ""):
        try:
            opts["num_gpu"] = int(num_gpu)
        except Exception:
            pass

    return opts


# Lightweight LLM-based reranker for doc results
async def _rerank_with_llm(client: httpx.AsyncClient, query: str, items: List[Dict[str, Any]]) -> List[int]:
    try:
        if not items:
            return []
        top = items[: min(16, len(items))]
        def to_entry(it: Dict[str, Any]) -> str:
            iid = it.get("id")
            title = (it.get("title") or it.get("doc_id") or "").strip()
            host = (it.get("host") or it.get("source") or "").strip()
            pv = (it.get("preview") or "").strip()
            return f"- id:{iid} | {host} | {title}\n{pv[:260]}"
        listing = "\n".join(to_entry(it) for it in top)
        sys = (
            "You are a reranker. Rank items by their usefulness for answering the query.\n"
            "Respond strictly as JSON: {\"order\":[<ids in best→worst>]}"
        )
        user = f"Query: {query}\n\nItems:\n{listing}\n\nReturn only JSON with an array of ids in descending relevance."
        req = {
            "model": REASON_SUMMARY_MODEL,
            "prompt": sys + "\n\n" + user,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "top_p": 0.8, "top_k": 30},
        }
        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=45.0)
        if r.status_code >= 400:
            return [int(it.get("id")) for it in top]
        try:
            data = r.json()
        except Exception:
            data = {"response": r.text}
        raw = (data.get("response") or "").strip()
        import orjson as _oj
        try:
            obj = _oj.loads(raw)
        except Exception:
            # try to extract braces
            i = raw.find("{"); j = raw.rfind("}")
            if i != -1 and j != -1 and j > i:
                try:
                    obj = _oj.loads(raw[i:j+1])
                except Exception:
                    obj = {}
            else:
                obj = {}
        order = obj.get("order") if isinstance(obj, dict) else None
        out: List[int] = []
        if isinstance(order, list):
            for x in order:
                try:
                    out.append(int(x))
                except Exception:
                    continue
        if not out:
            out = [int(it.get("id")) for it in top]
        return out
    except Exception:
        return [int(it.get("id")) for it in items[: min(12, len(items))]]


@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(APP_DIR, "index.html"))


@app.get("/api/health")
async def health():
    try:
        r = await app.state.client.get(f"{OLLAMA_HOST}/api/tags", timeout=3.0)
        ok = (r.status_code == 200)
    except Exception:
        ok = False
    return {"ok": ok, "ollama": OLLAMA_HOST, "model": MODEL}


@app.post("/api/chat/stream")
async def chat_stream(payload: Dict[str, Any]):
    messages = payload.get("messages", [])
    settings = payload.get("settings", {})
    system_prompt = payload.get("system", "")
    developer_prompt = payload.get("developer") or DEFAULT_DEVELOPER_PROMPT

    combined_prompt_parts: List[str] = []
    if system_prompt:
        combined_prompt_parts.append(system_prompt.strip())
    if developer_prompt:
        combined_prompt_parts.append(
            "Developer Instructions (these take priority over system prompts):\n" + developer_prompt.strip()
        )

    combined_prompt = "\n\n".join(combined_prompt_parts)

    initial_msgs: List[Dict[str, Any]] = []
    if combined_prompt:
        initial_msgs.append({"role": "system", "content": combined_prompt})

    messages = initial_msgs + messages

    async def event_gen():
        DATA = b"data: "
        END = b"\n\n"
        convo = list(messages)
        client = app.state.client
        active_model = MODEL
        requested_tools = bool(payload.get("tools", True))
        supports_tools = model_supports_tools(active_model)
        have_tools = requested_tools and supports_tools
        if requested_tools and not supports_tools:
            convo.append({
                "role": "system",
                "content": (
                    "Current model does not support tool calling. Respond directly using the conversation and any provided context."
                ),
            })

        # Pre-tool: if the user asked to answer questions from an uploaded file,
        # proactively extract the questions with the assistant tool so the model
        # can work from a clean list. This also surfaces in the Thinking panel.
        ran_pre_assistant = False

        def _last_user_text(msgs: List[Dict[str, Any]]) -> str:
            for m in reversed(msgs):
                if m.get("role") == "user":
                    try:
                        return str(m.get("content") or "")
                    except Exception:
                        return ""
            return ""

        def _extract_file_paths(msgs: List[Dict[str, Any]]) -> List[str]:
            # Find a system note that ends with: Files: [ {title, path}, ... ]
            import re as _re
            paths: List[str] = []
            for m in reversed(msgs):
                if m.get("role") != "system":
                    continue
                content = m.get("content") or ""
                try:
                    s = str(content)
                except Exception:
                    continue
                if "Files:" not in s:
                    continue
                try:
                    # Capture the JSON array after 'Files:'
                    # Greedy to last closing bracket
                    mobj = _re.search(r"Files:\s*(\[.*\])\s*$", s, flags=_re.DOTALL)
                    blob = None
                    if mobj:
                        blob = mobj.group(1)
                    else:
                        # Fallback: find first '[' after 'Files:' and the last ']'
                        idx = s.find("Files:")
                        if idx != -1:
                            jstart = s.find('[', idx)
                            jend = s.rfind(']')
                            if jstart != -1 and jend != -1 and jend > jstart:
                                blob = s[jstart:jend+1]
                    if not blob:
                        continue
                    arr = orjson.loads(blob)
                    if isinstance(arr, list):
                        for it in arr:
                            try:
                                p = str((it or {}).get("path") or "")
                                if p:
                                    paths.append(p)
                            except Exception:
                                continue
                    if paths:
                        return paths
                except Exception:
                    continue
            return paths

        # Heuristic: look for "questions" intent in the last user message
        user_text = _last_user_text(convo).lower()
        wants_questions = any(kw in user_text for kw in (
            "answer the questions",
            "extract the questions",
            "questions from this",
            "worksheet",
            "question set",
            "list the questions",
        ))
        file_paths = _extract_file_paths(convo) if have_tools else []

        if have_tools and wants_questions and file_paths:
            try:
                call_id = "pre_assistant_1"
                args = {
                    "instruction": (
                        "Extract every question prompt in the document verbatim when possible. "
                        "Return a clean numbered list (1., 2., …). Ignore answers, rubrics, and non-question text."
                    ),
                    "path": file_paths[0],
                    "max_chars": 200000,
                }
                # Let UI prepare placeholders and set pill label
                yield DATA + orjson.dumps({
                    "type": "tool_calls",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "assistant", "arguments": args}
                    }]
                }) + END

                # Mirror into the conversation like a normal model tool call
                # IMPORTANT: Ollama expects function.arguments as a JSON string
                convo.append({"role": "assistant", "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "assistant", "arguments": orjson.dumps(args).decode()}
                }]})

                # Execute assistant tool (reuse same logic as dispatcher)
                instr = args["instruction"].strip()
                max_chars = int(args.get("max_chars") or 200000)
                pth = str(args.get("path") or "").strip()
                tool_payload: Dict[str, Any]
                src_text = ""
                source_label = ""
                # Path guard + extraction
                try:
                    real = os.path.realpath(pth)
                    if not real.startswith(TMP_DIR + os.sep):
                        raise ValueError("path not allowed")
                    low = real.lower()
                    if low.endswith(".pdf"):
                        src_text = _pdf_text(real) or _ocr_pdf(real)
                    elif low.endswith(".docx"):
                        src_text = _docx_text(real)
                    elif low.endswith(".rtf"):
                        src_text = _rtf_text(real)
                    elif low.endswith(".html") or low.endswith(".htm"):
                        src_text = _html_text(real)
                    elif any(low.endswith(suf) for suf in (".png", ".jpg", ".jpeg", ".webp")):
                        src_text = _ocr_image(real)
                    else:
                        try:
                            with open(real, "rb") as f:
                                src_text = f.read().decode("utf-8", "replace")
                        except Exception:
                            src_text = ""
                    source_label = os.path.basename(real)
                except Exception as e:
                    src_text = ""
                    tool_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}

                if src_text and len(src_text.strip()) >= 10:
                    if len(src_text) > max_chars:
                        src_text = src_text[:max_chars]
                    # Chunk and summarize with small model
                    chunk_size = 12000
                    overlap = 2000
                    segs: List[str] = []
                    i = 0
                    n = len(src_text)
                    while i < n:
                        end = min(n, i + chunk_size)
                        segs.append(src_text[i:end])
                        if end >= n:
                            break
                        i = max(end - overlap, i + 1)
                    partials: List[str] = []
                    for seg in segs:
                        sys = (
                            "You are a capable file assistant. Perform the TASK on the provided text chunk only. "
                            "Return useful results (extractions, lists, or concise prose). Do not reveal chain-of-thought."
                        )
                        user = f"TASK: {instr}\n\nTEXT CHUNK:\n{seg}"
                        req = {
                            "model": REASON_SUMMARY_MODEL,
                            "prompt": sys + "\n\n" + user,
                            "stream": False,
                            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "num_ctx": min(DEFAULT_NUM_CTX, 8192), "num_predict": 800},
                        }
                        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=60.0)
                        if r.status_code < 400:
                            try:
                                j = r.json(); partials.append((j.get("response") or "").strip())
                            except Exception:
                                partials.append(r.text.strip())
                        else:
                            partials.append("")
                    merged = "\n\n".join([p for p in partials if p])[: max(2000, min(20000, max_chars // 4))]
                    sys2 = (
                        "You are a consolidator. Merge the following partial results into a coherent final result that fulfills the TASK. "
                        "Be faithful to the provided content; do not speculate. No chain-of-thought."
                    )
                    user2 = f"TASK: {instr}\n\nPARTIAL RESULTS:\n{merged}"
                    req2 = {
                        "model": REASON_SUMMARY_MODEL,
                        "prompt": sys2 + "\n\n" + user2,
                        "stream": False,
                        "options": {"temperature": 0.25, "top_p": 0.9, "top_k": 40, "num_ctx": min(DEFAULT_NUM_CTX, 8192), "num_predict": 900},
                    }
                    r2 = await client.post(f"{OLLAMA_HOST}/api/generate", json=req2, timeout=75.0)
                    final_text = ""
                    if r2.status_code < 400:
                        try:
                            j2 = r2.json(); final_text = (j2.get("response") or "").strip()
                        except Exception:
                            final_text = r2.text.strip()
                    tool_payload = {"ok": True, "result": final_text, "consumed_chars": len(src_text), "chunks": len(segs), "source": source_label}
                else:
                    tool_payload = {"ok": False, "error": "no source text (path/doc_id)"}

                # Emit tool_result event
                yield DATA + orjson.dumps({
                    "type": "tool_result",
                    "id": call_id,
                    "name": "assistant",
                    "args": args,
                    "output": tool_payload,
                }) + END

                # Append tool message so the model can consume it
                convo.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "assistant",
                    "content": orjson.dumps(tool_payload).decode(),
                })

                # Nudge inline citation usage
                try:
                    if tool_payload.get("ok") and tool_payload.get("source"):
                        convo.append({"role": "system", "content": (
                            f"When using results from the assistant tool, include a short inline 'From your docs:' citation like '(From your docs — {tool_payload.get('source')})'."
                        )})
                except Exception:
                    pass

                ran_pre_assistant = True
            except Exception:
                # Fail silently – model can still proceed
                ran_pre_assistant = False

        while True:
            nudged_write = False
            options = build_options(settings, convo)
            # Enforce a 6k baseline context for qwen3 and gemma 4B families
            try:
                if _is_qwen3_or_gemma4b(active_model):
                    dyn = bool(settings.get("dynamic_ctx", True))
                    if dyn:
                        max_ctx = int(settings.get("max_ctx", USER_MAX_CTX))
                        options["num_ctx"] = clamp(int(options.get("num_ctx", 0)), 6000, max_ctx)
            except Exception:
                # Non-fatal: fall back to previously computed options
                pass
            req: Dict[str, Any] = {
                "model": MODEL,
                "messages": convo,
                "stream": True,
                "options": options,
            }
            if have_tools:
                req["tools"] = TOOLS
            tool_calls = []
            done_payload = None
            try:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=req, timeout=None) as resp:
                    status_code = resp.status_code
                    if status_code >= 400:
                        try:
                            raw = await resp.aread()
                        except Exception:
                            raw = b""
                        text = raw.decode("utf-8", "ignore").strip()
                        err_msg = f"HTTP {status_code} from model host"
                        if text:
                            try:
                                payload_err = orjson.loads(text)
                                if isinstance(payload_err, dict):
                                    detail = payload_err.get("error") or payload_err.get("message") or payload_err.get("detail")
                                    if detail:
                                        err_msg = f"HTTP {status_code}: {detail}"
                                else:
                                    err_msg = f"HTTP {status_code}: {text}"
                            except Exception:
                                err_msg = f"HTTP {status_code}: {text}"
                        yield DATA + orjson.dumps({"type": "error", "message": err_msg}) + END
                        done_payload = {"type": "error", "message": err_msg}
                        break
                    buffer = b""
                    # Ensure we always have a defined mapping to query, even before any lines parse
                    data: Dict[str, Any] = {}
                    saw_payload = False
                    last_event: Optional[str] = None
                    async for chunk in resp.aiter_bytes():
                        if not chunk:
                            continue
                        buffer += chunk
                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith(b":"):
                                # Comment/heartbeat per SSE spec
                                continue
                            if line.lower().startswith(b"event:"):
                                try:
                                    last_event = line.split(b":", 1)[1].strip().decode()
                                except Exception:
                                    last_event = None
                                continue
                            if line.lower().startswith(b"data:"):
                                line = line.split(b":", 1)[1].strip()
                                if not line:
                                    continue

                            if line in (b"[DONE]", b"done", b"{\"done\":true}"):
                                data = {"done": True}
                            else:
                                try:
                                    data = orjson.loads(line)
                                except Exception:
                                    # Some servers send event: done + data: {}
                                    if (last_event or "").lower() == "done":
                                        data = {"done": True}
                                    else:
                                        continue

                            saw_payload = True
                            msg = data.get("message", {})
                            if isinstance(msg, dict) and "content" in msg:
                                yield DATA + orjson.dumps({"type": "delta", "delta": msg["content"]}) + END

                            err_text = None
                            if isinstance(data, dict):
                                if "error" in data and data["error"]:
                                    err_text = str(data["error"])
                                elif status_code >= 400 and not data.get("done") and not msg:
                                    # Non-OK response without explicit error field
                                    err_text = f"HTTP {status_code}: {data}"
                            if err_text:
                                yield DATA + orjson.dumps({"type": "error", "message": err_text}) + END
                                done_payload = {"type": "error"}
                                break

                            if isinstance(msg, dict) and "tool_calls" in msg:
                                tool_calls = msg["tool_calls"]
                                # Let the UI know about tool_calls payload
                                yield DATA + orjson.dumps({"type": "tool_calls", "tool_calls": tool_calls}) + END
                                # Stop reading more deltas; we'll execute the tools now
                                break

                        # If the last parsed event indicated completion, finalize this round
                        if isinstance(data, dict) and data.get("done"):
                            metrics = data.get("metrics", {})
                            usage = {
                                "prompt_eval_count": metrics.get("prompt_eval_count"),
                                "eval_count": metrics.get("eval_count"),
                                "total_duration_ms": int(metrics.get("total_duration", 0) / 1e6)
                                if metrics.get("total_duration") else None,
                                "eval_duration_ms": int(metrics.get("eval_duration", 0) / 1e6)
                                if metrics.get("eval_duration") else None,
                            }
                            done_payload = {"type": "done", "options": options, "usage": usage}
                            break
                    else:
                        # aiter_bytes exhausted without break -> stream ended naturally
                        if not saw_payload and status_code >= 400:
                            yield DATA + orjson.dumps({"type": "error", "message": f"HTTP {status_code} from model host"}) + END
                            done_payload = {"type": "error"}
            except httpx.RequestError as e:
                yield DATA + orjson.dumps({"type": "error", "message": f"Backend request failed: {e}"}) + END
                break
            except Exception as e:
                yield DATA + orjson.dumps({"type": "error", "message": f"Unexpected error: {e.__class__.__name__}: {e}"}) + END
                break

            if tool_calls:
                # Mirror the tool_calls back into the convo so the model sees it
                convo.append({"role": "assistant", "tool_calls": tool_calls})

                for tc in tool_calls:
                    call_id = tc.get("id")
                    name = tc.get("function", {}).get("name")
                    args_raw = tc.get("function", {}).get("arguments") or {}
                    args = {}
                    try:
                        args = orjson.loads(args_raw) if isinstance(args_raw, str) else args_raw

                        # Dispatch
                        if name == "web_search":
                            tool_payload = await web_search(args.get("query", ""), int(args.get("k", 5)))
                        elif name == "open_url":
                            tool_payload = await open_url(args["url"], int(args.get("max_chars", 6000)))
                            # Auto-ingest opened page into docstore
                            try:
                                if isinstance(tool_payload, dict) and tool_payload.get("ok") and isinstance(tool_payload.get("page"), dict):
                                    page = tool_payload["page"]
                                    text = (page.get("text") or "").strip()
                                    if text:
                                        uri = page.get("canonical_url") or page.get("url") or args.get("url")
                                        title = page.get("title") or (uri or "")
                                        host = page.get("site_name") or "web"
                                        doc_id = f"web:{uri}" if uri else f"web:{title[:80]}"
                                        await get_store().ingest(app.state.client, doc_id=doc_id, text=text, source="web", uri=uri, title=title, tags=[str(host)])
                            except Exception:
                                pass
                            try:
                                page_info = tool_payload.get("page") if isinstance(tool_payload, dict) else None
                                host_raw = (page_info.get("host") or "") if isinstance(page_info, dict) else ""
                                host = host_raw.lower()
                                if host and host not in OFFICIAL_HOSTS:
                                    convo.append({
                                        "role": "system",
                                        "content": (
                                            f"Note: {host_raw or host} is not on the trusted-official list. Treat dates/specs as unverified until confirmed by an official site."
                                        )
                                    })
                            except Exception:
                                pass
                        elif name == "summarize_file":
                            # Validate path stays under TMP_DIR
                            p = str(args.get("path") or "").strip()
                            if not p:
                                tool_payload = {"ok": False, "error": "missing path"}
                            else:
                                try:
                                    real = os.path.realpath(p)
                                    if not real.startswith(TMP_DIR + os.sep):
                                        raise ValueError("path not allowed")
                                    # Extract text
                                    text = ""
                                    low = real.lower()
                                    if low.endswith(".pdf"):
                                        text = _pdf_text(real) or _ocr_pdf(real)
                                    elif low.endswith(".docx"):
                                        text = _docx_text(real)
                                    elif low.endswith(".rtf"):
                                        text = _rtf_text(real)
                                    elif low.endswith(".html") or low.endswith(".htm"):
                                        text = _html_text(real)
                                    elif any(low.endswith(suf) for suf in (".png", ".jpg", ".jpeg", ".webp")):
                                        text = _ocr_image(real)
                                    else:
                                        # try plain text
                                        try:
                                            with open(real, "rb") as f:
                                                text = f.read().decode("utf-8", "replace")
                                        except Exception:
                                            text = ""
                                    if not text or len(text.strip()) < 10:
                                        tool_payload = {"ok": False, "error": "no extractable text", "path": real}
                                    else:
                                        # Summarize with small model
                                        take = int(args.get("max_chars") or 12000)
                                        snippet = text[: take]
                                        sys = (
                                            "You are a careful summarizer. Produce a clear, faithful summary of the file contents. "
                                            "No chain-of-thought; emphasize key sections, entities, and data."
                                        )
                                        prompt = f"Summarize the following file contents in 1–3 short paragraphs.\n\n{snippet}"
                                        req = {
                                            "model": REASON_SUMMARY_MODEL,
                                            "prompt": sys + "\n\n" + prompt,
                                            "stream": False,
                                            "options": {"temperature": 0.2, "top_p": 0.9, "top_k": 50},
                                        }
                                        rsum = await client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=60.0)
                                        if rsum.status_code >= 400:
                                            detail = (rsum.text or "").strip()
                                            tool_payload = {"ok": False, "error": f"summarizer failed: {detail}", "path": real}
                                        else:
                                            sj = rsum.json()
                                            summary = (sj.get("response") or "").strip()
                                            tool_payload = {"ok": True, "path": real, "summary": summary, "length": len(text)}
                                except Exception as e:
                                    tool_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        elif name == "read_whole_file":
                            p = str(args.get("path") or "").strip()
                            if not p:
                                tool_payload = {"ok": False, "error": "missing path"}
                            else:
                                try:
                                    real = os.path.realpath(p)
                                    if not real.startswith(TMP_DIR + os.sep):
                                        raise ValueError("path not allowed")
                                    text = ""
                                    low = real.lower()
                                    if low.endswith(".pdf"):
                                        text = _pdf_text(real) or _ocr_pdf(real)
                                    elif low.endswith(".docx"):
                                        text = _docx_text(real)
                                    elif low.endswith(".rtf"):
                                        text = _rtf_text(real)
                                    elif low.endswith(".html") or low.endswith(".htm"):
                                        text = _html_text(real)
                                    elif any(low.endswith(suf) for suf in (".png", ".jpg", ".jpeg", ".webp")):
                                        text = _ocr_image(real)
                                    else:
                                        try:
                                            with open(real, "rb") as f:
                                                text = f.read().decode("utf-8", "replace")
                                        except Exception:
                                            text = ""
                                    if not text:
                                        tool_payload = {"ok": False, "error": "no extractable text", "path": real}
                                    else:
                                        cap = int(args.get("max_chars") or 100000)
                                        out = text if len(text) <= cap else text[:cap]
                                        tool_payload = {"ok": True, "path": real, "content": out, "truncated": len(out) < len(text)}
                                except Exception as e:
                                    tool_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                        elif name == "assistant":
                            instr = (args.get("instruction") or "").strip()
                            if not instr:
                                tool_payload = {"ok": False, "error": "missing instruction"}
                            else:
                                max_chars = int(args.get("max_chars") or 200000)
                                src_text = ""
                                source_label = ""
                                # Optional path input
                                pth = (args.get("path") or "").strip()
                                if pth:
                                    try:
                                        real = os.path.realpath(pth)
                                        if not real.startswith(TMP_DIR + os.sep):
                                            raise ValueError("path not allowed")
                                        low = real.lower()
                                        if low.endswith(".pdf"):
                                            src_text = _pdf_text(real) or _ocr_pdf(real)
                                        elif low.endswith(".docx"):
                                            src_text = _docx_text(real)
                                        elif low.endswith(".rtf"):
                                            src_text = _rtf_text(real)
                                        elif low.endswith(".html") or low.endswith(".htm"):
                                            src_text = _html_text(real)
                                        elif any(low.endswith(suf) for suf in (".png", ".jpg", ".jpeg", ".webp")):
                                            src_text = _ocr_image(real)
                                        else:
                                            try:
                                                with open(real, "rb") as f:
                                                    src_text = f.read().decode("utf-8", "replace")
                                            except Exception:
                                                src_text = ""
                                        source_label = os.path.basename(real)
                                    except Exception as e:
                                        tool_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                                # Optional doc_id input when no usable path text
                                if not src_text:
                                    did = (args.get("doc_id") or "").strip()
                                    if did:
                                        try:
                                            res = get_store().get_document_text(did, max_chars=max_chars)
                                            if isinstance(res, dict) and res.get("ok"):
                                                src_text = (res.get("text") or "").strip()
                                                meta = res.get("meta") or {}
                                                ttl = (meta.get("title") or "").strip()
                                                source_label = ttl or did
                                        except Exception:
                                            pass
                                if not src_text or len(src_text.strip()) < 10:
                                    tool_payload = {"ok": False, "error": "no source text (path/doc_id)"}
                                else:
                                    if len(src_text) > max_chars:
                                        src_text = src_text[:max_chars]
                                    # Chunking strategy for high effective context
                                    chunk_size = 12000
                                    overlap = 2000
                                    segs: List[str] = []
                                    i = 0
                                    n = len(src_text)
                                    while i < n:
                                        end = min(n, i + chunk_size)
                                        segs.append(src_text[i:end])
                                        if end >= n:
                                            break
                                        i = max(end - overlap, i + 1)
                                    partials: List[str] = []
                                    for seg in segs:
                                        sys = (
                                            "You are a capable file assistant. Perform the TASK on the provided text chunk only. "
                                            "Return useful results (extractions, lists, or concise prose). Do not reveal chain-of-thought."
                                        )
                                        user = f"TASK: {instr}\n\nTEXT CHUNK:\n{seg}"
                                        req = {
                                            "model": REASON_SUMMARY_MODEL,
                                            "prompt": sys + "\n\n" + user,
                                            "stream": False,
                                            "options": {
                                                "temperature": 0.2,
                                                "top_p": 0.9,
                                                "top_k": 40,
                                                "num_ctx": min(DEFAULT_NUM_CTX, 8192),
                                                "num_predict": 800,
                                            },
                                        }
                                        r = await client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=60.0)
                                        if r.status_code < 400:
                                            try:
                                                j = r.json(); partials.append((j.get("response") or "").strip())
                                            except Exception:
                                                partials.append(r.text.strip())
                                        else:
                                            partials.append("")
                                    merged = "\n\n".join([p for p in partials if p])[: max(2000, min(20000, max_chars // 4))]
                                    sys2 = (
                                        "You are a consolidator. Merge the following partial results into a coherent final result that fulfills the TASK. "
                                        "Be faithful to the provided content; do not speculate. No chain-of-thought."
                                    )
                                    user2 = f"TASK: {instr}\n\nPARTIAL RESULTS:\n{merged}"
                                    req2 = {
                                        "model": REASON_SUMMARY_MODEL,
                                        "prompt": sys2 + "\n\n" + user2,
                                        "stream": False,
                                        "options": {
                                            "temperature": 0.25,
                                            "top_p": 0.9,
                                            "top_k": 40,
                                            "num_ctx": min(DEFAULT_NUM_CTX, 8192),
                                            "num_predict": 900,
                                        },
                                    }
                                    r2 = await client.post(f"{OLLAMA_HOST}/api/generate", json=req2, timeout=75.0)
                                    final_text = ""
                                    if r2.status_code < 400:
                                        try:
                                            j2 = r2.json(); final_text = (j2.get("response") or "").strip()
                                        except Exception:
                                            final_text = r2.text.strip()
                                    tool_payload = {
                                        "ok": True,
                                        "result": final_text,
                                        "consumed_chars": len(src_text),
                                        "chunks": len(segs),
                                        "source": source_label,
                                    }
                                    # Nudge the main model to include an inline citation
                                    try:
                                        if source_label:
                                            convo.append({
                                                "role": "system",
                                                "content": (
                                                    f"When using results from the assistant tool, include a short inline 'From your docs:' citation like '(From your docs — {source_label})'."
                                                )
                                            })
                                    except Exception:
                                        pass
                        elif name == "search_docs":
                            q = args.get("query", "")
                            k = int(args.get("k", 6))
                            filters = args.get("filters") if isinstance(args.get("filters"), dict) else {}
                            rerank = bool(args.get("rerank", False))
                            topn = max(k, 12) if rerank else k
                            tool_payload = await get_store().hybrid_search(app.state.client, query=q, k=topn, filters=filters)
                            if isinstance(tool_payload, dict) and not tool_payload.get("results") and tool_payload.get("reason") == "no_relevant":
                                tool_payload.setdefault("message", "no relevant docs")
                                try:
                                    convo.append({"role": "system", "content": "Doc search returned no relevant matches. Rely on other sources."})
                                except Exception:
                                    pass
                            # Optional re-rank with summarizer model
                            if rerank and isinstance(tool_payload, dict) and tool_payload.get("ok"):
                                items = tool_payload.get("results") or []
                                try:
                                    order = await _rerank_with_llm(app.state.client, q, items)
                                    if order:
                                        id_to_item = {int(it.get("id")): it for it in items}
                                        new_items = [id_to_item[i] for i in order if i in id_to_item]
                                        tool_payload["results"] = new_items[:k]
                                except Exception:
                                    tool_payload["results"] = (items or [])[:k]
                            # Encourage inline citations
                            try:
                                if q:
                                    convo.append({"role": "system", "content": (
                                        "When using information from search_docs results, cite inline using (host — title). "
                                        "Prefer short, precise quotes when helpful and avoid vague attributions."
                                    )})
                            except Exception:
                                pass
                        elif name == "eval_expr":
                            tool_payload = await eval_expr(args.get("expr", ""))
                        elif name == "execute":
                            tool_payload = await execute(args.get("code", ""))
                        elif name == "read_file":
                            tool_payload = await read_file(args.get("path", ""))
                        elif name == "write_file":
                            tool_payload = await write_file(args.get("path", ""), args.get("contents", ""))
                            # Auto-ingest written content as a file doc
                            try:
                                pth = (args.get("path") or "").strip()
                                contents = (args.get("contents") or "").strip()
                                if pth and contents:
                                    await get_store().ingest(app.state.client, doc_id=f"file:{pth}", text=contents, source="file", uri=pth, title=pth, tags=["file"])
                            except Exception:
                                pass
                        elif name == "terminal_open":
                            tool_payload = await terminal_open()
                        elif name == "terminal_run":
                            tool_payload = await terminal_run(args.get("cmd", ""))
                            # Optional auto-heal: if terminal wasn't open, open and retry once
                            if isinstance(tool_payload, dict) and tool_payload.get("error") == "terminal not open":
                                _ = await terminal_open()
                                tool_payload = await terminal_run(args.get("cmd", ""))
                        elif name == "terminal_terminate":
                            tool_payload = await terminal_terminate()
                        elif name == "notes_write":
                            tool_payload = await notes_write(args.get("key", ""), args.get("content", ""))
                            # Auto-ingest note content
                            try:
                                key = (args.get("key") or "").strip()
                                content = (args.get("content") or "").strip()
                                if key and content:
                                    await get_store().ingest(app.state.client, doc_id=f"note:{key}", text=content, source="note", uri=f"note:{key}", title=key, tags=["note"])
                            except Exception:
                                pass
                        elif name == "notes_list":
                            tool_payload = await notes_list()
                        elif name == "notes_read":
                            tool_payload = await notes_read(args.get("key", ""))
                        elif name == "user_prefs_write":
                            tool_payload = await user_prefs_write(args.get("key", ""), args.get("content", ""))
                        elif name == "user_prefs_list":
                            tool_payload = await user_prefs_list()
                        elif name == "user_prefs_read":
                            tool_payload = await user_prefs_read(args.get("key", ""))
                        elif name == "open_related_links":
                            tool_payload = await open_related_links(args.get("url", ""), args.get("query", ""), int(args.get("k", 5)))
                        else:
                            tool_payload = {"ok": False, "error": f"Unknown tool {name}"}
                    except Exception as e:
                        tool_payload = {"ok": False, "error": f"{type(e).__name__}: {e}"}

                    # UI event for this tool result
                    yield DATA + orjson.dumps({
                        "type": "tool_result",
                        "id": call_id,
                        "name": name,
                        "args": args,
                        "output": tool_payload,
                    }) + END

                    # Append tool message for the model (include name and raw JSON)
                    convo.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        # Keep JSON content so the model can parse per the developer prompt
                        "content": orjson.dumps(tool_payload).decode(),
                    })

                    # If this was a web_search, immediately nudge the model to open URLs
                    if name == "web_search":
                        try:
                            top_urls = []
                            if isinstance(tool_payload, dict) and tool_payload.get("ok") and isinstance(tool_payload.get("results"), list):
                                ranked = tool_payload.get("results", [])
                                # Already host-deduped and scored in tools.py
                                for r in ranked[:3]:
                                    u = (r or {}).get("url")
                                    if isinstance(u, str) and u.startswith("http"):
                                        top_urls.append(u)
                            if top_urls:
                                # Proactive background ingestion of top links (non-blocking)
                                async def _ingest_url(u: str):
                                    try:
                                        page_res = await open_url(u, int(args.get("max_chars", 2000)))
                                        if isinstance(page_res, dict) and page_res.get("ok") and isinstance(page_res.get("page"), dict):
                                            page = page_res["page"]
                                            text = (page.get("text") or "").strip()
                                            if text:
                                                uri = page.get("canonical_url") or page.get("url") or u
                                                title = page.get("title") or uri
                                                host = page.get("site_name") or "web"
                                                await get_store().ingest(client, doc_id=f"web:{uri}", text=text, source="web", uri=uri, title=title, tags=[str(host)])
                                    except Exception:
                                        pass
                                try:
                                    for u in top_urls[:3]:
                                        asyncio.create_task(_ingest_url(u))
                                except Exception:
                                    pass

                                # Nudge model: open URLs, then consult docstore with the same query
                                q = (args.get("query") or "").strip()
                                instruction = (
                                    "You just performed web_search. Immediately call open_url on the top-ranked links below.\n"
                                    + "\n".join(f"- {u}" for u in top_urls)
                                    + (f"\nAfter at least one open_url succeeds, call search_docs with query: '{q}' to retrieve relevant chunks from previously ingested sources before you draft the answer." if q else "\nAfter at least one open_url succeeds, call search_docs with your original query to retrieve relevant chunks from previously ingested sources before you draft the answer.")
                                    + "\nPolicy: Do not produce prose until at least one open_url succeeds. Prefer authoritative sources (official sites, docs, Wikipedia)."
                                )
                                convo.append({"role": "system", "content": instruction})
                        except Exception:
                            # Non-fatal; continue the loop
                            pass

                    # After successful evidence-gathering tools, prompt the model to write
                    try:
                        if (not nudged_write) and isinstance(tool_payload, dict) and tool_payload.get("ok"):
                            if name in ("open_url", "search_docs", "assistant", "summarize_file", "read_whole_file"):
                                convo.append({"role": "system", "content": (
                                    "You have gathered source material. If at least one authoritative page was opened or relevant doc chunks were retrieved, "
                                    "draft the final answer now with citations. Only continue using tools if you truly need more evidence."
                                )})
                                nudged_write = True
                    except Exception:
                        pass
                # Loop again: the model can now issue another batch without user intervention
                continue

            if done_payload:
                yield DATA + orjson.dumps(done_payload) + END
            else:
                # No completion payload and no tools to run; surface a generic error to client
                yield DATA + orjson.dumps({"type": "error", "message": "Model returned no data."}) + END
            break

        # End the SSE stream
        yield b"event: close\ndata: {}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# Docstore endpoints
@app.post("/api/ingest")
async def api_ingest(doc: Dict[str, Any] = Body(...)):
    required = (doc.get("doc_id"), doc.get("text"), doc.get("source"))
    if not all(required):
        raise HTTPException(status_code=400, detail="doc_id, text, source are required")
    res = await get_store().ingest(
        app.state.client,
        doc_id=str(doc.get("doc_id")),
        text=str(doc.get("text")),
        source=str(doc.get("source")),
        uri=(doc.get("uri") or None),
        title=(doc.get("title") or None),
        tags=(doc.get("tags") or None),
        meta=(doc.get("meta") or None),
    )
    return res


@app.post("/api/search")
async def api_search(body: Dict[str, Any] = Body(...)):
    query = (body.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    k = int(body.get("k") or 6)
    filters = body.get("filters") if isinstance(body.get("filters"), dict) else {}
    bm25_boost = float(body.get("bm25_boost") or 0.25)
    rerank = bool(body.get("rerank", False))
    topn = max(k, 12) if rerank else k
    res = await get_store().hybrid_search(app.state.client, query=query, k=topn, filters=filters, bm25_boost=bm25_boost)
    if rerank and isinstance(res, dict) and res.get("ok"):
        items = res.get("results") or []
        order = await _rerank_with_llm(app.state.client, query, items)
        if order:
            id_to_item = {int(it.get("id")): it for it in items}
            res["results"] = [id_to_item[i] for i in order if i in id_to_item][:k]
    return res


@app.post("/api/gemma3")
async def gemma3_vision(
    prompt: str = Form(""),
    model: str = Form(REASONING_OFF_MODEL),
    reasoning: str = Form("off"),
    history: str = Form("[]"),
    image: Optional[UploadFile] = File(None),
):
    history_entries: List[Dict[str, Any]] = []
    try:
        parsed = orjson.loads(history) if history else []
        if isinstance(parsed, list):
            history_entries = [msg for msg in parsed if isinstance(msg, dict)]
    except Exception:
        history_entries = []

    messages: List[Dict[str, Any]] = []
    system_prompt = "You are a helpful assistant. Describe images accurately and keep answers concise."
    messages.append({"role": "system", "content": system_prompt})

    for msg in history_entries:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not isinstance(content, str):
            try:
                content = orjson.dumps(content).decode()
            except Exception:
                content = str(content)
        messages.append({"role": role, "content": content})

    prompt_text = (prompt or "").strip() or "Describe the image."

    images_list: List[str] = []
    if image is not None:
        data = await image.read()
        if data:
            images_list.append(base64.b64encode(data).decode("ascii"))

    messages.append({"role": "user", "content": prompt_text, **({"images": images_list} if images_list else {})})

    option_messages = list(history_entries)
    option_messages.append({"role": "user", "content": prompt_text})
    options = build_options({}, option_messages)
    # Enforce a 6k baseline context for qwen3 and gemma 4B families (vision endpoint)
    try:
        model_used = (model or MODEL)
        if _is_qwen3_or_gemma4b(model_used):
            # build_options defaults to dynamic_ctx=True when settings are {}, so adjust here
            options["num_ctx"] = clamp(int(options.get("num_ctx", 0)), 6000, USER_MAX_CTX)
    except Exception:
        pass

    req: Dict[str, Any] = {
        "model": model or MODEL,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    try:
        resp = await app.state.client.post(f"{OLLAMA_HOST}/api/chat", json=req, timeout=None)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Backend request failed: {exc}") from exc

    # If the chosen tag rejects images, try a known vision-capable default
    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = {}
        detail = (payload.get("error") or payload.get("message") or resp.text or "").strip()
        # Fallback to gemma3:4b for vision if initial model fails
        alt_model = "gemma3:4b"
        if (model or "").strip() != alt_model:
            req["model"] = alt_model
            resp = await app.state.client.post(f"{OLLAMA_HOST}/api/chat", json=req, timeout=None)
            if resp.status_code >= 400:
                try:
                    payload = resp.json()
                except Exception:
                    payload = {}
                detail2 = payload.get("error") or payload.get("message") or resp.text.strip() or (detail or "Vision request failed")
                raise HTTPException(status_code=resp.status_code, detail=detail2)
        else:
            raise HTTPException(status_code=resp.status_code, detail=detail or "Vision request failed")

    data = resp.json()
    message = data.get("message")
    answer = ""
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            answer = "\n\n".join(filter(None, parts))
        elif isinstance(content, str):
            answer = content

    if not answer:
        answer = data.get("response") or data.get("reply") or data.get("output") or ""

    return {"response": answer}


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...), tags: Optional[str] = Form(None)):
    """Accepts uploaded files, extracts text (PDF text layer → OCR fallback), saves into tmp/, and ingests into docstore.
    Returns a list of ingested entries with doc_ids.
    """
    results: List[Dict[str, Any]] = []
    tag_list: List[str] = []
    try:
        if tags:
            import json as _json
            try:
                tag_list = _json.loads(tags)
            except Exception:
                tag_list = [tags]
    except Exception:
        tag_list = []

    for uf in files:
        try:
            raw = await uf.read()
            if not raw:
                results.append({"name": uf.filename, "ok": False, "error": "empty file"})
                continue
            sha = _sha1(raw)
            ext = _guess_ext(uf.filename or "", uf.content_type)
            safe = _safe_name(uf.filename or f"upload-{sha}{ext}")
            save_path = os.path.join(TMP_DIR, safe)
            try:
                with open(save_path, "wb") as f:
                    f.write(raw)
            except Exception as e:
                results.append({"name": uf.filename, "ok": False, "error": str(e)})
                continue

            text = ""
            ctype = (uf.content_type or "").lower()
            low = (uf.filename or "").lower()
            if ctype.startswith("text/") or any(low.endswith(suf) for suf in (".txt", ".md", ".csv", ".json")):
                try:
                    text = raw.decode("utf-8", "replace")
                except Exception:
                    text = ""
            elif low.endswith(".pdf") or ctype == "application/pdf":
                text = _pdf_text(save_path)
                if len(text) < 200:
                    text = _ocr_pdf(save_path)
            elif low.endswith(".docx") or ctype in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
                text = _docx_text(save_path)
            elif low.endswith(".rtf") or ctype in ("application/rtf", "text/rtf"):
                text = _rtf_text(save_path)
            elif low.endswith(".html") or low.endswith(".htm") or ctype in ("text/html",):
                text = _html_text(save_path)
            elif ctype.startswith("image/") or any(low.endswith(suf) for suf in (".png", ".jpg", ".jpeg", ".webp")):
                text = _ocr_image(save_path)
            else:
                # try decode as utf-8 fallback
                try:
                    text = raw.decode("utf-8", "replace")
                except Exception:
                    text = ""

            if not text or len(text.strip()) < 10:
                results.append({"name": uf.filename, "ok": False, "error": "no extractable text", "path": save_path})
                continue

            title = uf.filename or f"upload-{sha}"
            doc_id = f"file:{sha}"
            tags2 = list(tag_list) if tag_list else ["file"]
            ing = await get_store().ingest(app.state.client, doc_id=doc_id, text=text, source="file", uri=save_path, title=title, tags=tags2)
            ok = bool(ing.get("ok"))
            results.append({"name": uf.filename, "ok": ok, "doc_id": doc_id, "chunks": ing.get("chunks"), "path": save_path, "title": title})
        except Exception as e:
            results.append({"name": uf.filename, "ok": False, "error": f"{type(e).__name__}: {e}"})
    return {"ok": True, "files": results}


@app.post("/api/reason/summarize")
async def summarize_reasoning(payload: Dict[str, Any] = Body(...)):
    """Summarize hidden thinking into a short, user-friendly step.
    Strictly avoid tool/meta/internal references. Output is post-filtered
    and forced into "**Title**: sentence" format.
    """
    text = (payload.get("text") or "").strip()
    observations = payload.get("observations") if isinstance(payload.get("observations"), list) else []
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    prior = payload.get("prior") if isinstance(payload.get("prior"), list) else []
    topic = (payload.get("topic") or "").strip()
    snapshot = bool(payload.get("snapshot", False))
    if not text and not observations and not pages:
        raise HTTPException(status_code=400, detail="insufficient content")

    # Truncate excessively long chunks to keep summary snappy
    max_in = 2200
    if len(text) > max_in:
        text = text[-max_in:]

    # Strong instruction + JSON format to reduce leakage
    # Extract prior titles and recent search for variety and better titles
    def _extract_prior_titles(items: List[str]) -> List[str]:
        acc: List[str] = []
        for s in items[-5:]:
            try:
                st = str(s)
            except Exception:
                continue
            if "**" in st:
                try:
                    t = st.split("**", 2)[1]
                    if t and t.strip():
                        acc.append(t.strip())
                        continue
                except Exception:
                    pass
            first = st.strip().splitlines()[0]
            if first:
                acc.append(first.strip("* _#"))
        return acc

    prior_titles = _extract_prior_titles(prior)
    recent_search = ""
    try:
        for o in observations[::-1]:
            so = str(o).strip()
            if so.lower().startswith("searched:"):
                recent_search = so.split(":", 1)[-1].strip()
                break
    except Exception:
        recent_search = ""
    topic_hint = topic or recent_search or "the current user request"
    focus = topic_hint

    system = (
        "You are StepSummarizer. Turn hidden reasoning into a crisp, user-friendly progress update.\n"
        "RULES: No tool/meta vocabulary (web_search, function call, JSON, etc.). Never reveal chain-of-thought mechanics.\n"
        "STYLE: Use the Action–Finding–Next pattern in 2–3 first‑person sentences.\n"
        "  • Action: what I just did (Investigating/Verifying/Comparing/Summarizing/Resolving).\n"
        "  • Finding: the strongest take-away so far (what holds or is missing).\n"
        "  • Next: the immediate follow‑up step.\n"
        "TOPIC: Focus on {topic_hint}. If this chunk drifts, briefly note the mismatch while still capturing useful progress.\n"
        "CONSTRAINTS: No filler like 'no progress' or 'still checking'. No URLs. Mention hosts only if essential.\n"
        "TITLE: 3–7 words, present‑tense gerund (e.g., Verifying details for <X>).\n"
        "OUTPUT: STRICT JSON with keys: title (string), paragraph (string)."
    )
    # Clean observations to avoid echoing tool/meta lines
    def _clean_obs_line(s: str) -> str:
        try:
            s = str(s).strip()
        except Exception:
            return ""
        low = s.lower()
        # Drop tool/meta lines entirely
        if any(low.startswith(prefix) for prefix in (
            "web_search", "open_url", "opened:", "preview (", "function:", "tool:", "arguments:", "event:",
        )):
            return ""
        if any(tok in low for tok in ("open_url", "web_search", "tool call", "function call")):
            return ""
        # Drop bare URLs and host-only lines
        if low.startswith("http") or low.startswith("www."):
            return ""
        # Trim and collapse
        s = s.replace("\n", " ").strip()
        if len(s) > 200:
            s = s[:200].rsplit(' ', 1)[0] + '…'
        return s

    _cleaned_obs = []
    for o in observations[:10]:
        co = _clean_obs_line(o)
        if co:
            _cleaned_obs.append(f"- {co}")
    obs_text = "\n".join(_cleaned_obs)
    def page_lines() -> str:
        out: List[str] = []
        for p in pages[:2]:
            try:
                host = (p.get("host") or "").strip()
                title = (p.get("title") or "").strip()
                summ = (p.get("summary") or "").strip()
                if host and title:
                    line = f"- Source: {title} ({host})"
                    if summ:
                        line += f" — {summ[:160]}"
                    out.append(line)
            except Exception:
                continue
        return "\n".join(out)
    page_block = page_lines()
    # Clean topic for subject-friendly phrasing
    def _clean_focus_text(s: str) -> str:
        import re as __re
        s = (s or "").strip()
        s = s.replace("the current user request", "current topic")
        s = __re.sub(r"^tell me about( the)?\s+", "", s, flags=__re.IGNORECASE)
        s = __re.sub(r"^about\s+", "", s, flags=__re.IGNORECASE)
        return s.strip()
    topic_hint = _clean_focus_text(topic_hint)

    user = (
        "Summarize this hidden thinking. Cover evidence gathered, constraints, and partial conclusions.\n"
        "Avoid process/meta commentary; focus on what you learned.\n\n"
        + (f"TOPIC:\n{topic_hint}\n\n" if topic_hint else "")
        + (f"OBSERVATIONS:\n{obs_text}\n\n" if obs_text else "")
        + (f"RECENT SOURCES:\n{page_block}\n\n" if page_block else "")
        + ("PRIOR TITLES:\n- " + "\n- ".join(prior_titles) + "\n\n" if prior_titles else "")
        + f"THINKING:\n{text}\n\n"
        + "Output JSON only."
    )

    req = {
        "model": REASON_SUMMARY_MODEL,
        "prompt": system + "\n\n" + user,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.35 if snapshot else 0.25,
            "top_p": 0.85 if snapshot else 0.8,
            "top_k": 40 if snapshot else 30,
            "num_ctx": min(REASON_SUMMARY_THINK_CTX, DEFAULT_NUM_CTX),
            "num_predict": 220,
        },
    }

    # Fallback summary generator (never leaks raw COT). Uses Action–Finding–Next.
    def _fallback_summary() -> Dict[str, Any]:
        obs = payload.get("observations") if isinstance(payload.get("observations"), list) else []
        pgs = payload.get("pages") if isinstance(payload.get("pages"), list) else []
        t = (payload.get("text") or "")
        t_low = t.lower()

        # Signals
        has_conflict = any(k in t_low for k in ("conflict", "contradict", "disagree"))
        has_network = any(k in t_low for k in ("timeout", "rate limit", "504", "502", "network error"))
        has_confirm = ("confirm" in t_low or "verified" in t_low or "press release" in t_low)
        import re as __re
        pages_blob = " ".join([(pg.get("summary") or "") + " " + (pg.get("title") or "") for pg in pgs])
        has_dates = bool(__re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", t_low + " " + pages_blob.lower())) or bool(__re.search(r"\b20\d{2}\b", t_low + " " + pages_blob))

        # Entity hint from latest search or page title/host
        try:
            searched = [str(o).split(":", 1)[-1].strip() for o in obs if str(o).lower().startswith("searched:")]
        except Exception:
            searched = []
        entity = ""
        if searched:
            q = searched[-1]
            if len(q) > 48:
                q = q[:48].rsplit(' ', 1)[0] + '…'
            entity = q
        elif pgs:
            ptitle = (pgs[-1].get("title") or "").strip()
            entity = ptitle

        # Choose action verb deterministically
        if has_network:
            verb = "Resolving connectivity"
        elif has_conflict:
            verb = "Comparing reports"
        elif has_confirm:
            verb = "Summarizing findings"
        elif has_dates:
            verb = "Verifying dates"
        elif searched or pgs:
            verb = "Cross-checking details"
        else:
            verb = "Investigating context"

        # Compose title (3–7 words)
        subj = entity if entity else (topic_hint or recent_search or "current topic")
        title = f"{verb} for {subj}".strip()
        try:
            words = title.split()
            if len(words) < 3:
                title = f"{verb} for sources"
            elif len(words) > 7:
                title = " ".join(words[:7])
        except Exception:
            pass

        # Build Action–Finding–Next paragraph
        focus = topic_hint or recent_search or "this topic"
        # Clean focus wording for readability
        focus = _clean_focus_text(focus)
        try:
            latest = pgs[-1] if pgs else {}
            host = (latest.get('host') or '').strip()
            ptitle = (latest.get('title') or '').strip()
            psumm = (latest.get('summary') or '').strip()
        except Exception:
            host = ""
            ptitle = ""
            psumm = ""
        finding = ""
        if psumm:
            finding = f"I noted: {psumm[:220]}" + ("…" if len(psumm) > 220 else "")
        elif searched:
            q = searched[-1]
            finding = f"Searched for \"{q[:60]}\" and gathered promising leads." + ("" if len(q) <= 60 else "")
        elif obs:
            first_obs = str(obs[-1]).strip()
            if first_obs:
                finding = f"I captured a useful observation: {first_obs[:180]}" + ("…" if len(first_obs) > 180 else "")
        else:
            finding = f"I consolidated current notes for {focus}."

        if has_conflict:
            nxt = "Next I will reconcile discrepancies using timestamps and wording."
        elif has_confirm:
            nxt = "Next I will capture the confirmed details cleanly."
        elif has_network:
            nxt = "Next I will retry and switch approaches to restore continuity."
        elif has_dates:
            nxt = "Next I will cross-check dates across independent references."
        else:
            nxt = "Next I will corroborate these points across independent references."

        paragraph = f"I am {verb.lower()} on {focus}. {finding} {nxt}"
        final_text = f"**{title}**\n\n{paragraph.strip()}"
        return {"summary": final_text}

    # Try remote summarizer; on failure, return safe fallback with 200 OK
    try:
        async with app.state.sum_lock:
            resp = await app.state.client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=45.0)
        if resp.status_code >= 400:
            # degrade gracefully instead of 4xx/5xx
            return _fallback_summary()
        try:
            data = resp.json()
        except Exception:
            data = {"response": resp.text}
    except httpx.RequestError:
        return _fallback_summary()

    # Helper: extract JSON object from a possibly wrapped string
    def extract_json(s: str) -> Dict[str, Any]:
        s = s.strip()
        # strip code fences
        if s.startswith("```"):
            try:
                s = s.split("```", 2)[1].strip()
            except Exception:
                s = s.replace("```json", "").replace("```", "").strip()
        # direct parse
        try:
            return orjson.loads(s)
        except Exception:
            pass
        # find first/last brace and attempt
        try:
            i = s.find('{')
            j = s.rfind('}')
            if i != -1 and j != -1 and j > i:
                sub = s[i:j+1]
                return orjson.loads(sub)
        except Exception:
            pass
        return {}

    raw = (data.get("response") or "").strip()
    obj: Dict[str, Any] = extract_json(raw) if raw else {}

    # Collect fields (accept paragraph/summary or points[])
    title = (obj.get("title") or "").strip()
    points = obj.get("points") if isinstance(obj, dict) else None
    paragraph = (obj.get("paragraph") or obj.get("summary") or "").strip()
    if (not paragraph) and isinstance(points, list) and points:
        paragraph = " ".join(str(p).strip() for p in points if p)

    # Sanitize points: remove JSON-like wrappers, code fences, and quotes
    def sanitize_text(t: str) -> str:
        t = t.strip()
        # remove wrapping braces if it's a JSON-ish single-field blob
        if t.startswith('{') and t.endswith('}'):
            try:
                inner = orjson.loads(t)
                if isinstance(inner, dict) and inner:
                    # prefer common keys
                    textual_keys = (
                        "summary",
                        "extract",
                        "text",
                        "paragraph",
                        "content",
                        "title",
                        "details",
                        "note",
                    )
                    for k in textual_keys:
                        if k in inner and isinstance(inner[k], str):
                            return inner[k].strip()
                    toolish_keys = {"name", "arguments", "tool", "function", "actions", "action"}
                    if any(k in inner for k in toolish_keys):
                        return ''
                    fallback_parts: List[str] = []
                    for v in inner.values():
                        if isinstance(v, (dict, list)):
                            continue
                        s_val = str(v).strip()
                        if not s_val:
                            continue
                        low_val = s_val.lower()
                        if any(tok in low_val for tok in (
                            'web_search',
                            'open_url',
                            'tool',
                            'function',
                            'arguments',
                            'call',
                        )):
                            continue
                        if '_' in s_val and ' ' not in s_val:
                            continue
                        fallback_parts.append(s_val)
                    if fallback_parts:
                        return ' '.join(fallback_parts)
                    return ''
            except Exception:
                # strip braces/quotes as a fallback
                t = t.strip('{}').strip()
        # strip stray quotes
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()
        # drop code fences
        t = t.replace('```', '').strip()
        # drop obvious tool/JSON fragments within the sentence
        lowered = t.lower()
        bad_infixes = (
            '"name"', '"arguments"', 'open_url', 'web_search', 'function call', 'tool call', 'max_chars', 'search_docs'
        )
        if any(b in lowered for b in bad_infixes) or ('{"' in t or '}' in t or '[' in t or ']' in t):
            # remove most JSON-ish punctuation to salvage any residual text
            for ch in ['{','}','[',']','"']:
                t = t.replace(ch, '')
            low2 = t.lower()
            if any(b in low2 for b in ('open_url', 'web_search', 'arguments', 'name:')) or ':' in t:
                return ''
        # strip chain-of-thought/meta references explicitly
        banned_terms = (
            'hidden reasoning', 'hidden', 'chain of thought', 'chain-of-thought', 'cot', 'internal process', 'meta', 'process',
        )
        if any(bt in lowered for bt in banned_terms):
            return ''
        return t

    # Ensure title uses a gerund pattern; reconstruct if needed
    def ensure_gerund_title(t: str) -> str:
        tl = (t or "").strip()
        if tl and tl.split()[0].lower().endswith("ing") and 3 <= len(tl.split()) <= 7:
            return tl
        # Signals to pick an action verb
        base_text = (text or "").lower()
        has_conflict = any(k in base_text for k in ("conflict", "contradict", "disagree"))
        has_confirm = ("confirm" in base_text or "verified" in base_text or "press release" in base_text)
        # Choose primary verb then adjust to avoid repeating prior titles
        if has_conflict:
            verb, alts = "Comparing reports", ["Resolving discrepancies", "Analyzing contradictions"]
        elif has_confirm:
            verb, alts = "Summarizing findings", ["Finalizing summary", "Consolidating conclusions"]
        else:
            # Detect date-centric focus for variety
            import re as __re
            has_dates = bool(__re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", base_text)) or bool(__re.search(r"\b20\d{2}\b", base_text))
            if has_dates:
                verb, alts = "Verifying dates", ["Cross-checking dates", "Confirming timelines"]
            else:
                verb, alts = "Verifying details", ["Cross-checking details", "Evaluating evidence"]
        # Avoid repeating a prior title verb if possible
        first_token = verb.split()[0].lower()
        if prior_titles:
            prior_first = {p.split()[0].lower() for p in prior_titles if isinstance(p, str) and p}
            if first_token in prior_first and alts:
                verb = alts[0]
        # Subject from recent search or last page title/host
        subj = None
        try:
            searched = [str(o).split(":", 1)[-1].strip() for o in observations if str(o).lower().startswith("searched:")]
        except Exception:
            searched = []
        if searched:
            q = searched[-1]
            if len(q) > 48:
                q = q[:48].rsplit(' ', 1)[0] + '…'
            subj = q
        elif pages:
            ptitle = (pages[-1].get("title") or "").strip()
            host = (pages[-1].get("host") or "source").strip()
            subj = ptitle or host
        else:
            subj = topic_hint or "current topic"
        out = f"{verb} for {subj}".strip()
        words = out.split()
        if len(words) > 7:
            out = " ".join(words[:7])
        if len(out.split()) < 3:
            out = f"{verb} for sources"
        return out

    # Allowed years from recent sources (avoid hallucinated dates)
    import re as _re
    allowed_years: set[str] = set()
    try:
        for p in pages:
            for s in (p.get("summary") or "", p.get("title") or ""):
                for y in _re.findall(r"\b(?:19|20|21)\d{2}\b", s):
                    allowed_years.add(y)
    except Exception:
        pass

    # If remote returned points instead of paragraph, salvage as a paragraph
    uniq_points: List[str] = []
    if not paragraph and isinstance(points, list) and points:
        cleaned: List[str] = []
        for p in points[:5]:
            s = sanitize_text(str(p))
            # Filter out obviously meta/process lines
            bad_fragments = [
                'tool call', 'return only tool calls', 'function call', 'simulate', 'k=', 'assistant will', 'the ai will',
                'the next intended action', 'next intended action', 'i need to', 'i plan to', 'i am going to',
                'current focus', 'recent action', 'focused query', 'organizing the key details',
                'organizing key details', 'reviewing material', 'limited availability of information'
            ]
            if not s or any(fr in s.lower() for fr in bad_fragments):
                continue
            # Filter out unverified years/dates
            yrs = set(_re.findall(r"\b(?:19|20|21)\d{2}\b", s))
            if yrs and not yrs.issubset(allowed_years):
                continue
            # Strip unknown host tags in parentheses
            try:
                hosts_in_s = set(_re.findall(r"\(([^)]+)\)", s))
                if hosts_in_s:
                    page_hosts = { (p.get('host') or '').strip() for p in pages }
                    if not any(h in page_hosts for h in hosts_in_s):
                        s = _re.sub(r"\([^)]*\)", "", s).strip()
            except Exception:
                pass
            cleaned.append(s)
        # Deduplicate and trim
        seen = set()
        deduped: List[str] = []
        for s in cleaned:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            words = s.split()
            if len(words) > 32:
                s = " ".join(words[:32]).rstrip(',;:.') + '…'
            deduped.append(s)
        uniq_points = deduped
        paragraph = " ".join(deduped)

    # If nothing remains, craft a safe summary from observations if provided
    obs = payload.get("observations") if isinstance(payload.get("observations"), list) else []
    hosts = [str(o).split(':',1)[-1].strip() for o in obs if str(o).lower().startswith('opened:')]
    searched = [str(o).split(':',1)[-1].strip() for o in obs if str(o).lower().startswith('searched:')]
    if not paragraph:
        if hosts:
            h = ", ".join(hosts[:2])
            paragraph = (
                f"I’m reviewing material from {h} to extract the essentials and check claims against primary sources. "
                f"The aim is to keep the answer precise and grounded, avoiding speculation and unnecessary detail. "
                f"Next, I’ll consolidate the findings into a clear response."
            )
        else:
            paragraph = (
                "I’m consolidating the most relevant details to address the request clearly. "
                "The focus is on accuracy, succinct structure, and avoiding speculation. "
                "Next, I’ll assemble the final answer based on what’s verified."
            )

    # Strip unverified years from title
    try:
        yrs_title = set(_re.findall(r"\b(?:19|20|21)\d{2}\b", title))
        if yrs_title and not yrs_title.issubset(allowed_years):
            title = _re.sub(r"\b(?:19|20|21)\d{2}\b", "", title).strip()
    except Exception:
        pass

    # Promote a meaningful first point to title when current title is generic or banned
    generic_titles = {"progress", "update", "status", "working", "analysis", "clarifying the request", "reviewing sources"}
    banned_title_terms = ("hidden", "reasoning", "chain of thought", "chain-of-thought", "cot")
    def _is_generic_or_banned(t: str) -> bool:
        tl = (t or '').strip().lower()
        if not tl:
            return True
        if tl in generic_titles:
            return True
        return any(b in tl for b in banned_title_terms)

    if _is_generic_or_banned(title) and uniq_points:
        # Use the first informative point as title and drop it from bullets
        new_title = uniq_points[0]
        if len(new_title) <= 100:
            title = new_title
            uniq_points = uniq_points[1:]
    # If title is still generic/empty/banned, derive from observations
    if _is_generic_or_banned(title):
        if searched:
            q = (searched[-1] or '').strip()
            # keep it short and topic-focused
            if len(q) > 64:
                q = q[:64].rsplit(' ', 1)[0] + '…'
            title = f"Investigating {q}" if q else "Investigating sources"
        elif hosts:
            host = hosts[-1]
            title = f"Reviewing {host}"
        else:
            title = "Clarifying the request"

    # Sanitize title too and enforce gerund form
    title = sanitize_text(title)
    if not title:
        title = "Clarifying the request"
    title = ensure_gerund_title(title)
    # Avoid repeating prior titles (case-insensitive) if provided
    if prior_titles:
        low = title.lower()
        if any(low == p.lower() for p in prior_titles):
            alt = None
            if searched:
                q = (searched[-1] or '').strip()
                if len(q) > 48:
                    q = q[:48].rsplit(' ', 1)[0] + '…'
                alt = f"Exploring {q}" if q else None
            if not alt and hosts:
                alt = f"Reviewing {hosts[-1]}"
            if not alt and pages:
                ptitle2 = (pages[-1].get('title') or '').strip()
                if ptitle2:
                    alt = f"Understanding {ptitle2[:52]}"
            if alt:
                title = alt

    # Sanitize and trim paragraph to ~4 sentences
    paragraph = sanitize_text(paragraph).strip()
    # Remove filler phrases that users dislike
    _banned_fillers = (
        "no progress",
        "still checking",
        "i’ll keep digging",
        "ill keep digging",
        "still gathering evidence",
        "no progress on that yet",
        "checking sources",
    )
    low_para = paragraph.lower()
    if any(b in low_para for b in _banned_fillers):
        import re as __re
        for b in _banned_fillers:
            paragraph = paragraph.replace(b, "Continuing verification", 1) if b in paragraph else paragraph
            paragraph = __re.sub(b, "Continuing verification", paragraph, flags=__re.IGNORECASE)
        paragraph = __re.sub(r"\s+", " ", paragraph).strip()
    topic_terms: List[str] = []
    if topic_hint:
        topic_terms = [t for t in _re.findall(r"\b[a-z0-9]{3,}\b", topic_hint.lower()) if t not in {
            "apple", "iphone", "rumor", "rumors", "specs", "specifications", "release", "date", "dates", "info", "information", "about"
        }]
    allowed_numbers: set[str] = set()
    for source_text in (text, recent_search, topic_hint or "", obs_text, page_block):
        for num in _re.findall(r"\b\d+\b", source_text.lower()):
            allowed_numbers.add(num)
    nums_in_paragraph = set(_re.findall(r"\b\d+\b", paragraph.lower()))
    focus = topic_hint or recent_search or "this request"
    if nums_in_paragraph and allowed_numbers and any(n not in allowed_numbers for n in nums_in_paragraph):
        paragraph = f"I don’t have trusted numbers for {focus} yet—I’m double-checking before sharing specifics."
    elif nums_in_paragraph and not allowed_numbers:
        paragraph = f"I don’t have trusted numbers for {focus} yet—I’m double-checking before sharing specifics."
    try:
        import re as ___re
        sents = [s.strip() for s in ___re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
        if len(sents) > 4:
            paragraph = " ".join(sents[:4])
    except Exception:
        pass

    # If paragraph still missing, synthesize a blunt status update
    if not paragraph:
        summary_parts: List[str] = []
        if obs_text:
            first_obs = obs_text.splitlines()[0].lstrip("- ").strip()
            if first_obs:
                summary_parts.append(f"Recent step: {first_obs}.")
        if pages:
            latest = pages[-1]
            host = (latest.get('host') or '').strip()
            ptitle = (latest.get('title') or '').strip()
            if ptitle or host:
                subject = ptitle if ptitle else host
                summary_parts.append(f"Reviewing {subject} for {focus}.")
            psumm = (latest.get('summary') or '').strip()
            if psumm:
                summary_parts.append(f"Key takeaway: {psumm[:180]}" + ('…' if len(psumm) > 180 else ''))
        if recent_search and len(summary_parts) < 2:
            summary_parts.append(f"Latest query: {recent_search}.")
        if not summary_parts:
            summary_parts.append(f"Continuing analysis of {focus} with gathered material.")
        paragraph = " ".join(summary_parts)

    final_text = f"**{title}**\n\n{paragraph}" if paragraph else f"**{title}**\n\nContinuing analysis of {focus}."
    return {"summary": final_text}


@app.post("/api/reason/finalize")
async def finalize_reasoning(payload: Dict[str, Any] = Body(...)):
    """Generate a user-facing answer when the primary model fails to emit one."""
    question = (payload.get("question") or "").strip()
    thinking = (payload.get("thinking") or "").strip()
    observations = payload.get("observations") if isinstance(payload.get("observations"), list) else []
    pages = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    summaries = payload.get("summaries") if isinstance(payload.get("summaries"), list) else []

    if not (question or thinking or pages or summaries):
        raise HTTPException(status_code=400, detail="insufficient content")

    max_notes = max(1200, REASON_SUMMARY_THINK_CTX)
    if thinking and len(thinking) > max_notes:
        thinking = thinking[-max_notes:]

    obs_text = "\n".join(f"- {str(o)[:240]}" for o in observations[-8:])
    page_lines: List[str] = []
    for p in pages[-4:]:
        try:
            host = str(p.get("host") or "").strip()
            title = str(p.get("title") or "").strip()
            summ = str(p.get("summary") or "").strip()
            line = f"- {title or host or 'Source'}"
            if host:
                line += f" ({host})"
            if summ:
                line += f" — {summ[:260]}"
            page_lines.append(line)
        except Exception:
            continue

    system = (
        "You are FinalComposer. Turn investigation notes into a decisive answer for the user.\n"
        "RULES: Cite concrete facts pulled from the sources. Mention the host/site when referencing evidence (e.g., Apple.com).\n"
        "If only rumor or unofficial sources are available, say so explicitly and highlight the lack of confirmation.\n"
        "Avoid meta-commentary about tools or process. If evidence is missing, state that plainly."
    )

    user_sections: List[str] = []
    if question:
        user_sections.append(f"QUESTION:\n{question}")
    if page_lines:
        user_sections.append("SOURCES:\n" + "\n".join(page_lines))
    if obs_text:
        user_sections.append("OBSERVATIONS:\n" + obs_text)
    if summaries:
        user_sections.append("PRIOR SUMMARIES:\n" + "\n".join(str(s) for s in summaries[-3:]))
    if thinking:
        user_sections.append("NOTES:\n" + thinking)
    user_sections.append("OUTPUT: Write the final response for the user. Be concise, use inline host citations, and list any open questions if specifics remain unverified.")

    user_prompt = "\n\n".join(section for section in user_sections if section)

    req = {
        "model": REASON_SUMMARY_MODEL,
        "prompt": system + "\n\n" + user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.85,
            "top_k": 40,
            "num_ctx": min(DEFAULT_NUM_CTX, 4096),
            "num_predict": 320,
        },
    }

    try:
        async with app.state.sum_lock:
            resp = await app.state.client.post(f"{OLLAMA_HOST}/api/generate", json=req, timeout=45.0)
        if resp.status_code >= 400:
            return {"ok": False, "answer": ""}
        data = resp.json()
    except Exception:
        return {"ok": False, "answer": ""}

    answer = (data.get("response") or "").strip()
    return {"ok": bool(answer), "answer": answer}


@app.post("/api/models/set")
async def set_model(payload: Dict[str, Any]):
    global MODEL
    MODEL = payload.get("model", MODEL)
    return {"ok": True, "model": MODEL}


@app.get("/api/models")
async def list_models():
    r = await app.state.client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
    return JSONResponse(r.json())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",  # module_name:app_instance
        host="127.0.0.1",
        port=8000,
        reload=True,   # optional: auto-reload on file changes
    )
