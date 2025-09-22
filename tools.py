# tools.py
import re
import urllib.parse
from typing import List, Dict, Tuple, Any, Optional

import asyncio
import httpx
from bs4 import BeautifulSoup
from datetime import datetime
import subprocess


# ----------------- Helpers & Schema -----------------

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _host(url: str) -> str:
    try:
        return urllib.parse.urlsplit(url).hostname or ""
    except Exception:
        return ""


def _classify_source(host: str, url: str, title: str = "") -> Tuple[str, float]:
    """Classify host into a coarse type and provide a base authority score.
    Types: official, reference, academic, news, vendor, docs, community, other
    """
    h = (host or "").lower()
    t = (title or "").lower()
    base = 0.0
    stype = "other"

    # High authority TLDs
    if h.endswith(".gov") or ".gov/" in url:
        stype, base = "official", 5.0
    elif h.endswith(".edu") or ".edu/" in url:
        stype, base = "academic", 4.0

    # Well-known references and documentation portals
    elif "wikipedia.org" in h or "britannica.com" in h:
        stype, base = "reference", 4.2
    elif any(x in h for x in ("readthedocs", "docs.", "developer.", "dev.", "api.")):
        stype, base = "docs", 4.0
    elif any(x in h for x in ("arxiv.org", "acm.org", "ieee.org")):
        stype, base = "academic", 4.2
    elif any(x in h for x in ("newsroom", "press.")) or ("press" in t or "newsroom" in t):
        stype, base = "official", 4.6
    elif any(x in h for x in ("nytimes.com", "bbc.co", "reuters.com", "apnews.com", "bloomberg.com")):
        stype, base = "news", 3.8
    elif any(x in h for x in ("github.com", "gitlab.com")):
        stype, base = "community", 3.2
    elif any(x in h for x in ("reddit.com", "stackexchange.com", "stackoverflow.com")):
        stype, base = "community", 2.8

    # Vendor/product sites (e.g., apple.com, nvidia.com) treated as official
    elif any(x in h for x in ("apple.com", "microsoft.com", "google.com", "nvidia.com", "openai.com")):
        stype, base = "official", 4.2

    return stype, base


def _score_result(host: str, title: str, snippet: str, query: str, url: str = "") -> float:
    """Heuristic score: higher is better.
    Factors:
      - Domain authority (via _classify_source)
      - Query term coverage in title/snippet
      - Prefer official/reference/docs
      - Penalize low-signal/social domains
    """
    title_l = (title or "").lower()
    snip_l = (snippet or "").lower()
    q = (query or "").lower()
    stype, base = _classify_source(host, url, title)
    score = base

    # Query term coverage
    terms = [t for t in re.split(r"\W+", q) if t]
    hits = 0
    for t in terms[:8]:
        if t in title_l:
            hits += 2
        elif t in snip_l:
            hits += 1
    score += min(8, hits)

    # Prefer likely official pages (press/newsroom/docs)
    if stype in ("official", "docs"):
        score += 1.0

    # Prefer results with snippet content
    if len(snip_l) > 60:
        score += 0.6

    # Penalize low-signal image boards / social
    if any(b in host for b in ("pinterest.", "facebook.", "tiktok.", "reddit.")):
        score -= 2.5

    return float(score)


def _explain_rank(host: str, url: str, title: str, snippet: str, query: str) -> Tuple[str, str, float]:
    """Return (type, because, authority) for UI/model guidance."""
    stype, base = _classify_source(host, url, title)
    because_parts: List[str] = []
    if stype == "official":
        because_parts.append("official/source-of-record")
    elif stype == "reference":
        because_parts.append("neutral reference overview")
    elif stype == "docs":
        because_parts.append("technical documentation")
    elif stype == "academic":
        because_parts.append("academic/paper repository")
    elif stype == "news":
        because_parts.append("reputable news outlet")
    elif stype == "community":
        because_parts.append("community discussion/site")

    # Query match hint
    q = (query or "").lower()
    title_l = (title or "").lower()
    snip_l = (snippet or "").lower()
    terms = [t for t in re.split(r"\W+", q) if t]
    matches = [t for t in terms[:5] if t in title_l or t in snip_l]
    if matches:
        because_parts.append(f"matches: {', '.join(matches[:3])}")

    because = "; ".join(because_parts) or "relevant match"
    return stype, because, base


def _ok(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Wrap a payload as a success."""
    data: Dict[str, Any] = {"ok": True}
    if payload:
        data.update(payload)
    return data


def _err(msg: str, **extra) -> Dict[str, Any]:
    """Standardized error shape."""
    out = {"ok": False, "error": str(msg)}
    if extra:
        out.update(extra)
    return out


async def _retry_async(fn, *args, tries: int = 2, delay: float = 0.5, **kwargs):
    """
    Minimal async retry helper for flaky I/O.
    Retries when the result is a dict containing an error or ok=False.
    """
    last = None
    for i in range(max(1, tries)):
        res = await fn(*args, **kwargs)
        if not isinstance(res, dict):
            return res
        if res.get("ok", True) and not res.get("error"):
            return res
        last = res
        if i < tries - 1:
            await asyncio.sleep(delay)
    return last


# ----------------- Caches -----------------

# Simple in-memory caches to avoid repeating network calls
_SEARCH_CACHE: Dict[Tuple[str, int], Dict] = {}
_URL_CACHE: Dict[Tuple[str, int], Dict] = {}


# ----------------- Network: Search & Open -----------------

async def _ddg_search_html(q: str, k: int = 5) -> List[Dict[str, str]] | Dict[str, Any]:
    """
    Scrape DuckDuckGo's HTML results and return [{title,url,snippet}, ...].
    On failure, returns {"ok": False, "error": "..."}.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(
                "https://duckduckgo.com/html/",
                params={"q": q},
                headers=headers,
            )
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
    except httpx.HTTPError as e:
        return _err(f"web search failed: {e}")
    except Exception as e:
        return _err(f"web search failed: {e}")

    items = []
    for res in soup.select("div.result")[:k]:
        a = res.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            # Unwrap DDG redirect
            try:
                qs = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query)
                href = urllib.parse.unquote(qs.get("uddg", [href])[0])
            except Exception:
                pass
        snippet_el = res.select_one(".result__snippet")
        title = a.get_text(" ", strip=True)
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        items.append({"title": title, "url": href, "snippet": snippet})
    return items


async def _open_and_extract(url: str, max_chars: int = 6000) -> Dict[str, Any]:
    """
    Fetch a URL and return {'title','url','text'} (trimmed, readable).
    Returns {"ok": False, "error": "..."} on failure.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            r = await client.get(url, headers=headers)
            r.raise_for_status()
    except httpx.HTTPError as e:
        return _err(f"open url failed: {e}")
    except Exception as e:
        return _err(f"open url failed: {e}")

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()

    # Meta extraction helpers
    def meta(name: str) -> str:
        el = soup.find("meta", attrs={"name": name})
        return el.get("content", "").strip() if el else ""
    def prop(p: str) -> str:
        el = soup.find("meta", attrs={"property": p})
        return el.get("content", "").strip() if el else ""

    # Title/site name with fallbacks
    title = prop("og:title") or meta("twitter:title") or (soup.title.get_text(strip=True) if soup.title else "")
    site_name = prop("og:site_name") or _host(url)
    canonical = ""
    try:
        link_canon = soup.find("link", rel=lambda v: v and "canonical" in v)
        if link_canon:
            canonical = link_canon.get("href", "")
    except Exception:
        canonical = ""

    # Remove nav/header/footer/aside to reduce noise
    for tag in soup(["header", "footer", "nav", "aside"]):
        tag.decompose()

    # Find a plausible main content container
    main = (
        soup.find(["article", "main"]) or
        soup.find(id="content") or
        soup.find(class_=re.compile(r"(article|content|post|entry)", re.I)) or
        soup.body or soup
    )
    parts: List[str] = []
    for el in main.find_all(["h1", "h2", "h3", "p", "li"], limit=1600):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = _clean_text(" ".join(parts))
    if len(text) > max_chars:
        text = text[:max_chars] + " …"

    # Basic summary: first 2 sentences
    def first_sentences(t: str, n: int = 2) -> str:
        sents = re.split(r"(?<=[\.!?])\s+", t)
        return _clean_text(" ".join(sents[:n]))

    summary = first_sentences(text, 3)

    # Publication date/author/lang
    published = prop("article:published_time") or meta("date")
    try:
        if published:
            # Normalize to ISO date if possible
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            published = dt.date().isoformat()
    except Exception:
        # leave as-is
        pass
    author = meta("author") or prop("article:author") or meta("byline")
    lang = (soup.html.get("lang") if soup.html else "") or meta("lang")

    # Headings quick list
    headings = []
    try:
        for h in main.find_all(["h1", "h2"], limit=6):
            t = h.get_text(" ", strip=True)
            if t:
                headings.append(t)
    except Exception:
        pass

    # Word count / reading time estimate
    wc = len(text.split())
    reading_time_min = max(1, int(round(wc / 225))) if wc else 0

    # Meta description / lede
    description = prop("og:description") or meta("description")

    # Source classification
    stype, base_auth = _classify_source(_host(url), url, title)

    page = {
        "title": title,
        "url": url,
        "canonical_url": canonical or url,
        "site_name": site_name,
        "host": _host(url),
        "lang": lang,
        "text": text,
        "summary": summary,
        "description": description or None,
        "word_count": wc,
        "reading_time_min": reading_time_min,
        "published": published or None,
        "author": author or None,
        "headings": headings,
        "type": stype,
        "authority_hint": base_auth,
    }

    return _ok({"page": page})


async def open_related_links(url: str, query: str, k: int = 5) -> Dict[str, Any]:
    """
    From a given page, find related links and return top-K previews ranked by relevance.
    Result shape:
      {"ok": True, "related": [{"title","url","host","summary"}, ...]}
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:
        return _err(f"open related failed: {e}")

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "form"]):
        tag.decompose()

    base_host = _host(url)
    cands: List[Dict[str, Any]] = []
    for a in soup.find_all("a", href=True, limit=400):
        href = a.get("href", "").strip()
        if not href:
            continue
        # Normalize href
        abs_url = urllib.parse.urljoin(url, href)
        if not (abs_url.startswith("http://") or abs_url.startswith("https://")):
            continue
        host = _host(abs_url)
        # Prefer same-host first
        same_site_bonus = 1.0 if host == base_host else 0.0
        text = a.get_text(" ", strip=True) or ""
        title = a.get("title", "").strip() or text
        if not title:
            continue
        score = _score_result(host, title, text, query) + same_site_bonus
        cands.append({"url": abs_url, "host": host, "title": title, "score": score})

    # Dedup by URL and host/title combo
    seen_urls: set[str] = set()
    dedup: List[Dict[str, Any]] = []
    for c in sorted(cands, key=lambda x: x["score"], reverse=True):
        keyu = c["url"]
        keyt = (c["host"], c["title"])
        if keyu in seen_urls:
            continue
        seen_urls.add(keyu)
        dedup.append(c)
        if len(dedup) >= k * 3:
            break

    # Fetch small previews for top-K after dedup
    related: List[Dict[str, Any]] = []
    for cand in dedup:
        if len(related) >= k:
            break
        try:
            prev = await _retry_async(_open_and_extract, cand["url"], 800, tries=1, delay=0.0)
            if isinstance(prev, dict) and prev.get("ok") and isinstance(prev.get("page"), dict):
                p = prev["page"]
                related.append({
                    "title": p.get("title") or cand.get("title"),
                    "url": p.get("url") or cand.get("url"),
                    "host": cand.get("host"),
                    "summary": p.get("summary") or (p.get("text") or "")[:320]
                })
        except Exception:
            continue

    return _ok({"related": related, "source": url, "query": query})


async def web_search(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Search the web and return a normalized payload:
      Success: {"ok": True, "results": [...], "source": "duckduckgo_html"}
      Failure: {"ok": False, "error": "..."}
    """
    key = (query, k)
    if key not in _SEARCH_CACHE:
        try:
            raw = await _retry_async(_ddg_search_html, query, k, tries=2, delay=0.4)
            if isinstance(raw, dict) and (raw.get("ok") is False or raw.get("error")):
                return raw
            # Normalize -> score -> dedupe by host -> slice
            norm: List[Dict[str, Any]] = []
            seen_hosts: set[str] = set()
            for i, r in enumerate(raw):
                url = r.get("url", "")
                if not (url.startswith("http://") or url.startswith("https://")):
                    continue
                host = _host(url)
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                score = _score_result(host, title, snippet, query, url=url)
                # de-dup by host, prefer first higher score implicitly by ordering later
                if host in seen_hosts:
                    continue
                seen_hosts.add(host)
                stype, because, auth = _explain_rank(host, url, title, snippet, query)
                norm.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "host": host,
                    "score": float(score),
                    "rank": i + 1,
                    "type": stype,
                    "authority_hint": auth,
                    "because": because,
                })
            norm.sort(key=lambda x: x.get("score", 0), reverse=True)
            # Opportunistic previews for the top result(s) to save extra round-trips
            previews: List[Dict[str, Any]] = []
            try:
                preview_count = min(2, k)
                for cand in norm[: min(preview_count, len(norm)) ]:
                    opened = await _retry_async(_open_and_extract, cand["url"], 800, tries=1, delay=0.0)
                    if isinstance(opened, dict) and opened.get("ok") and isinstance(opened.get("page"), dict):
                        p = opened["page"]
                        previews.append({
                            "url": p.get("url"),
                            "title": p.get("title"),
                            "host": cand.get("host"),
                            "summary": p.get("summary"),
                            "site_name": p.get("site_name"),
                            "published": p.get("published"),
                            "reading_time_min": p.get("reading_time_min"),
                            "type": p.get("type"),
                            "headings": p.get("headings", [])[:4],
                        })
            except Exception:
                pass

            # Recommend best opens with reasons for the model
            recs = []
            for cand in norm[: min(3, len(norm)) ]:
                recs.append({
                    "url": cand["url"],
                    "title": cand["title"],
                    "host": cand["host"],
                    "type": cand.get("type"),
                    "because": cand.get("because"),
                })

            # Suggest query refinements to guide follow-ups
            ql = (query or "").strip()
            query_hints: List[str] = []
            if len(ql.split()) <= 3:
                query_hints.append(f"{ql} official site")
                query_hints.append(f"{ql} site:wikipedia.org summary")
            if any(tok in ql.lower() for tok in ("release date", "specs", "announcement")):
                query_hints.append(f"{ql} site:newsroom.* OR site:press.*")

            payload = {
                "results": norm[:k],
                "source": "duckduckgo_html",
                "query": query,
                "previews": previews,
                "recommended_open": recs,
                "query_hints": query_hints,
            }
            _SEARCH_CACHE[key] = _ok(payload)
        except httpx.HTTPError as e:
            return _err(f"web search failed: {e}")
        except Exception as e:
            return _err(f"web search failed: {e}")
    return _SEARCH_CACHE[key]


async def open_url(url: str, max_chars: int = 6000) -> Dict[str, Any]:
    """
    Open a URL and return:
      Success: {"ok": True, "page": {"title","url","text"}}
      Failure: {"ok": False, "error": "..."}
    """
    key = (url, max_chars)
    if key not in _URL_CACHE:
        page = await _retry_async(_open_and_extract, url, max_chars, tries=2, delay=0.4)
        if isinstance(page, dict) and page.get("ok") is False:
            return page
        if isinstance(page, dict) and page.get("error"):
            return page
        _URL_CACHE[key] = _ok({"page": page})
    return _URL_CACHE[key]


async def search_docs(query: str, k: int = 6, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Search the local docstore (hybrid: semantic + keyword). Returns:
      {"ok": True, "results": [{doc_id,title,source,uri,preview,score}, ...]}
    Note: The server wire-up handles this tool directly with its shared httpx client.
    This function exists for completeness when used outside the server loop.
    """
    try:
        from docstore import get_store  # local import to avoid cycles
        async with httpx.AsyncClient(timeout=30.0) as client:
            store = get_store()
            return await store.hybrid_search(client, query=query, k=k, filters=filters or {})
    except Exception as e:
        return _err(str(e))


# ----------------- Local utility tools -----------------

async def eval_expr(expr: str) -> Dict[str, Any]:
    """
    Evaluate a Python expression and return:
      Success: {"ok": True, "result": "<repr>"}
      Failure: {"ok": False, "error": "..."}
    """
    try:
        # Evaluate with no builtins for a bit of safety
        result = eval(expr, {"__builtins__": {}})
        return _ok({"result": repr(result)})
    except Exception as e:
        return _err(str(e))


async def execute(code: str) -> Dict[str, Any]:
    """
    Execute a Python code snippet and capture stdout/stderr.
      Success (rc==0): {"ok": True, "stdout": "...", "stderr": "...", "returncode": 0}
      Failure (rc!=0): {"ok": False, "error": "python exited with <rc>", "stdout": "...", "stderr": "...", "returncode": <rc>}
      Failure (exception): {"ok": False, "error": "..."}
    """
    try:
        proc = subprocess.run(
            ["python", "-"],
            input=code,
            text=True,
            capture_output=True,
            timeout=10,
        )
        payload = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
        if proc.returncode != 0:
            return _err(f"python exited with {proc.returncode}", **payload)
        return _ok(payload)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}")


async def read_file(path: str) -> Dict[str, Any]:
    """
    Read a text file.
      Success: {"ok": True, "content": "..."}
      Failure: {"ok": False, "error": "..."}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _ok({"content": f.read()})
    except Exception as e:
        return _err(str(e))


async def write_file(path: str, contents: str) -> Dict[str, Any]:
    """
    Write text to a file.
      Success: {"ok": True}
      Failure: {"ok": False, "error": "..."}
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(contents)
        return _ok()
    except Exception as e:
        return _err(str(e))


# ----------------- Pseudo-terminal -----------------

# Terminal session state (single session for simplicity)
_TERMINAL_OPEN = False


async def terminal_open() -> Dict[str, Any]:
    """
    Open a pseudo terminal session.
      Success: {"ok": True, "already_open": bool}
    """
    global _TERMINAL_OPEN
    already = _TERMINAL_OPEN
    _TERMINAL_OPEN = True
    return _ok({"already_open": already})


async def terminal_run(cmd: str) -> Dict[str, Any]:
    """
    Run a shell command in the terminal session.
      Success (rc==0): {"ok": True, "stdout": "...", "stderr": "...", "returncode": 0, "cmd": "..."}
      Failure (not open): {"ok": False, "error": "terminal not open"}
      Failure (rc!=0): {"ok": False, "error": "shell exited with <rc>", "stdout": "...", "stderr": "...", "returncode": <rc>, "cmd": "..."}
    """
    if not _TERMINAL_OPEN:
        return _err("terminal not open")

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        payload = {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "cmd": cmd,
        }
        if proc.returncode != 0:
            return _err(f"shell exited with {proc.returncode}", **payload)
        return _ok(payload)
    except Exception as e:
        return _err(f"{type(e).__name__}: {e}", cmd=cmd)


async def terminal_terminate() -> Dict[str, Any]:
    """
    Terminate the pseudo terminal session.
      Success: {"ok": True, "was_open": bool}
    """
    global _TERMINAL_OPEN
    was = _TERMINAL_OPEN
    _TERMINAL_OPEN = False
    return _ok({"was_open": was})


# ----------------- Notes & User Prefs -----------------

_NOTES: Dict[str, str] = {}
_USER_PREFS: Dict[str, str] = {}


async def notes_write(key: str, content: str) -> Dict[str, Any]:
    _NOTES[key] = content
    return _ok()


async def notes_list() -> Dict[str, Any]:
    return _ok({"keys": list(_NOTES.keys())})


async def notes_read(key: str) -> Dict[str, Any]:
    if key in _NOTES:
        return _ok({"content": _NOTES[key]})
    return _err("not found")


async def user_prefs_write(key: str, content: str) -> Dict[str, Any]:
    _USER_PREFS[key] = content
    return _ok()


async def user_prefs_list() -> Dict[str, Any]:
    return _ok({"keys": list(_USER_PREFS.keys())})


async def user_prefs_read(key: str) -> Dict[str, Any]:
    if key in _USER_PREFS:
        return _ok({"content": _USER_PREFS[key]})
    return _err("not found")
