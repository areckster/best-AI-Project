"""
Summarizer: Convert a reasoning-model CoT chunk into compact UI updates.

Pipeline (deterministic, no network/LLM required):
  1) preprocess_cot: strip tool markup, logs, timestamps, URLs; redact PII.
  2) analyze_text: extract load‑bearing cues and classify state
     into: confirmed | conflicting | only_speculative | none_found | partial | network_error | ongoing.
  3) Generate fields:
       - title (6–8 words, present-tense)
       - status (2–5 word gerund label, no sources/URLs)
       - status_detail (optional 1 short sentence ≤18 words)
       - user_summary (2–4 sentences, generic; no sources/URLs)
       - confidence (confirmed|partial|speculative|conflicting|network_error)
  4) validate_and_fix: enforce all constraints deterministically.

This module is intentionally rule-based for low latency and determinism.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


BANNED_PHRASES = {
    "still checking",
    "no progress",
    "working on it",
    "i think",
    "maybe",
    "i’ll keep digging",
    "still gathering evidence",
    "no progress on that yet",
    # Allowed only as STATUS label, not as a sentence
    "checking sources",
}


def _strip_urls(text: str) -> str:
    # Remove URLs (http, https, www) and bare domains with TLDs
    text = re.sub(r"https?://\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"www\.[^\s]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[a-z0-9.-]+\.(?:com|org|net|io|ai|co|gov|edu)\b\S*", " ", text, flags=re.IGNORECASE)
    return text


def _redact_pii(text: str) -> str:
    # Emails
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED]", text)
    # Phone numbers (simple patterns)
    text = re.sub(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}\b", "[REDACTED]", text)
    return text


def _sanitize_entity(ent: Optional[str]) -> Optional[str]:
    if not ent:
        return None
    # Remove URLs/domains and punctuation; keep ≤3 tokens alnum/hyphen
    ent = re.sub(r"https?://\S+|www\.[^\s]+", " ", ent)
    ent = re.sub(r"[^A-Za-z0-9\-\s]", " ", ent)
    tokens = [t for t in ent.split() if t]
    # Filter out obvious non-entity tokens
    stop = {"specs", "review", "reviews", "release", "date", "dates", "rumor", "leak", "news", "site", "homepage", "domain", "official"}
    clean = [t for t in tokens if t.lower() not in stop and "." not in t and "/" not in t]
    clean = clean[:3]
    if not clean:
        return None
    return " ".join(clean)


def _entity_from_query(q: str) -> Optional[str]:
    # Focus on early tokens before stopwords like specs/release/date/etc.
    q = q.strip()
    q = re.sub(r"\s+", " ", q)
    # Remove parentheses blocks (e.g., Preview (...))
    q = re.sub(r"\([^)]*\)", " ", q)
    tokens = [t for t in q.split() if t]
    stop = {"specs", "spec", "release", "date", "dates", "rumor", "leak", "leaks", "news", "review", "reviews", "about", "on", "for"}
    picked: List[str] = []
    for t in tokens:
        tl = t.lower()
        if tl in stop:
            break
        if tl.startswith("site:"):
            break
        if tl.startswith("http") or "." in tl:
            break
        picked.append(t)
        if len(picked) >= 4:
            break
    ent = " ".join(picked)
    return _sanitize_entity(ent)


def preprocess_cot(cot_text: str) -> Tuple[str, Optional[str], bool]:
    s = cot_text or ""
    # Drop code fences / blocks
    s = re.sub(r"```.+?```", " ", s, flags=re.DOTALL)
    # Drop XML-like think/reasoning/tool tags
    s = re.sub(r"<\s*(think|reasoning|tool)[^>]*>.*?<\s*/\s*\1\s*>", " ", s, flags=re.DOTALL | re.IGNORECASE)
    # Remove Preview(...) inline blocks entirely
    s = re.sub(r"Preview\s*\([^)]*\)", " ", s, flags=re.IGNORECASE)

    # Scan lines for entity candidates before stripping tool lines
    entity_hint: Optional[str] = None
    human_lines: List[str] = []
    seen: set[str] = set()
    for line in s.splitlines():
        l = line.strip()
        if not l:
            continue
        # Tool/log prefixes to remove
        if re.match(r"^(TOOL|LOG|DEBUG|TRACE|INFO|WARN|ERROR|CMD|RUN|STDERR|STDOUT|EVENT)[: ]", l, flags=re.IGNORECASE):
            continue
        if re.match(r"^\d{1,2}:\d{2}(:\d{2})?", l):
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", l):
            continue
        if "NOTE TO REVIEWER" in l.upper() or "REVIEWER'S NOTE" in l.upper():
            continue
        if re.match(r"^(web_search|open_url)\b", l, flags=re.IGNORECASE):
            continue
        # Capture entity from patterns then strip the line
        m = re.match(r"^searched for\s+(.+)$", l, flags=re.IGNORECASE)
        if m:
            entity_hint = entity_hint or _entity_from_query(m.group(1))
            continue
        m = re.match(r"^(Investigating|Reviewing)\s+(.+)$", l, flags=re.IGNORECASE)
        if m:
            entity_hint = entity_hint or _entity_from_query(m.group(2))
            continue
        if l.lower().startswith("preview ("):
            continue
        if "tool_call" in l or "function_call" in l:
            continue
        # Keep human-readable line, deduplicated
        l = _strip_urls(l)
        l = re.sub(r"\s+", " ", l).strip()
        if not l:
            continue
        if l in seen:
            continue
        seen.add(l)
        human_lines.append(l)
    s = " ".join(human_lines)
    s = _strip_urls(s)
    s = _redact_pii(s)
    # Collapse whitespace and keep only readable sentences-ish
    s = re.sub(r"\s+", " ", s).strip()
    search_event_only = (len(human_lines) == 0)
    return s, _sanitize_entity(entity_hint), search_event_only


def _split_sentences(text: str) -> List[str]:
    # Basic sentence splitter; honors ., !, ? and semicolons as soft breaks
    parts = re.split(r"(?<=[\.!?])\s+|;\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 3:
            continue
        # Keep sentences with letters
        if re.search(r"[A-Za-z]", p):
            out.append(p)
    return out


def _detect_source_hint(text: str) -> Optional[str]:
    # Generic hints only; no publisher names.
    t = text.lower()
    if "press release" in t:
        return "press"
    if "official" in t:
        return "official"
    if "blog" in t:
        return "blog"
    if "forum" in t or "discussion" in t:
        return "forum"
    if "news" in t:
        return "news"
    return None


def _extract_entity(sentences: List[str]) -> Optional[str]:
    # Try to find a capitalized entity span (≤3 tokens), ignoring filler words.
    ignore = {"I", "We", "Network", "One", "Another", "Press", "Blog", "Forum", "Site"}
    for s in sentences:
        # Allow hyphenated tokens
        m = re.search(r"\b([A-Z][A-Za-z0-9\-]+(?: [A-Z][A-Za-z0-9\-]+){0,3})\b", s)
        if m:
            ent = m.group(1).strip()
            if ent in ignore:
                continue
            # Truncate to ≤3 tokens
            tokens = [t for t in ent.split() if t]
            ent3 = " ".join(tokens[:3])
            # Remove punctuation at ends
            ent3 = re.sub(r"^[\W_]+|[\W_]+$", "", ent3)
            if ent3:
                return ent3
        if re.search(r"\bX\b", s):
            return "X"
    return None


def _classify(sentences: List[str]) -> Tuple[str, Optional[str], Optional[str]]:
    """Return (category, source_hint, evidence_snippet)."""
    text = " ".join(sentences).lower()

    # Network issues
    if re.search(r"network (error|timeout|fail|unavailable)|http \d{3}|rate limit|429|504", text):
        return "network_error", None, None

    # Conflicts
    if any(k in text for k in ("conflict", "contradict", "disagree")):
        return "conflicting", None, None

    # Confirmed
    if ("confirm" in text or "verified" in text) and any(k in text for k in ("official", "press", "blog", "site")):
        return "confirmed", _detect_source_hint(text), None
    # Press-release + date + outlets pattern → confirmed
    if "press release" in text:
        # simple date cues: month names or YYYY
        has_date = bool(re.search(r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b", text)) or bool(re.search(r"\b20\d{2}\b", text))
        has_outlets = ("multiple" in text and "outlet" in text) or ("several" in text and "outlet" in text) or ("echo" in text)
        if has_date and has_outlets:
            return "confirmed", "press", None

    # Only speculative / weak sources
    if any(k in text for k in ("unverified", "speculative", "rumor", "leak", "forum", "blog")):
        return "only_speculative", _detect_source_hint(text), None

    # None found / still searching
    if any(k in text for k in ("cannot find", "no results", "nothing found", "not find", "has nothing", "need to search", "search more")):
        return "none_found", None, None

    # Partial confirmation
    if "partial" in text or ("some" in text and "not" in text and "confirm" in text):
        return "partial", _detect_source_hint(text), None

    # Default ongoing
    return "ongoing", _detect_source_hint(text), None


def _title_for(category: str, entity: Optional[str], source_hint: Optional[str]) -> str:
    ent = (entity or "findings").strip()
    if category == "confirmed":
        return "Confirming announcement via credible sources today"
    if category == "conflicting":
        if entity:
            return f"Resolving conflicting reports on {ent} today"
        return "Resolving conflicting reports across sources today"
    if category == "only_speculative":
        return "Assessing speculative reporting and seeking verification"
    if category == "none_found":
        return "Searching additional reliable sources for confirmation"
    if category == "partial":
        return "Verifying partial claims across credible sources"
    if category == "network_error":
        return "Recovering from network errors retrying sources"
    # ongoing
    return "Reviewing current findings and next verification steps"


def _status_label_for(category: str, entity: Optional[str]) -> str:
    # Deterministic first-choice mapping
    base = {
        "network_error": "Retrying fetch",
        "conflicting": "Resolving conflicts",
        "partial": "Finding more info",
        "only_speculative": "Finding more info",
        "none_found": "Finding more info",
        "confirmed": "Summarizing findings",
        "ongoing": "Checking sources",
    }.get(category, "Checking sources")

    # Append a short entity tail (≤3 tokens) if present; strip punctuation
    tail = None
    base_tokens = base.split()
    if entity:
        t = re.sub(r"[\W_]+", " ", entity).strip()
        tokens = [tok for tok in t.split() if tok]
        # Trim tail so total words ≤5
        room = max(0, 5 - len(base_tokens))
        if room > 0 and tokens:
            tail = " ".join(tokens[: min(3, room)])

    label = base if not tail else f"{base} {tail}"
    # Validate later with regex; return for now
    return label


def _status_detail_for(category: str) -> Optional[str]:
    # One short sentence (≤18 words), generic.
    if category == "network_error":
        return "Retrying requests and pivoting approaches to restore checks."
    if category == "conflicting":
        return "Comparing claims using timestamps, wording, and corroboration."
    if category in ("partial", "only_speculative", "none_found"):
        return "Continuing checks using general credibility signals and corroboration."
    if category == "confirmed":
        return "Summarizing verified details for the final answer."
    return None


def _word_count(s: str) -> int:
    return len([w for w in re.findall(r"\b\w+\b", s)])


STATUS_LABEL_RE = re.compile(r"^[A-Za-z]+ing( [A-Za-z0-9\-]{1,20}){1,4}$")


def _valid_status_label(label: str) -> bool:
    if not (2 <= len(label.split()) <= 5):
        return False
    if not STATUS_LABEL_RE.match(label):
        return False
    if any(ch in label for ch in ",.;:!?"):
        return False
    if re.search(r"https?://|www\.", label, flags=re.IGNORECASE):
        return False
    # No tool-y tokens
    low = label.lower()
    for tok in ("tool", "function", "call", "open_url", "web_search", "http", "https"):
        if tok in low:
            return False
    return True


def validate_and_fix(title: str, status_label: str, status_detail: Optional[str], user_summary: str, confidence: str) -> Tuple[str, str, Optional[str], str, str]:
    # Title: 6–8 words. If not, deterministically pad/trim.
    def fix_title(t: str) -> str:
        words = t.split()
        if 6 <= len(words) <= 8:
            return t
        # If too short, pad with deterministic tokens based on hashless pool
        fillers = ["today", "now", "carefully", "thoroughly", "reliably"]
        i = 0
        while len(words) < 6 and i < len(fillers):
            words.append(fillers[i])
            i += 1
        # If too long, trim from the end
        if len(words) > 8:
            words = words[:8]
        return " ".join(words)

    def fix_status_label(lbl: str) -> str:
        # Remove punctuation; compress spaces
        lbl = re.sub(r"[\.,;:!?]", "", lbl).strip()
        lbl = re.sub(r"\s+", " ", lbl)
        # Enforce mapping tokens (2–5 words) and regex
        if not _valid_status_label(lbl):
            return "Checking sources"
        return lbl

    def fix_status_detail(sd: Optional[str]) -> Optional[str]:
        if not sd:
            return None
        s = _strip_urls(sd)
        s = re.sub(r"[\r\n]+", " ", s).strip()
        s = re.sub(r"\s+", " ", s).strip()
        if _word_count(s) > 18:
            words = s.split()[:18]
            s = " ".join(words).rstrip(".,;") + "."
        return s or None

    def fix_user_summary(us: str, conf: str) -> str:
        s = _strip_urls(us)
        s = re.sub(r"\s+", " ", s).strip()
        # Ensure 2–4 sentences; if not, compress/expand deterministically
        sents = [seg.strip() for seg in re.split(r"(?<=[\.!?])\s+", s) if seg.strip()]
        if len(sents) < 2:
            # Append a deterministic tail according to confidence
            tails = {
                "confirmed": "Details are being summarized for clarity.",
                "conflicting": "We are comparing timestamps and wording for consistency.",
                "partial": "Remaining items will be verified before concluding.",
                "speculative": "We are seeking corroboration from independent materials.",
                "network_error": "We are retrying and switching approaches.",
            }
            sents.append(tails.get(conf, "We will continue validating details."))
        if len(sents) > 4:
            sents = sents[:4]
        return " ".join(sents)

    # Remove banned phrases from detail/summary, never from status label unless it invalidates the regex
    def _remove_banned(text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        out = text
        low = out.lower()
        for b in BANNED_PHRASES:
            if b in low:
                out = re.sub(re.escape(b), "", out, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", out).strip() or None

    title = fix_title(title)
    status_label = fix_status_label(status_label)
    status_detail = fix_status_detail(_remove_banned(status_detail))
    user_summary = fix_user_summary(_remove_banned(user_summary) or "", confidence)
    # Final regex guard for label
    if not _valid_status_label(status_label):
        status_label = "Checking sources"
    return title, status_label, status_detail, user_summary, confidence


def _confidence_from_category(category: str) -> str:
    return {
        "confirmed": "confirmed",
        "partial": "partial",
        "only_speculative": "speculative",
        "none_found": "speculative",
        "conflicting": "conflicting",
        "network_error": "network_error",
        "ongoing": "speculative",
    }.get(category, "speculative")


def _user_summary_for(category: str) -> str:
    if category == "confirmed":
        return (
            "We confirmed the core claim using credible materials. "
            "Details are being summarized for clarity."
        )
    if category == "conflicting":
        return (
            "Reports disagree on key details. "
            "We are comparing timestamps and wording for consistency."
        )
    if category in ("partial",):
        return (
            "Some details are consistent while others remain unverified. "
            "We will verify remaining facts before concluding."
        )
    if category in ("only_speculative", "none_found"):
        return (
            "Evidence appears limited or speculative at this stage. "
            "We are seeking corroboration from independent, authoritative materials."
        )
    if category == "network_error":
        return (
            "Network issues interrupted checks. "
            "We are retrying and switching approaches."
        )
    # ongoing
    return (
        "Review is in progress. "
        "We will continue validating details."
    )


def summarize_cot(cot_text: str) -> Dict[str, str]:
    """Convert raw CoT to compact UI JSON per spec.

    Returns keys: title, status, status_detail (optional), user_summary, confidence.
    """
    cleaned, entity_hint, search_event_only = preprocess_cot(cot_text or "")
    sentences = _split_sentences(cleaned)
    if search_event_only and not sentences:
        category = "only_speculative"
        entity = entity_hint
    else:
        if not sentences:
            category = "ongoing"
        else:
            category, source_hint, _ = _classify(sentences)
        entity = _extract_entity(sentences) or entity_hint

    confidence = _confidence_from_category(category)
    title = _title_for(category, entity, None)
    status_label = _status_label_for(category, entity)
    status_detail = _status_detail_for(category)
    user_summary = _user_summary_for(category)

    # Consistency check: forbid contradictions like confirmed + not confirmed
    us_low = user_summary.lower()
    if confidence == "confirmed" and any(k in us_low for k in ("not confirmed", "unverified", "speculative")):
        category = "partial"
        confidence = _confidence_from_category(category)
        title = _title_for(category, entity, None)
        status_label = _status_label_for(category, entity)
        status_detail = _status_detail_for(category)
        user_summary = _user_summary_for(category)

    title, status_label, status_detail, user_summary, confidence = validate_and_fix(
        title, status_label, status_detail, user_summary, confidence
    )
    out: Dict[str, str] = {
        "title": title,
        "status": status_label,
        "user_summary": user_summary,
        "confidence": confidence,
    }
    if status_detail:
        out["status_detail"] = status_detail
    return out


__all__ = [
    "summarize_cot",
    "preprocess_cot",
    "validate_and_fix",
]
