import re

from summarizer import summarize_cot


STATUS_LABEL_RE = re.compile(r"^[A-Za-z]+ing( [A-Za-z0-9\-]{1,20}){1,4}$")


def _wc(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s))


def _title_ok(t: str) -> bool:
    # 6–8 words and present-tense/progressive verb heuristic
    words = t.split()
    if not (6 <= len(words) <= 8):
        return False
    return words[0].endswith("ing") or words[0].lower() in {
        "checks",
        "confirms",
        "verifies",
        "assesses",
        "reviews",
        "resolves",
    }


def _status_ok(label: str) -> bool:
    return 2 <= len(label.split()) <= 5 and STATUS_LABEL_RE.match(label) is not None and "http" not in label.lower()


def _user_summary_ok(text: str, min_s: int = 2, max_s: int = 4) -> bool:
    sents = [seg for seg in re.split(r"(?<=[\.!?])\s+", text.strip()) if seg]
    return min_s <= len(sents) <= max_s and "http" not in text.lower()


def test_confirmed_case():
    text = (
        "Found an official blog post that lists specs and a press release—"
        "this confirms the announcement."
    )
    out = summarize_cot(text)
    assert isinstance(out, dict)
    assert "title" in out and "status" in out and "user_summary" in out and "confidence" in out
    assert _title_ok(out["title"]) is True
    assert _status_ok(out["status"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] == "confirmed"


def test_speculative_case_blog_only():
    text = (
        "An unverified blog mentions the feature; no bylines or dates; seeking corroboration."
    )
    out = summarize_cot(text)
    assert _status_ok(out["status"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] == "speculative"


def test_conflicting_case():
    text = "One text says Sept 9; another says Oct 1; conflicting claims."
    out = summarize_cot(text)
    assert _status_ok(out["status"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] == "conflicting"


def test_network_error_case():
    text = "Timeouts and rate limits; repeated 504s disrupt checks."
    out = summarize_cot(text)
    assert _status_ok(out["status"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] == "network_error"


def test_search_event_only_lines():
    text = (
        "web_search\n"
        "searched for iPhone 17 Pro specs 2025\n"
        "Preview (www.example.com/page)\n"
        "open_url\n"
        "Reviewing support.apple.com\n"
        "Investigating iPhone 17 Pro specs 2025\n"
    )
    out = summarize_cot(text)
    assert _status_ok(out["status"]) is True
    # Either with or without entity tail
    assert out["status"].startswith("Finding more info")
    assert _title_ok(out["title"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] in {"speculative", "partial"}
    # Ensure no tool tokens or URLs leak
    blob = " ".join([out.get("title", ""), out.get("status", ""), out.get("status_detail", "") or "", out.get("user_summary", "")])
    low = blob.lower()
    assert all(tok not in low for tok in ["web_search", "open_url", "preview (", "http", "www."])


def test_partial_confirmation():
    text = (
        "Coverage aligns, but primary details remain missing; partial confirmation only for the specs."
    )
    out = summarize_cot(text)
    assert _status_ok(out["status"]) is True
    assert _user_summary_ok(out["user_summary"]) is True
    assert out["confidence"] == "partial"


def test_filler_rejection():
    text = "No progress on that yet—still checking sources."
    out = summarize_cot(text)
    blob = " ".join([out.get("title", ""), out.get("status_detail", "") or "", out.get("user_summary", "")])
    low = blob.lower()
    assert "no progress" not in low and "still checking" not in low
    assert _status_ok(out["status"]) is True


def test_press_release_date_multiple_outlets_confirmed():
    text = (
        "A press release mentions the announcement and includes a date Sept 9, 2025. "
        "Multiple outlets echo the details; this confirms the claim."
    )
    out = summarize_cot(text)
    assert out["confidence"] == "confirmed"
    assert out["status"].startswith("Summarizing findings")
    assert _user_summary_ok(out["user_summary"]) is True
