import os
import io
import json
import math
import re
import sqlite3
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from urllib.parse import urlsplit
import httpx

try:
    import faiss  # type: ignore
except Exception as _e:  # pragma: no cover
    faiss = None  # Allow import; raise at runtime if used without dep


def _app_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DOCSTORE_PATH = os.getenv("DOCSTORE_DB", os.path.join(_app_dir(), "docstore.db"))


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _as_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps(str(obj))


class DocStore:
    """
    SQLite + FTS5 metadata + in-memory FAISS vectors.
    - Metadata persists in SQLite (documents, chunks, FTS index, vectors as blobs)
    - FAISS index held in memory; rebuilt on startup or after ingestion
    - Embeddings via Ollama /api/embeddings (model configurable by EMBED_MODEL)
    """

    def __init__(self, db_path: str = DOCSTORE_PATH, embed_model: str = EMBED_MODEL, ollama_host: str = OLLAMA_HOST):
        self.db_path = db_path
        self.embed_model = embed_model
        self.ollama = ollama_host.rstrip("/")
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._dim: Optional[int] = None
        self._index = None  # type: ignore
        self._ensure_schema()
        self._rebuild_faiss()

    # ——— SQLite schema
    def _ensure_schema(self) -> None:
        c = self._conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              doc_id TEXT PRIMARY KEY,
              uri TEXT,
              source TEXT,
              title TEXT,
              tags TEXT,
              meta TEXT,
              updated_at TEXT
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              doc_id TEXT,
              idx INTEGER,
              text TEXT,
              uri TEXT,
              source TEXT,
              title TEXT,
              tags TEXT,
              vector BLOB
            )
            """
        )
        # FTS5 contentless index backed by chunks table
        c.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
              text, content='chunks', content_rowid='id'
            )
            """
        )
        # Triggers to sync FTS with chunks
        c.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
              INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        c.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;
            """
        )
        c.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
              INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
              INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        self._conn.commit()

    # ——— FAISS index helpers
    def _empty_index(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError("faiss-cpu is not installed")
        # Cosine similarity via normalized vectors + inner product
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._dim = dim

    def _rebuild_faiss(self) -> None:
        c = self._conn.cursor()
        c.execute("SELECT id, vector FROM chunks WHERE vector IS NOT NULL ORDER BY id")
        rows = c.fetchall()
        if not rows:
            self._index = None
            self._dim = None
            return
        vecs: List[np.ndarray] = []
        ids: List[int] = []
        dim: Optional[int] = None
        for r in rows:
            b = r["vector"]
            if b is None:
                continue
            arr = np.frombuffer(b, dtype=np.float32)
            if dim is None:
                dim = int(arr.shape[0])
            vecs.append(arr)
            ids.append(int(r["id"]))
        if dim is None or not vecs:
            self._index = None
            self._dim = None
            return
        mat = np.vstack([v for v in vecs]).astype(np.float32)
        mat = _norm(mat)
        self._empty_index(dim)
        self._index.add_with_ids(mat, np.array(ids, dtype=np.int64))

    # ——— Chunking
    @staticmethod
    def _chunk_text(text: str, target_tokens: int = 350, overlap_tokens: int = 60) -> List[str]:
        if not text:
            return []
        # Approximate 4 chars per token
        size = max(200, target_tokens * 4)
        overlap = max(0, overlap_tokens * 4)
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            end = min(n, i + size)
            seg = text[i:end]
            # try to end on a sentence/paragraph boundary for readability
            cut = max(seg.rfind("\n\n"), seg.rfind(". "), seg.rfind(".\n"))
            if cut >= 0 and (end - i) > 180:
                seg = seg[:cut + 1]
                end = i + len(seg)
            chunks.append(seg.strip())
            if end >= n:
                break
            i = max(end - overlap, i + 1)
        return [c for c in chunks if c]

    # ——— Embeddings
    async def _embed_batch(self, client: httpx.AsyncClient, texts: List[str]) -> List[np.ndarray]:
        """Embed a list with limited concurrency; preserve order."""
        if not texts:
            return []
        sem = asyncio.Semaphore(4)
        out: List[Optional[np.ndarray]] = [None] * len(texts)  # type: ignore

        async def _one(i: int, t: str) -> None:
            payload = {"model": self.embed_model, "prompt": t}
            async with sem:
                r = await client.post(f"{self.ollama}/api/embeddings", json=payload, timeout=60.0)
            if r.status_code >= 400:
                detail = r.text
                raise RuntimeError(f"embed failed: {r.status_code}: {detail}")
            j = r.json()
            emb = j.get("embedding") or (j.get("data", [{}])[0].get("embedding") if isinstance(j.get("data"), list) else None)
            if not emb:
                raise RuntimeError("embedding missing from response")
            out[i] = np.array(emb, dtype=np.float32)

        await asyncio.gather(*[_one(i, t) for i, t in enumerate(texts)])
        return [v for v in out if v is not None]

    # ——— Ingestion
    async def ingest(
        self,
        client: httpx.AsyncClient,
        *,
        doc_id: str,
        text: str,
        source: str,
        uri: Optional[str] = None,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Chunk → embed → persist → rebuild FAISS (simple + safe)."""
        if not text or not doc_id:
            return {"ok": False, "error": "missing text or doc_id"}
        chunks = self._chunk_text(text)
        max_chunks = int(os.getenv("DOCSTORE_MAX_CHUNKS", "500"))
        truncated = False
        if len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            truncated = True
        embs = await self._embed_batch(client, chunks) if chunks else []

        if embs:
            dim = int(embs[0].shape[0])
            if self._dim is None:
                # Initialize new index
                self._empty_index(dim)

        tag_json = _as_json(tags or [])
        meta_json = _as_json(meta or {})
        with self._conn:
            # Upsert document
            self._conn.execute(
                """
                INSERT INTO documents(doc_id, uri, source, title, tags, meta, updated_at)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(doc_id) DO UPDATE SET uri=excluded.uri, source=excluded.source,
                  title=excluded.title, tags=excluded.tags, meta=excluded.meta, updated_at=excluded.updated_at
                """,
                (doc_id, uri or "", source, title or "", tag_json, meta_json, _now_iso()),
            )
            # Remove prior chunks for this doc_id
            self._conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            # Insert new chunks
            for i, (seg, vec) in enumerate(zip(chunks, embs)):
                vb = np.asarray(vec, dtype=np.float32).tobytes()
                self._conn.execute(
                    """
                    INSERT INTO chunks(doc_id, idx, text, uri, source, title, tags, vector)
                    VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (doc_id, i, seg, uri or "", source, title or "", tag_json, vb),
                )
        # Rebuild FAISS once per ingest (simple + consistent)
        self._rebuild_faiss()
        res = {"ok": True, "doc_id": doc_id, "chunks": len(chunks)}
        if truncated:
            res["truncated"] = True
        return res

    # ——— Hybrid search
    async def hybrid_search(
        self,
        client: httpx.AsyncClient,
        *,
        query: str,
        k: int = 6,
        filters: Optional[Dict[str, Any]] = None,
        bm25_boost: float = 0.25,
    ) -> Dict[str, Any]:
        if not query:
            return {"ok": True, "results": []}
        # Embed query
        qv = (await self._embed_batch(client, [query]))[0]
        if self._index is None or self._dim is None:
            return {"ok": True, "results": []}
        qv = _norm(qv.reshape(1, -1).astype(np.float32))

        # Candidates from FAISS
        expand = max(12, k * 4)
        D, I = self._index.search(qv, expand)  # inner product in [-1,1]
        cos_scores: Dict[int, float] = {}
        for idx, score in zip(I[0], D[0]):
            if idx == -1:
                continue
            cos_scores[int(idx)] = (float(score) + 1.0) / 2.0  # → [0,1]

        # Candidates from FTS
        cands: Dict[int, float] = dict(cos_scores)
        bm25_scores: Dict[int, float] = {}
        f = filters or {}
        where_extra = []
        params: List[Any] = []
        if f.get("source"):
            where_extra.append("c.source = ?")
            params.append(str(f["source"]))
        if f.get("doc_id"):
            where_extra.append("c.doc_id = ?")
            params.append(str(f["doc_id"]))
        # tags stored as JSON list; do a simple LIKE fallback
        if f.get("tag"):
            where_extra.append("c.tags LIKE ?")
            params.append(f"%{f['tag']}%")
        where_sql = (" AND " + " AND ".join(where_extra)) if where_extra else ""
        c = self._conn.cursor()
        try:
            sql = (
                "SELECT c.id as id, bm25(chunks_fts) as r FROM chunks_fts "
                "JOIN chunks c ON c.id = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ?" + where_sql + " ORDER BY r LIMIT ?"
            )
            params2 = [query] + params + [expand]
            c.execute(sql, params2)
            for row in c.fetchall():
                rid = int(row["id"])
                r = float(row["r"]) if row["r"] is not None else 100.0
                bm25_scores[rid] = 1.0 / (1.0 + r)
                cands.setdefault(rid, 0.0)
        except sqlite3.OperationalError:
            # If MATCH fails on syntax, ignore FTS
            pass

        # Fetch metadata for candidates
        if not cands:
            return {"ok": True, "results": []}
        ids = tuple(sorted(cands.keys()))
        qmarks = ",".join(["?"] * len(ids))
        c.execute(
            f"SELECT id, doc_id, title, uri, source, tags, text FROM chunks WHERE id IN ({qmarks})",
            list(ids),
        )
        rows = c.fetchall()
        items: List[Dict[str, Any]] = []
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) >= 3]

        for r in rows:
            rid = int(r["id"])
            base = cands.get(rid, 0.0)
            score = base + bm25_boost * bm25_scores.get(rid, 0.0)
            tags = []
            try:
                tags = json.loads(r["tags"]) if r["tags"] else []
            except Exception:
                tags = []
            text = r["text"] or ""
            preview = text[:360]
            preview_l = preview.lower()
            title_l = (r["title"] or "").lower()
            term_hits = 0
            if terms:
                for term in terms:
                    if term in preview_l or term in title_l:
                        term_hits += 1

            # derive host
            uri = r["uri"] or ""
            host = ""
            try:
                host = urlsplit(uri).hostname or ""
            except Exception:
                host = ""
            if not host and tags:
                try:
                    host = str(tags[0])
                except Exception:
                    host = ""

            items.append(
                {
                    "id": rid,
                    "doc_id": r["doc_id"],
                    "title": r["title"] or "",
                    "uri": r["uri"] or "",
                    "source": r["source"] or "",
                    "tags": tags,
                    "host": host,
                    "preview": preview,
                    "score": round(score, 4),
                    "term_hits": term_hits,
                }
            )
        # Sort desc by score and cut to k
        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        filtered = items
        if not (filters or {}).get("doc_id"):
            filtered = [it for it in items if (it.get("term_hits", 0) > 0) or it.get("score", 0.0) >= 0.9]
        if not filtered:
            return {"ok": True, "results": [], "reason": "no_relevant"}
        return {"ok": True, "results": filtered[: max(1, k)]}

    # ——— Fetch concatenated text for a document
    def get_document_text(self, doc_id: str, max_chars: int = 200_000) -> Dict[str, Any]:
        """
        Return concatenated chunk text for a given doc_id, ordered by original index.
        Useful for downstream tools that need larger context windows.
        """
        try:
            c = self._conn.cursor()
            c.execute(
                "SELECT uri, source, title, tags FROM documents WHERE doc_id=?",
                (doc_id,),
            )
            doc_row = c.fetchone()
            meta = {
                "uri": doc_row["uri"] if doc_row else "",
                "source": doc_row["source"] if doc_row else "",
                "title": doc_row["title"] if doc_row else "",
                "tags": []
            }
            try:
                if doc_row and doc_row["tags"]:
                    import json as _json
                    meta["tags"] = _json.loads(doc_row["tags"])  # type: ignore
            except Exception:
                meta["tags"] = []

            c.execute(
                "SELECT text FROM chunks WHERE doc_id=? ORDER BY idx ASC",
                (doc_id,),
            )
            parts: List[str] = []
            total = 0
            for r in c.fetchall():
                t = r["text"] or ""
                if not t:
                    continue
                if total + len(t) > max_chars:
                    t = t[: max(0, max_chars - total)]
                parts.append(t)
                total += len(t)
                if total >= max_chars:
                    break
            text = "\n\n".join(parts)
            return {"ok": True, "text": text, "meta": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}


_STORE: Optional[DocStore] = None


def init_store(*, db_path: Optional[str] = None, embed_model: Optional[str] = None, ollama_host: Optional[str] = None) -> DocStore:
    global _STORE
    if _STORE is None:
        _STORE = DocStore(
            db_path=db_path or DOCSTORE_PATH,
            embed_model=embed_model or EMBED_MODEL,
            ollama_host=ollama_host or OLLAMA_HOST,
        )
    return _STORE


def get_store() -> DocStore:
    if _STORE is None:
        # Lazy init with defaults; suitable for scripts/tests
        return init_store()
    return _STORE
