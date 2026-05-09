# agents/literature_agent.py
"""
LiteratureAgent: PubMed — FAISS + BioBERT RAG when index exists; always merge live PubMed hits.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import requests

from config.settings import (
    FAISS_TOP_K,
    PUBMED_API,
    PUBMED_INDEX_PATH,
    PUBMED_METADATA_PATH,
)
from models.biobert_embedder import get_biobert_embedder
from models.qwen_client import qwen_chat


class LiteratureAgent:
    def __init__(self):
        self._index = None
        self._meta: list[dict] = []
        self._load_local_index()

    def _load_local_index(self) -> None:
        root = Path(__file__).resolve().parents[1]
        ip = Path(PUBMED_INDEX_PATH)
        mp = Path(PUBMED_METADATA_PATH)
        if not ip.is_absolute():
            ip = root / ip
        if not mp.is_absolute():
            mp = root / mp
        if not ip.is_file() or not mp.is_file():
            return
        try:
            self._index = faiss.read_index(str(ip))
            with open(mp, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            print(f"[LiteratureAgent] Loaded PubMed FAISS index ({self._index.ntotal} docs)")
        except Exception as e:
            print(f"[LiteratureAgent] Could not load PubMed index: {e}")
            self._index = None
            self._meta = []

    def _faiss_search(self, query: str, top_k: int) -> list[dict]:
        if self._index is None or not self._meta or self._index.ntotal == 0:
            return []
        q_emb = get_biobert_embedder().embed_query(query).astype("float32")
        faiss.normalize_L2(q_emb)
        k = min(top_k, self._index.ntotal)
        dist, idx = self._index.search(q_emb, k)
        rows = []
        for rank, i in enumerate(idx[0]):
            if i < 0 or i >= len(self._meta):
                continue
            m = dict(self._meta[i])
            m["faiss_score"] = float(dist[0][rank])
            rows.append(m)
        return rows

    def _live_pubmed(self, query: str, max_papers: int) -> tuple[list[dict], str]:
        search_url = f"{PUBMED_API}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_papers * 2,
            "retmode": "json",
        }
        try:
            r = requests.get(search_url, params=params, timeout=15)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])[:max_papers]
        except Exception:
            return [], ""

        if not ids:
            return [], ""

        fetch_url = f"{PUBMED_API}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "text",
            "rettype": "abstract",
        }
        try:
            abstract_text = requests.get(fetch_url, params=params, timeout=15).text
        except Exception:
            abstract_text = ""

        papers = [{"pmid": pid, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"} for pid in ids]
        return papers, abstract_text

    def search(self, disease: str, target: str, max_papers: int = 5) -> dict:
        query = f"{disease} {target} drug therapy treatment target"

        faiss_hits = self._faiss_search(query, top_k=max(FAISS_TOP_K, 12))
        live_papers, live_abstracts = self._live_pubmed(query, max_papers=max_papers)

        seen: set[str] = set()
        merged: list[dict] = []

        for h in faiss_hits:
            pid = str(h.get("pmid", ""))
            if pid and pid not in seen:
                seen.add(pid)
                merged.append(
                    {
                        "pmid": pid,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                        "title": h.get("title", ""),
                        "abstract": (h.get("abstract") or "")[:1500],
                        "source": "faiss",
                    }
                )

        for p in live_papers:
            pid = str(p.get("pmid", ""))
            if pid and pid not in seen:
                seen.add(pid)
                merged.append({**p, "title": "", "abstract": "", "source": "live"})

        context_parts = []
        for m in merged[:15]:
            t = m.get("title") or ""
            a = m.get("abstract") or ""
            if t or a:
                context_parts.append(f"PMID {m['pmid']}: {t}\n{a}".strip())
        context_text = "\n\n".join(context_parts)[:6000]

        if not context_text and live_abstracts:
            context_text = live_abstracts[:4500]

        if context_text:
            summary = qwen_chat(
                "You are a pharmaceutical scientist. Summarize evidence for drug discovery in 3–4 sentences. "
                "Cite PMIDs casually if helpful.",
                f"Disease: {disease}\nTarget: {target}\nEvidence snippets:\n{context_text}",
                max_tokens=500,
            )
        else:
            summary = (
                "No abstract text retrieved. Build a local PubMed index: "
                "`python data/build_pubmed_index.py` (set ENTREZ_EMAIL)."
            )

        return {
            "papers": [{"pmid": m["pmid"], "url": m["url"]} for m in merged[: max_papers * 3]],
            "summary": summary,
            "query": query,
            "faiss_rows_used": len(faiss_hits),
            "live_rows_used": len(live_papers),
        }
