"""
Build PubMed abstract FAISS index with BioBERT embeddings (run once / periodically).

Respect NCBI rate limits: https://www.ncbi.nlm.nih.gov/books/NBK25497/
Usage (from repo root):
  set ENTREZ_EMAIL=you@example.com
  python data/build_pubmed_index.py --limit 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import faiss
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    ENTREZ_EMAIL,
    ENTREZ_TOOL,
    PUBMED_API,
    PUBMED_FETCH_BATCH,
    PUBMED_INDEX_PATH,
    PUBMED_METADATA_PATH,
)
from models.biobert_embedder import BioBERTEmbedder

DEFAULT_QUERY = (
    '("drug discovery" OR repurposing OR "medicinal chemistry" OR '
    '"small molecule" OR antiviral OR antibiotic) AND english[lang]'
)


def entrez_params(extra: dict) -> dict:
    base = {"tool": ENTREZ_TOOL, "retmode": "json"}
    if ENTREZ_EMAIL:
        base["email"] = ENTREZ_EMAIL
    base.update(extra)
    return base


def esearch_ids(term: str, retmax: int, sort: str = "relevance") -> list[str]:
    url = f"{PUBMED_API}/esearch.fcgi"
    params = entrez_params(
        {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "sort": sort,
        }
    )
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def efetch_xml(id_batch: list[str]) -> str:
    url = f"{PUBMED_API}/efetch.fcgi"
    params = entrez_params(
        {
            "db": "pubmed",
            "id": ",".join(id_batch),
            "retmode": "xml",
        }
    )
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.text


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()) if title_el is not None else ""

        abstract_parts = []
        ab = article.find(".//Abstract")
        if ab is not None:
            for at in ab.findall("AbstractText"):
                label = at.get("Label")
                label_prefix = f"{label}: " if label else ""
                abstract_parts.append(label_prefix + "".join(at.itertext()))
        abstract = " ".join(abstract_parts).strip()

        if not pmid or (not abstract and not title):
            continue
        text_for_embed = f"{title}. {abstract}".strip()[:2000]
        records.append(
            {
                "pmid": pmid,
                "title": title[:500],
                "abstract": abstract[:4000],
                "text": text_for_embed,
            }
        )
    return records


def build(limit: int, term: str, sleep_s: float) -> None:
    if not ENTREZ_EMAIL:
        print("Warning: ENTREZ_EMAIL not set. Set it in .env for responsible NCBI usage.")

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"PubMed esearch (max {limit})...")
    ids = esearch_ids(term, min(limit * 2, 20000))[:limit]
    if not ids:
        print("No PMIDs returned; check query or network.")
        return

    all_records: list[dict] = []
    for i in range(0, len(ids), PUBMED_FETCH_BATCH):
        batch = ids[i : i + PUBMED_FETCH_BATCH]
        xml_text = efetch_xml(batch)
        all_records.extend(parse_pubmed_xml(xml_text))
        time.sleep(sleep_s)

    if not all_records:
        print("No abstracts parsed.")
        return

    print(f"Embedding {len(all_records)} abstracts with BioBERT...")
    embedder = BioBERTEmbedder()
    texts = [r["text"] for r in all_records]
    emb = embedder.embed_texts(texts, batch_size=8)

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb.astype("float32"))

    idx_path = Path(PUBMED_INDEX_PATH)
    meta_path = Path(PUBMED_METADATA_PATH)
    if not idx_path.is_absolute():
        idx_path = ROOT / idx_path
    if not meta_path.is_absolute():
        meta_path = ROOT / meta_path

    faiss.write_index(index, str(idx_path))
    meta_save = [{k: v for k, v in r.items() if k != "text"} for r in all_records]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_save, f, indent=2)

    print(f"Wrote {index.ntotal} vectors → {idx_path}")
    print(f"Metadata → {meta_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2500)
    ap.add_argument("--term", type=str, default=DEFAULT_QUERY)
    ap.add_argument("--sleep", type=float, default=0.35, help="Seconds between efetch batches")
    args = ap.parse_args()
    build(args.limit, args.term, args.sleep)


if __name__ == "__main__":
    main()
