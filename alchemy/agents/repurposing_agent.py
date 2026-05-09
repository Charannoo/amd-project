# agents/repurposing_agent.py
"""
RepurposingAgent: Given a protein embedding, searches the pre-built
FAISS index of drugs and returns the most similar candidates.
"""
import json
from pathlib import Path

import faiss
import numpy as np

from config.settings import DRUG_INDEX_PATH, DRUG_METADATA_PATH, FAISS_TOP_K


class RepurposingAgent:
    def __init__(self):
        index_path = Path(DRUG_INDEX_PATH)
        meta_path = Path(DRUG_METADATA_PATH)
        if not index_path.is_file() or not meta_path.is_file():
            print("[RepurposingAgent] FAISS index or metadata missing — run main.py once or build_drug_index.py")
            self.index = None
            self.metadata = []
            return
        print("[RepurposingAgent] Loading FAISS drug index...")
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"[RepurposingAgent] Loaded {self.index.ntotal} drug embeddings")

    def search(self, query_embedding: list, top_k: int = FAISS_TOP_K) -> list:
        """Find top-K similar drugs for a query protein embedding."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            drug = self.metadata[idx].copy()
            drug["similarity_score"] = float(dist)
            drug["repurposing_confidence"] = int(min(100, float(dist) * 100))
            results.append(drug)

        return sorted(results, key=lambda x: x["similarity_score"], reverse=True)

    def run(self, target_result: dict) -> dict:
        """Run repurposing search on primary disease target."""
        primary = target_result.get("primary_target")
        if not primary or "embedding" not in primary:
            return {"error": "No valid target embedding", "candidates": []}
        if self.index is None:
            return {
                "error": "Drug index not loaded",
                "candidates": [],
                "target_protein": primary["name"],
            }

        print(f"[RepurposingAgent] Searching {self.index.ntotal} drugs...")
        candidates = self.search(primary["embedding"])
        print(f"[RepurposingAgent] Found {len(candidates)} repurposing candidates")

        return {
            "target_protein": primary["name"],
            "candidates": candidates,
            "top_candidate": candidates[0] if candidates else None,
        }
