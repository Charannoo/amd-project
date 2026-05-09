"""BioBERT mean-pooled embeddings for PubMed RAG (768-dim)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import BIOBERT_MODEL

_embedder_singleton: "BioBERTEmbedder | None" = None


class BioBERTEmbedder:
    def __init__(self, model_name: str | None = None):
        name = model_name or BIOBERT_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()

    @torch.inference_mode()
    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        """Return float32 array (N, 768), L2-normalized rows."""
        out_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self.model(**enc)
            last = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            pooled = (summed / counts).float().cpu().numpy()
            out_list.append(pooled)
        emb = np.vstack(out_list).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
        emb = emb / norms
        return emb

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_texts([text], batch_size=1)


def get_biobert_embedder() -> BioBERTEmbedder:
    global _embedder_singleton
    if _embedder_singleton is None:
        _embedder_singleton = BioBERTEmbedder()
    return _embedder_singleton
