"""Load cached pipeline results for live demos when APIs or GPUs are unavailable."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def offline_demo_path_for(disease_name: str) -> Path | None:
    if not os.getenv("ALCHEMY_OFFLINE_DEMO", "").lower() in ("1", "true", "yes"):
        return None
    key = disease_name.strip().lower().replace(" ", "_")
    # Chagas and common aliases
    if "chagas" in key:
        return ROOT / "data" / "demo_results" / "chagas_pipeline.json"
    p = ROOT / "data" / "demo_results" / f"{key}.json"
    if p.is_file():
        return p
    return None


def load_offline_disease_result(disease_name: str) -> dict | None:
    path = offline_demo_path_for(disease_name)
    if path is None or not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = data.get("result") if isinstance(data, dict) and "result" in data else data
        if isinstance(out, dict):
            out = {**out, "offline_demo": True, "offline_demo_source": str(path)}
            return out
    except Exception as e:
        print(f"[offline_demo] Failed to load {path}: {e}")
    return None
