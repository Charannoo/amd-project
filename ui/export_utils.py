"""Export helpers for Gradio callbacks."""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "outputs"


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip().lower())
    return cleaned.strip("_") or "alchemy_result"


def _pharma_duel_result(result: dict[str, Any]) -> dict[str, Any]:
    pharma = result.get("pharma_duel")
    if isinstance(pharma, dict) and pharma:
        return pharma
    nested = result.get("pipeline_steps", {}).get("pharma_duel", {})
    return nested if isinstance(nested, dict) else {}


def write_result_exports(result: dict[str, Any], name: str) -> list[str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = EXPORT_DIR / f"{_safe_name(name)}-{stamp}"

    json_path = base.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    rows = []
    pharma = _pharma_duel_result(result)
    for status, items in (("survived", pharma.get("survivors", [])), ("eliminated", pharma.get("eliminated", []))):
        for item in items:
            rows.append(
                {
                    "status": status,
                    "initial_smiles": item.get("initial_smiles", ""),
                    "final_smiles": item.get("final_smiles", ""),
                    "best_binding_score": item.get("best_binding_score", ""),
                    "rounds_survived": item.get("rounds_survived", ""),
                    "elimination_reason": "; ".join(item.get("elimination_reason") or []),
                }
            )

    if not rows:
        for item in pharma.get("battle_results", []):
            rows.append(
                {
                    "status": "survived" if item.get("survived") else "eliminated",
                    "initial_smiles": item.get("initial_smiles", ""),
                    "final_smiles": item.get("final_smiles", ""),
                    "best_binding_score": item.get("best_binding_score", ""),
                    "rounds_survived": item.get("rounds_survived", ""),
                    "elimination_reason": "; ".join(item.get("elimination_reason") or []),
                }
            )

    paths = [str(json_path)]
    if rows:
        csv_path = base.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(str(csv_path))

    return paths
