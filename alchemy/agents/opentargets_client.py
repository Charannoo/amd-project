"""
Open Targets Platform GraphQL client — disease clinical candidates + drug mechanisms.
Schema: https://api.platform.opentargets.org/api/v4/graphql/schema
"""
from __future__ import annotations

import re
from typing import Any

import requests

from config.settings import OPENTARGETS_API


def _graphql(query: str, variables: dict | None = None) -> dict[str, Any]:
    payload = {"query": query, "variables": variables or {}}
    r = requests.post(OPENTARGETS_API, json=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        errs = data["errors"]
        raise RuntimeError(errs[0].get("message", str(errs)) if errs else "GraphQL error")
    return data.get("data") or {}


def search_disease(query: str, page_size: int = 5) -> list[dict[str, Any]]:
    q = """
    query SearchDisease($q: String!, $s: Int!) {
      search(queryString: $q, entityNames: ["disease"], page: { index: 0, size: $s }) {
        hits {
          id
          name
          entity
          description
        }
      }
    }
    """
    try:
        data = _graphql(q, {"q": query, "s": page_size})
        return list(data.get("search", {}).get("hits") or [])
    except Exception as e:
        print(f"[OpenTargets] search_disease: {e}")
        return []


def _norm_chembl(s: str | None) -> str | None:
    if not s:
        return None
    m = re.search(r"(CHEMBL\d+)", s.upper())
    if m:
        return m.group(1)
    return None


def clinical_candidates_for_disease(efo_id: str, max_rows: int = 40) -> list[dict[str, Any]]:
    """Drugs / clinical candidates with trial indications for this disease (EFO id)."""
    q = """
    query ClinCand($efoId: String!) {
      disease(efoId: $efoId) {
        id
        name
        drugAndClinicalCandidates {
          count
          rows {
            maxClinicalStage
            drug { id name drugType }
          }
        }
      }
    }
    """
    try:
        data = _graphql(q, {"efoId": efo_id})
    except Exception as e:
        print(f"[OpenTargets] clinical_candidates_for_disease: {e}")
        return []

    dis = data.get("disease") or {}
    block = dis.get("drugAndClinicalCandidates") or {}
    rows_out: list[dict[str, Any]] = []
    for row in (block.get("rows") or [])[:max_rows]:
        drug = row.get("drug") or {}
        cid = _norm_chembl(drug.get("id"))
        if not cid:
            continue
        rows_out.append(
            {
                "chembl_id": cid,
                "drug_name": drug.get("name"),
                "drug_type": drug.get("drugType"),
                "max_clinical_stage": row.get("maxClinicalStage"),
            }
        )
    return rows_out


def drug_mechanism_targets(chembl_id: str, max_targets: int = 20) -> list[dict[str, Any]]:
    """Target symbols from curated mechanisms of action."""
    q = """
    query Mech($id: String!) {
      drug(chemblId: $id) {
        id
        name
        mechanismsOfAction {
          rows {
            targets { id approvedSymbol }
          }
        }
      }
    }
    """
    cid = _norm_chembl(chembl_id) or chembl_id
    try:
        data = _graphql(q, {"id": cid})
    except Exception as e:
        print(f"[OpenTargets] drug_mechanism_targets: {e}")
        return []

    drug = data.get("drug") or {}
    moa = drug.get("mechanismsOfAction") or {}
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in moa.get("rows") or []:
        for t in row.get("targets") or []:
            sym = t.get("approvedSymbol")
            tid = t.get("id")
            if not sym and not tid:
                continue
            key = sym or tid
            if key in seen:
                continue
            seen.add(key)
            out.append({"target_id": tid, "symbol": sym})
            if len(out) >= max_targets:
                return out
    return out


def enrich_candidates_with_opentargets(
    candidates: list[dict],
    disease_name: str | None,
    drug_name: str | None = None,
) -> list[dict]:
    known_by_chembl: dict[str, dict] = {}
    if disease_name:
        hits = search_disease(disease_name, page_size=3)
        if hits:
            efo = hits[0].get("id")
            if efo:
                for row in clinical_candidates_for_disease(efo, max_rows=50):
                    known_by_chembl[row["chembl_id"]] = row

    out = []
    for c in candidates:
        cc = c.copy()
        cid = _norm_chembl(cc.get("chembl_id"))
        if cid and cid in known_by_chembl:
            cc["opentargets"] = {
                "clinical_indication_match": True,
                **known_by_chembl[cid],
            }
        elif disease_name and cid:
            cc["opentargets"] = {"clinical_indication_match": False}
        if drug_name and cid:
            try:
                cc["opentargets_mechanism_targets"] = drug_mechanism_targets(cid)
            except Exception:
                pass
        out.append(cc)
    return out
