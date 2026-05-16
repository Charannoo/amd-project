# modes/repurposing_mode.py
"""Mode 5: bidirectional repurposing (drug ↔ disease) with PharmaDuel on top hits."""
import textwrap

import requests

from agents.opentargets_client import enrich_candidates_with_opentargets
from agents.pharma_duel.pharma_duel import PharmaDuel
from agents.repurposing_agent import RepurposingAgent
from agents.target_agent import TargetAgent
from config.settings import UNIPROT_API
from modes.async_utils import run_coro_sync
from models.qwen_client import qwen_chat


class RepurposingMode:
    def __init__(self, esm2_model, esm2_alphabet):
        self.esm2_model = esm2_model
        self.esm2_alphabet = esm2_alphabet
        self.target_agent = TargetAgent(esm2_model, esm2_alphabet)
        self.repurposing = RepurposingAgent()
        self.pharma_duel = PharmaDuel(esm2_model, esm2_alphabet)

    def run(self, query: str, query_type: str, progress_callback=None) -> dict:
        def cb(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        if query_type.startswith("Disease"):
            cb("Identifying disease targets...", 10)
            tgt = self.target_agent.run(query)
            if not tgt.get("proteins"):
                return {"error": f"No targets for {query}", "survivors": []}
            primary = tgt["primary_target"]
            rep = self.repurposing.run(tgt)
            cb("Enriching candidates with OpenTargets...", 35)
            if rep.get("candidates"):
                rep = {
                    **rep,
                    "candidates": enrich_candidates_with_opentargets(
                        rep["candidates"], disease_name=query, drug_name=None
                    ),
                }
            smiles_list = [d["smiles"] for d in rep.get("candidates", [])[:5] if d.get("smiles")]
            if not smiles_list:
                smiles_list = ["CC(=O)Oc1ccccc1C(=O)O"]
        else:
            cb("Resolving drug target via PubChem + UniProt...", 10)
            primary = self._drug_to_target_embedding(query)
            if not primary:
                return {"error": f"Could not resolve drug target for {query}", "survivors": []}
            rep = self.repurposing.run({"primary_target": primary})
            if rep.get("candidates"):
                rep = {
                    **rep,
                    "candidates": enrich_candidates_with_opentargets(
                        rep["candidates"], disease_name=None, drug_name=query
                    ),
                }
            smiles_list = [d["smiles"] for d in rep.get("candidates", [])[:5] if d.get("smiles")] or [
                "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
            ]

        cb("Running PharmaDuel adversarial battle on top candidates...", 60)
        battle = run_coro_sync(self.pharma_duel.run(smiles_list, primary))
        survivors = battle.get("survivors", [])
        cb("Generating repurposing report...", 90)
        report = qwen_chat(
            "You are a drug repurposing scientist.",
            textwrap.shorten(
                f"Query: {query}, type: {query_type}. Top similar hits: {len(smiles_list)}. "
                f"Survivors after PharmaDuel: {len(survivors)}.",
                width=1000,
            ),
            max_tokens=500,
        )
        cb("Complete!", 100)
        return {"survivors": survivors, "pharma_duel": battle, "report": report, "repurpose_hits": rep}

    def _drug_to_target_embedding(self, drug_name: str) -> dict | None:
        try:
            import pubchempy as pcp
        except ImportError:
            pcp = None
        smiles = None
        if pcp:
            try:
                compounds = pcp.get_compounds(drug_name, "name")
                if compounds:
                    smiles = compounds[0].canonical_smiles
            except Exception:
                pass
        if not smiles:
            return None
        params = {
            "query": f"({drug_name}) AND reviewed:true",
            "format": "json",
            "fields": "accession,sequence,proteinDescription",
            "size": 1,
        }
        r = requests.get(UNIPROT_API, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        sequence = results[0].get("sequence", {}).get("value", "")
        if len(sequence) < 50:
            return None
        from models.esm2_loader import get_embedding

        desc = results[0].get("proteinDescription", {}) or {}
        rec = desc.get("recommendedName", {}) or {}
        full = rec.get("fullName", {}) or {}
        name = full.get("value", drug_name)
        return {
            "name": name,
            "sequence": sequence,
            "embedding": get_embedding(sequence[:1022], self.esm2_model, self.esm2_alphabet),
            "seed_smiles": smiles,
        }
