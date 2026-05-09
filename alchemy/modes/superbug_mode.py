# modes/superbug_mode.py
"""Mode 4: bacteria + resistance mechanism → target protein → antibiotics."""
import textwrap

import requests

from agents.molecule_agent import get_shared_molecule_agent
from agents.pharma_duel.pharma_duel import PharmaDuel
from agents.target_agent import TargetAgent
from config.settings import UNIPROT_API
from modes.async_utils import run_coro_sync
from models.qwen_client import qwen_chat


class SuperbugMode:
    RESISTANCE_QUERIES = {
        "beta-lactamase": "gene:TEM1 AND organism_name:Escherichia coli",
        "efflux": "gene:acrB AND reviewed:true",
        "mrsa": "gene:mecA AND reviewed:true",
    }

    def __init__(self, esm2_model, esm2_alphabet):
        self.target_agent = TargetAgent(esm2_model, esm2_alphabet)
        self._molecule_agent = None
        self.pharma_duel = PharmaDuel(esm2_model, esm2_alphabet)

    @property
    def molecule_agent(self):
        if self._molecule_agent is None:
            self._molecule_agent = get_shared_molecule_agent()
        return self._molecule_agent

    def _uniprot_fallback(self, bacteria: str, mechanism: str) -> dict:
        mech = mechanism.lower()
        qkey = "beta-lactamase"
        for k in self.RESISTANCE_QUERIES:
            if k in mech:
                qkey = k
                break
        params = {
            "query": self.RESISTANCE_QUERIES.get(qkey, f"{bacteria} AND reviewed:true"),
            "format": "json",
            "fields": "accession,sequence,proteinDescription,geneNames,length",
            "size": 1,
        }
        r = requests.get(UNIPROT_API, params=params, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return {}
        res = results[0]
        sequence = res.get("sequence", {}).get("value", "")
        desc = res.get("proteinDescription", {}) or {}
        rec = desc.get("recommendedName", {}) or {}
        full = rec.get("fullName", {}) or {}
        name = full.get("value", "Resistance-associated protein")
        return {"name": name, "sequence": sequence, "uniprot_id": res.get("primaryAccession")}

    def run(self, bacteria: str, resistance_mechanism: str, progress_callback=None) -> dict:
        def cb(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        cb("Fetching resistance-associated protein targets...", 10)
        ta = self.target_agent
        proteins = ta.fetch_proteins(f"{bacteria} antibiotic resistance", max_proteins=2)
        if not proteins:
            fb = self._uniprot_fallback(bacteria, resistance_mechanism)
            if not fb or not fb.get("sequence"):
                return {"error": "Could not resolve resistance target", "survivors": []}
            proteins = ta.embed_proteins(
                [
                    {
                        "name": fb["name"],
                        "sequence": fb["sequence"],
                        "gene": "unknown",
                        "length": len(fb["sequence"]),
                        "uniprot_id": fb.get("uniprot_id"),
                    }
                ]
            )
        else:
            proteins = ta.embed_proteins(proteins)

        primary = proteins[0]
        cb(f"Target resolved: {primary['name']} — generating antibiotic scaffolds...", 45)
        prompt = (
            f"Novel Gram-negative penetrant antibiotic avoiding {resistance_mechanism} for {bacteria}:"
        )
        smiles_list = self.molecule_agent.generate_from_prompt(prompt, num_return=6)
        candidates = smiles_list[:5] or ["CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"]

        cb("Running PharmaDuel adversarial battle...", 65)
        battle = run_coro_sync(self.pharma_duel.run(candidates, primary))
        survivors = battle.get("survivors", [])
        cb("Generating antibiotic strategy report...", 90)
        report = qwen_chat(
            "You are an antibiotic discovery expert. Summarize scaffold strategy.",
            textwrap.shorten(
                f"Bacteria: {bacteria}, mechanism: {resistance_mechanism}. "
                f"Target: {primary['name']}. Survivors: {len(survivors)}.",
                width=1000,
            ),
            max_tokens=450,
        )
        cb("Complete!", 100)
        return {
            "survivors": survivors,
            "pharma_duel": battle,
            "report": report,
            "target": primary["name"],
        }
