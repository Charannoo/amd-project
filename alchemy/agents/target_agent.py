# agents/target_agent.py
"""
TargetAgent: Given a disease name, fetches relevant protein targets
from UniProt and computes ESM2-3B embeddings.
"""
import requests

from config.settings import UNIPROT_API
from models.esm2_loader import get_embedding


class TargetAgent:
    def __init__(self, esm2_model, esm2_alphabet):
        self.model = esm2_model
        self.alphabet = esm2_alphabet

    def _parse_uniprot_results(self, data: dict) -> list:
        proteins = []
        for result in data.get("results", []):
            sequence = result.get("sequence", {}).get("value", "")
            if sequence and len(sequence) > 50:
                desc = result.get("proteinDescription", {})
                rec = desc.get("recommendedName", {}) if desc else {}
                full = rec.get("fullName", {}) if rec else {}
                name_val = full.get("value", "Unknown") if full else "Unknown"
                genes = result.get("genes", [])
                gene_name = (
                    genes[0].get("geneName", {}).get("value", "Unknown") if genes else "Unknown"
                )
                proteins.append(
                    {
                        "uniprot_id": result.get("primaryAccession"),
                        "name": name_val,
                        "sequence": sequence,
                        "gene": gene_name,
                        "length": len(sequence),
                    }
                )
        return proteins

    def fetch_proteins(self, disease_name: str, max_proteins: int = 5) -> list:
        """Fetch protein sequences for a disease from UniProt."""
        base = {
            "format": "json",
            "fields": "accession,sequence,proteinDescription,geneNames,length",
            "size": max_proteins,
        }
        queries = [
            f'disease:"{disease_name}" AND reviewed:true',
            f"({disease_name}) AND reviewed:true",
        ]
        for q in queries:
            try:
                params = {**base, "query": q}
                response = requests.get(UNIPROT_API, params=params, timeout=15)
                response.raise_for_status()
                proteins = self._parse_uniprot_results(response.json())
                if proteins:
                    return proteins
            except Exception as e:
                print(f"UniProt fetch error ({q[:40]}...): {e}")
        return []

    def embed_proteins(self, proteins: list) -> list:
        """Add ESM2 embeddings to protein list."""
        for protein in proteins:
            seq = protein["sequence"][:1022]
            protein["embedding"] = get_embedding(seq, self.model, self.alphabet)
            print(f"  Embedded protein: {protein['name'][:50]} ({len(seq)} aa)")
        return proteins

    def run(self, disease_name: str) -> dict:
        """Full target identification pipeline."""
        print(f"[TargetAgent] Identifying targets for: {disease_name}")
        proteins = self.fetch_proteins(disease_name)
        if not proteins:
            return {"error": f"No proteins found for '{disease_name}'", "proteins": []}
        proteins = self.embed_proteins(proteins)
        print(f"[TargetAgent] Found {len(proteins)} protein targets")
        return {
            "disease": disease_name,
            "proteins": proteins,
            "primary_target": proteins[0] if proteins else None,
        }
