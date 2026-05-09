# modes/pandemic_mode.py
"""Mode 3: viral genome / RNA → translated proteins → antiviral candidates."""
import re
import textwrap

from Bio.Seq import Seq

from agents.molecule_agent import get_shared_molecule_agent
from agents.pharma_duel.pharma_duel import PharmaDuel
from modes.async_utils import run_coro_sync
from models.esm2_loader import get_embedding
from models.qwen_client import qwen_chat


class PandemicMode:
    def __init__(self, esm2_model, esm2_alphabet):
        self.esm2_model = esm2_model
        self.esm2_alphabet = esm2_alphabet
        self._molecule_agent = None
        self.pharma_duel = PharmaDuel(esm2_model, esm2_alphabet)

    @property
    def molecule_agent(self):
        if self._molecule_agent is None:
            self._molecule_agent = get_shared_molecule_agent()
        return self._molecule_agent

    def _clean_sequence(self, raw: str) -> str:
        s = raw.strip().upper()
        if s.startswith(">"):
            lines = s.splitlines()
            s = "".join(lines[1:])
        return re.sub(r"[^ACGTUN]", "", s)

    def _orfs_to_proteins(self, nuc: str, max_proteins: int = 3) -> list:
        """Minimal ORF scan on both strands."""
        seq = Seq(nuc.replace("U", "T"))
        proteins_found = []
        for strand in (seq, seq.reverse_complement()):
            for frame in range(3):
                aa = str(strand[frame:].translate(to_stop=False))
                for segment in aa.split("*"):
                    for chunk in re.findall(r"M[A-Z]{30,}", segment):
                        proteins_found.append(chunk[:1022])
                        if len(proteins_found) >= max_proteins:
                            return proteins_found
        if not proteins_found and len(nuc) > 90:
            for segment in str(seq.translate(to_stop=False)).split("*"):
                chunk = re.sub(r"[^A-Z]", "", segment)[:1022]
                if len(chunk) > 50:
                    proteins_found.append(chunk[:1022])
                    break
        return proteins_found[:max_proteins]

    def run(self, genome_sequence: str, progress_callback=None) -> dict:
        def cb(msg, pct):
            if progress_callback:
                progress_callback(msg, pct)

        cleaned = self._clean_sequence(genome_sequence)
        if len(cleaned) < 90:
            return {"error": "Genome sequence too short", "proteins_found": [], "survivors": []}

        cb("Scanning genome for open reading frames...", 10)
        proteins_found = self._orfs_to_proteins(cleaned)
        if not proteins_found:
            return {"error": "No protein-like ORFs found", "proteins_found": [], "survivors": []}

        cb(f"Found {len(proteins_found)} viral protein(s) — computing ESM2 embeddings...", 30)
        primary = proteins_found[0]
        target_protein = {
            "name": "Viral polyprotein fragment",
            "sequence": primary,
            "embedding": get_embedding(primary, self.esm2_model, self.esm2_alphabet),
        }

        cb("Generating antiviral candidate molecules...", 50)
        novel = self.molecule_agent.run("viral RdRp / structural protease target", num_candidates=5)
        candidates = novel.get("novel_candidates", []) or ["CC(=O)Oc1ccccc1C(=O)O"]

        cb("Running PharmaDuel adversarial battle...", 65)
        battle = run_coro_sync(self.pharma_duel.run(candidates, target_protein))

        survivors = battle.get("survivors", [])
        cb("Generating antiviral strategy report...", 90)
        report = qwen_chat(
            "You are a virology-focused medicinal chemist. Summarize antiviral strategy in 2 short paragraphs.",
            textwrap.shorten(
                f"Viral proteins (first 200 aa): {primary[:200]}. Survivors: {len(survivors)}. Top SMILES: "
                f"{survivors[0].get('final_smiles') if survivors else 'n/a'}",
                width=1200,
            ),
            max_tokens=400,
        )

        cb("Complete!", 100)
        return {
            "proteins_found": [{"length": len(p), "preview": p[:120]} for p in proteins_found],
            "survivors": survivors,
            "pharma_duel": battle,
            "report": report,
        }
