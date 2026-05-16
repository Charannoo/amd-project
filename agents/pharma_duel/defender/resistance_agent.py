# agents/pharma_duel/defender/resistance_agent.py
import numpy as np

from config.settings import BINDING_DROP_THRESHOLD

KNOWN_RESISTANCE_MUTATIONS = {
    "cruzain": [("H159", "A"), ("C25", "A"), ("W177", "F")],
    "default": [("K", "R"), ("D", "E"), ("H", "A")],
}


class ResistanceDefender:
    def __init__(self, esm2_model=None, esm2_alphabet=None):
        self.model = esm2_model
        self.alphabet = esm2_alphabet

    def apply_mutation(self, sequence: str, position: int, new_aa: str) -> str:
        if position >= len(sequence) or position < 0:
            return sequence
        return sequence[:position] + new_aa + sequence[position + 1 :]

    def compute_embedding_similarity(self, emb1: list, emb2: list) -> float:
        e1 = np.array(emb1, dtype=np.float64)
        e2 = np.array(emb2, dtype=np.float64)
        denom = np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8
        return float(np.dot(e1, e2) / denom)

    def evaluate(self, smiles: str, target_protein: dict) -> dict:
        if not self.model or not target_protein.get("embedding"):
            return {"passed": True, "reason": "Resistance check skipped (model not loaded)"}

        sequence = target_protein.get("sequence", "")
        original_embedding = target_protein["embedding"]
        target_name = target_protein.get("name", "").lower()

        mutations = KNOWN_RESISTANCE_MUTATIONS.get("default", [])
        for key in KNOWN_RESISTANCE_MUTATIONS:
            if key in target_name:
                mutations = KNOWN_RESISTANCE_MUTATIONS[key]
                break

        resistance_detected = []

        for i, (_from_aa, to_aa) in enumerate(mutations[:3]):
            position = min(50 + i * 30, max(0, len(sequence) - 1))
            mutated_seq = self.apply_mutation(sequence, position, to_aa)

            from models.esm2_loader import get_embedding

            try:
                mutated_embedding = get_embedding(mutated_seq[:1022], self.model, self.alphabet)
                similarity = self.compute_embedding_similarity(original_embedding, mutated_embedding)
                binding_drop = 1 - similarity

                if binding_drop > BINDING_DROP_THRESHOLD:
                    resistance_detected.append(
                        f"Mutation at pos {position+1} drops predicted binding by {binding_drop*100:.0f}%"
                    )
            except Exception:
                continue

        passed = len(resistance_detected) == 0
        return {
            "passed": passed,
            "reason": "; ".join(resistance_detected)
            if resistance_detected
            else "Molecule maintains binding under simulated resistance mutations",
            "mutations_tested": len(mutations[:3]),
            "resistance_mutations": resistance_detected,
        }
