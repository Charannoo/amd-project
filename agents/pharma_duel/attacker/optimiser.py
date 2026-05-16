# agents/pharma_duel/attacker/optimiser.py
from rdkit import Chem

from agents.molecule_agent import get_shared_molecule_agent
from models.qwen_client import qwen_json


class Optimiser:
    SYSTEM_PROMPT = """You are an expert medicinal chemist AI.
You receive a molecule (as SMILES) and a list of objections from drug safety reviewers.
Your job is to propose a single, specific chemical modification to address the objections.

Respond ONLY with JSON in this exact format:
{
  "action": "one of: substitute, remove, add, replace_ring, add_polar_group, reduce_logp",
  "position_description": "describe where the modification happens",
  "modification": "what to change it to",
  "reason": "why this addresses the objection",
  "modified_smiles": "your best attempt at the modified SMILES string"
}

Rules:
- Keep changes minimal — one modification per round
- modified_smiles MUST be a valid SMILES string
- Address the HIGHEST SEVERITY objection first"""

    def __init__(self):
        self.mol_agent = get_shared_molecule_agent()

    def address_objections(self, smiles: str, objections: list, target_name: str) -> dict:
        if not objections:
            return {"modified_smiles": smiles, "reason": "No objections to address"}

        objection_text = "\n".join([f"- [{o['agent']}] {o['issue']}" for o in objections])
        user_msg = f"""Current molecule: {smiles}
Target protein: {target_name}
Objections from safety reviewers:
{objection_text}

Propose ONE specific chemical modification to address these issues."""

        result = qwen_json(self.SYSTEM_PROMPT, user_msg)
        proposed = result.get("modified_smiles", smiles)
        mol = Chem.MolFromSmiles(str(proposed))
        if mol is None:
            instruction = result.get("modification", "reduce toxicity")
            proposed = self.mol_agent.modify_smiles(smiles, instruction)
            result["modified_smiles"] = proposed
            result["smiles_source"] = "molt5_fallback"

        return result

    def get_improvements(self, smiles: str, objections: list, target_name: str) -> dict:
        user_msg = f"""Current molecule: {smiles}
Target protein: {target_name}
The molecule has PASSED all safety checks.
Now improve its binding affinity without introducing new toxicity."""

        result = qwen_json(self.SYSTEM_PROMPT, user_msg)
        proposed = result.get("modified_smiles", smiles)
        mol = Chem.MolFromSmiles(str(proposed))
        if mol is None:
            return {"modified_smiles": smiles, "reason": "Optimisation skipped — invalid SMILES"}
        return result
