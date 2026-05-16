# modes/molecule_chat_mode.py
"""Mode 2: chat to modify molecules with live ADMET / 3D."""
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem

from agents.admet_agent import ADMETAgent
from agents.molecule_agent import get_shared_molecule_agent
from models.qwen_client import qwen_json


class MoleculeChatMode:
    SYSTEM_PROMPT = """You are an expert medicinal chemist AI.
The user is chatting with a drug molecule and wants to modify it.
Interpret their natural language instruction as a chemical modification.

Respond ONLY with JSON:
{
  "interpretation": "what you understood",
  "modification_type": "substitute/add/remove/ring_change/optimize",
  "chemical_instruction": "specific instruction for generation",
  "modified_smiles": "your best SMILES for the modified molecule",
  "explanation": "plain English explanation"
}"""

    def __init__(self):
        self.admet_agent = ADMETAgent()
        self.mol_agent = get_shared_molecule_agent()
        self.conversation_history = []
        self.current_smiles = None
        self.admet_history = []

    def load_molecule(self, smiles_or_name: str) -> dict:
        smiles = smiles_or_name.strip()
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            try:
                results = pcp.get_compounds(smiles_or_name, "name")
                if results:
                    smiles = results[0].canonical_smiles
                    mol = Chem.MolFromSmiles(smiles)
            except Exception:
                pass

        if mol is None:
            return {"error": f"Could not find molecule: {smiles_or_name}"}

        self.current_smiles = smiles
        admet = self.admet_agent.run(smiles)
        self.admet_history = [admet]

        return {
            "smiles": smiles,
            "admet": admet,
            "3d_sdf": self.smiles_to_3d_sdf(smiles),
            "name": smiles_or_name,
        }

    def chat(self, user_message: str) -> dict:
        if not self.current_smiles:
            return {"error": "No molecule loaded. Please load a molecule first."}

        self.conversation_history.append({"role": "user", "content": user_message})

        user_prompt = f"""Current molecule: {self.current_smiles}
User instruction: {user_message}
Previous ADMET score: {self.admet_history[-1].get('admet_score', 'N/A') if self.admet_history else 'N/A'}"""

        result = qwen_json(self.SYSTEM_PROMPT, user_prompt)

        modified_smiles = result.get("modified_smiles", self.current_smiles)
        mol = Chem.MolFromSmiles(str(modified_smiles))

        if mol is None:
            instruction = result.get("chemical_instruction", user_message)
            modified_smiles = self.mol_agent.modify_smiles(self.current_smiles, instruction)

        new_admet = self.admet_agent.run(modified_smiles)
        prev_admet = self.admet_history[-1] if self.admet_history else {}
        admet_deltas = self._compute_deltas(prev_admet, new_admet)

        self.current_smiles = modified_smiles
        self.admet_history.append(new_admet)

        response = {
            "modified_smiles": modified_smiles,
            "interpretation": result.get("interpretation", ""),
            "explanation": result.get("explanation", ""),
            "admet": new_admet,
            "admet_deltas": admet_deltas,
            "3d_sdf": self.smiles_to_3d_sdf(modified_smiles),
            "improved": new_admet.get("admet_score", 0) > prev_admet.get("admet_score", 0),
        }

        self.conversation_history.append(
            {"role": "assistant", "content": f"Modified: {result.get('explanation', 'Molecule updated')}"}
        )

        return response

    def smiles_to_3d_sdf(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
            AllChem.MMFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
            return Chem.MolToMolBlock(mol)
        except Exception:
            return ""

    def _compute_deltas(self, prev: dict, curr: dict) -> dict:
        deltas = {}
        prev_props = prev.get("properties", {})
        curr_props = curr.get("properties", {})
        for key in ["molecular_weight", "logp", "tpsa", "qed", "sa_score"]:
            if key in prev_props and key in curr_props:
                delta = curr_props[key] - prev_props[key]
                deltas[key] = {"before": prev_props[key], "after": curr_props[key], "delta": delta}
        return deltas
