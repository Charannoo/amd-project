# agents/molecule_agent.py
"""
MoleculeAgent: Generates novel drug candidates using MolT5.
Validates every output with RDKit before returning.
"""
import torch
from rdkit import Chem
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config.settings import MOLT5_MODEL


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


_shared_molecule_agent = None


def get_shared_molecule_agent() -> "MoleculeAgent":
    global _shared_molecule_agent
    if _shared_molecule_agent is None:
        _shared_molecule_agent = MoleculeAgent()
    return _shared_molecule_agent


class MoleculeAgent:
    def __init__(self):
        print("[MoleculeAgent] Loading MolT5...")
        self.tokenizer = T5Tokenizer.from_pretrained(MOLT5_MODEL)
        self.model = T5ForConditionalGeneration.from_pretrained(MOLT5_MODEL)
        self.model = self.model.eval().to(_device())
        print("[MoleculeAgent] MolT5 loaded")

    def generate_from_prompt(self, prompt: str, num_return: int = 5) -> list:
        """Generate SMILES from a text prompt using MolT5."""
        dev = _device()
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=num_return,
                num_beams=max(num_return, 5),
                early_stopping=True,
                do_sample=False,
            )

        smiles_list = []
        for output in outputs:
            smiles = self.tokenizer.decode(output, skip_special_tokens=True)
            smiles = smiles.strip()
            if self.validate_smiles(smiles):
                smiles_list.append(smiles)

        return smiles_list

    def modify_smiles(self, smiles: str, instruction: str) -> str:
        """Modify an existing molecule based on a chemical instruction."""
        prompt = f"Modify the following molecule: {smiles}\nInstruction: {instruction}\nResult:"
        generated = self.generate_from_prompt(prompt, num_return=3)
        return generated[0] if generated else smiles

    def validate_smiles(self, smiles: str) -> bool:
        """Validate SMILES using RDKit."""
        if not smiles:
            return False
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None

    def generate_from_protein(self, protein_name: str, num_candidates: int = 5) -> list:
        """Generate drug candidates for a protein target."""
        prompt = f"Generate a drug-like small molecule that binds to {protein_name}:"
        smiles_list = self.generate_from_prompt(prompt, num_return=num_candidates * 2)
        return smiles_list[:num_candidates]

    def run(self, target_name: str, num_candidates: int = 5) -> dict:
        """Generate novel candidates for a disease target."""
        print(f"[MoleculeAgent] Generating {num_candidates} candidates for {target_name}")
        candidates = self.generate_from_protein(target_name, num_candidates)
        print(f"[MoleculeAgent] Generated {len(candidates)} valid SMILES")
        return {
            "target": target_name,
            "novel_candidates": candidates,
            "count": len(candidates),
        }
