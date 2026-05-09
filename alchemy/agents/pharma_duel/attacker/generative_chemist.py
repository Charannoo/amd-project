"""Generative chemist — MolT5-based proposals."""
from rdkit import Chem

from agents.molecule_agent import get_shared_molecule_agent


class GenerativeChemist:
    def __init__(self):
        self._mol = get_shared_molecule_agent()

    def propose(self, smiles: str, instruction: str) -> str:
        out = self._mol.modify_smiles(smiles, instruction)
        if Chem.MolFromSmiles(out):
            return out
        return smiles
