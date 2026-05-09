"""3D / export helpers for agents (UI delegates to ui.components.molecule_viewer)."""
from rdkit import Chem


class VisualiserAgent:
    """Prepares structure data for downstream viewers."""

    @staticmethod
    def validate_smiles(smiles: str) -> bool:
        return Chem.MolFromSmiles(smiles) is not None
