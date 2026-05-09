# agents/pharma_duel/defender/synthesis_agent.py
import requests
from rdkit import Chem

from config.settings import ASKCOS_API, SA_SCORE_THRESHOLD


class SynthesisDefender:
    def compute_sa_score(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 10.0
            from rdkit.Contrib.SA_Score import sascorer

            return float(sascorer.calculateScore(mol))
        except Exception:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                n_rings = mol.GetRingInfo().NumRings()
                n_stereo = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
                n_atoms = mol.GetNumHeavyAtoms()
                score = 1 + (n_rings * 0.5) + (n_stereo * 0.8) + (max(0, n_atoms - 30) * 0.1)
                return min(10.0, score)
            return 10.0

    def check_askcos(self, smiles: str, timeout: int = 5) -> dict:
        try:
            response = requests.post(
                f"{ASKCOS_API}/tree-builder/",
                json={"smiles": smiles, "max_depth": 4, "max_branching": 25},
                timeout=timeout,
            )
            if response.status_code == 200:
                data = response.json()
                trees = data.get("trees", [])
                return {
                    "available": True,
                    "routes_found": len(trees) > 0,
                    "num_routes": len(trees),
                }
        except Exception:
            pass
        return {"available": False, "routes_found": None}

    def evaluate(self, smiles: str) -> dict:
        sa_score = self.compute_sa_score(smiles)
        askcos = self.check_askcos(smiles)

        issues = []
        if sa_score > SA_SCORE_THRESHOLD:
            issues.append(f"SA score {sa_score:.1f} > {SA_SCORE_THRESHOLD} — too complex to synthesize")

        if askcos["available"] and askcos["routes_found"] is False:
            issues.append("No retrosynthesis route found in ASKCOS")

        mol = Chem.MolFromSmiles(smiles)
        if mol:
            exotic = [
                atom.GetSymbol()
                for atom in mol.GetAtoms()
                if atom.GetAtomicNum() not in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
            ]
            if exotic:
                issues.append(f"Exotic elements: {', '.join(set(exotic))} — manufacturing complexity")

        passed = len(issues) == 0
        return {
            "passed": passed,
            "reason": "; ".join(issues) if issues else "Molecule is synthesisable",
            "sa_score": sa_score,
            "sa_pass": sa_score <= SA_SCORE_THRESHOLD,
            "askcos": askcos,
            "issues": issues,
        }
