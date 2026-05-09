# agents/admet_agent.py
"""
ADMETAgent: Computes ADMET-oriented properties for any SMILES string.
Uses RDKit (fast, local).
"""
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors


class ADMETAgent:
    PROPERTY_THRESHOLDS = {
        "molecular_weight": (150, 500),
        "logp": (-0.4, 5.0),
        "hbd": (0, 5),
        "hba": (0, 10),
        "tpsa": (0, 140),
        "rotatable_bonds": (0, 10),
        "qed": (0.3, 1.0),
        "sa_score": (1, 4),
    }

    def compute_rdkit(self, smiles: str) -> dict:
        """Compute all RDKit-based properties."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        try:
            from rdkit.Contrib.SA_Score import sascorer

            sa = sascorer.calculateScore(mol)
        except Exception:
            sa = 3.0

        props = {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 3),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "qed": round(QED.qed(mol), 3),
            "sa_score": round(sa, 2),
            "heavy_atoms": mol.GetNumHeavyAtoms(),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
        }
        return props

    def check_lipinski(self, props: dict) -> dict:
        """Check Lipinski Rule of Five."""
        violations = []
        if props.get("molecular_weight", 999) > 500:
            violations.append("MW > 500")
        if props.get("logp", 999) > 5:
            violations.append("LogP > 5")
        if props.get("hbd", 999) > 5:
            violations.append("HBD > 5")
        if props.get("hba", 999) > 10:
            violations.append("HBA > 10")
        return {
            "lipinski_pass": len(violations) == 0,
            "violations": violations,
            "num_violations": len(violations),
        }

    def compute_pass_fail(self, props: dict) -> dict:
        """Compute pass/fail for each property."""
        results = {}
        for prop, (low, high) in self.PROPERTY_THRESHOLDS.items():
            val = props.get(prop)
            if val is not None:
                results[f"{prop}_pass"] = low <= val <= high
        results["overall_pass_rate"] = (
            sum(1 for v in results.values() if v is True) / len(results) if results else 0
        )
        return results

    def run(self, smiles: str) -> dict:
        """Full ADMET computation for a SMILES string."""
        props = self.compute_rdkit(smiles)
        if "error" in props:
            return props
        lipinski = self.check_lipinski(props)
        pass_fail = self.compute_pass_fail(props)

        return {
            "smiles": smiles,
            "properties": props,
            "lipinski": lipinski,
            "pass_fail": pass_fail,
            "drug_like": lipinski["lipinski_pass"] and pass_fail.get("qed_pass", False),
            "admet_score": round(pass_fail["overall_pass_rate"] * 100, 1),
        }
