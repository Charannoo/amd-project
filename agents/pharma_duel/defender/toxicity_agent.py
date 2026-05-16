# agents/pharma_duel/defender/toxicity_agent.py
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, MolFromSmarts
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


class ToxicityDefender:
    def evaluate(self, smiles: str) -> dict:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"passed": False, "reason": "Invalid SMILES — molecule cannot be parsed"}

        issues = []

        mw = Descriptors.MolWt(mol)
        if mw > 600:
            issues.append(f"Molecular weight too high ({mw:.0f} Da > 600)")

        logp = Crippen.MolLogP(mol)
        if logp > 5:
            issues.append(f"LogP too high ({logp:.2f} > 5.0) — risk of toxicity")
        if logp < -2:
            issues.append(f"LogP too low ({logp:.2f} < -2.0) — poor absorption")

        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
            catalog = FilterCatalog(params)
            if catalog.HasMatch(mol):
                entry = catalog.GetFirstMatch(mol)
                issues.append(f"PAINS alert: {entry.GetDescription()} — assay interference risk")
        except Exception:
            pass

        try:
            p2 = FilterCatalogParams()
            p2.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            brenk = FilterCatalog(p2)
            if brenk.HasMatch(mol):
                e2 = brenk.GetFirstMatch(mol)
                issues.append(f"Brenk medicinal-chemistry alert: {e2.GetDescription()}")
        except Exception:
            pass

        num_basic_n = sum(
            1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7 and atom.GetTotalValence() < 4
        )
        if num_basic_n > 0 and logp > 3.5:
            issues.append(f"hERG cardiotoxicity risk: basic N present + high LogP ({logp:.2f})")

        mutagenic_smarts = ["[N+]([O-])=O", "c1ccc2cc3ccccc3cc2c1", "NC(=O)ON"]
        for smarts in mutagenic_smarts:
            pattern = MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                issues.append("Mutagenicity alert: structural alert detected")
                break

        passed = len(issues) == 0
        return {
            "passed": passed,
            "reason": "; ".join(issues) if issues else "All toxicity checks passed",
            "issues": issues,
            "mw": mw,
            "logp": logp,
        }
