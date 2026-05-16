"""
Binding score for PharmaDuel: higher = better (stronger predicted binding / affinity proxy).

1. If BINDING_USE_SMINA=1 and VINA_RECEPTOR_PDBQT + VINA_BOX_CENTER are set and `smina` runs,
   score ≈ −(reported affinity kcal/mol) so good binders score higher.
2. Else: RDKit 3D conformer + shape descriptors + lipophilic/H-bond terms (research demo only).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors

from config.settings import (
    BINDING_USE_SMINA,
    SMINA_BINARY,
    SMINA_EXHAUSTIVENESS,
    SMINA_TIMEOUT_SEC,
    VINA_BOX_CENTER,
    VINA_BOX_SIZE,
    VINA_RECEPTOR_PDBQT,
)


def _mol_3d(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        if AllChem.EmbedMolecule(mol, params) != 0:
            AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        return None
    return mol


def _parse_box_center() -> tuple[float, float, float] | None:
    raw = (VINA_BOX_CENTER or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def _parse_box_size() -> tuple[float, float, float]:
    raw = (VINA_BOX_SIZE or "22,22,22").strip()
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        return (22.0, 22.0, 22.0)
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return (22.0, 22.0, 22.0)


def _run_smina(ligand_pdb: str, receptor_pdbqt: str, center: tuple[float, float, float]) -> float | None:
    smina = shutil.which(SMINA_BINARY) or SMINA_BINARY
    if shutil.which(SMINA_BINARY) is None and not Path(smina).is_file():
        return None
    rec = Path(receptor_pdbqt)
    if not rec.is_file():
        return None

    sx, sy, sz = _parse_box_size()
    cx, cy, cz = center
    out_lig = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False).name
    try:
        cmd = [
            smina,
            "-r",
            str(rec.resolve()),
            "-l",
            ligand_pdb,
            "--out",
            out_lig,
            "--center_x",
            str(cx),
            "--center_y",
            str(cy),
            "--center_z",
            str(cz),
            "--size_x",
            str(sx),
            "--size_y",
            str(sy),
            "--size_z",
            str(sz),
            "--exhaustiveness",
            str(SMINA_EXHAUSTIVENESS),
            "--cpu",
            str(min(8, os.cpu_count() or 4)),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=SMINA_TIMEOUT_SEC,
        )
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        m = re.search(r"Affinity:\s*([-\d.]+)", text)
        if not m:
            m = re.search(r"^\s*([-\d.]+)\s*kcal/mol", text, re.MULTILINE)
        if not m:
            return None
        affinity = float(m.group(1))
        # More negative affinity => better; map to higher score for PharmaDuel
        return float(-affinity)
    except Exception as e:
        print(f"[BindingPredictor] smina failed: {e}")
        return None
    finally:
        Path(out_lig).unlink(missing_ok=True)


def _score_heuristic(smiles: str, target_protein: dict) -> float:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    rot = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Lipophilic efficiency–style balance + H-bond complement (demo-only)
    lipoph = -2.8 * np.tanh(logp / 4.0)
    polar = 0.018 * hba + 0.02 * hbd - 0.00085 * tpsa
    size = -max(0.0, (mw - 420.0) / 450.0) - 0.04 * max(0, rot - 8)
    seq_len = len(target_protein.get("sequence", "")) or 300
    pocket = -0.35 * np.log1p(seq_len / 400.0)

    conf_boost = 0.0
    m3 = _mol_3d(smiles)
    if m3 is not None:
        try:
            if hasattr(rdMolDescriptors, "CalcNPR1") and hasattr(rdMolDescriptors, "CalcNPR2"):
                npr1 = rdMolDescriptors.CalcNPR1(m3)
                npr2 = rdMolDescriptors.CalcNPR2(m3)
                conf_boost = 0.15 * (1.0 - abs(npr1 - 0.35)) + 0.1 * (1.0 - abs(npr2 - 0.55))
        except Exception:
            pass

    base = 5.0 + lipoph + polar + size + pocket + conf_boost
    return float(base)


class BindingPredictor:
    """Higher score = better predicted binding (for PharmaDuel comparison)."""

    def __init__(self):
        self.last_method: str = "heuristic"

    def score(self, smiles: str, target_protein: dict) -> float:
        if BINDING_USE_SMINA and VINA_RECEPTOR_PDBQT:
            center = _parse_box_center()
            if center is not None:
                m3 = _mol_3d(smiles)
                if m3 is None:
                    self.last_method = "heuristic_invalid_3d"
                    return _score_heuristic(smiles, target_protein)
                lig_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False, encoding="utf-8")
                try:
                    lig_tmp.write(Chem.MolToPDBBlock(m3))
                    lig_tmp.close()
                    s = _run_smina(lig_tmp.name, VINA_RECEPTOR_PDBQT, center)
                    if s is not None:
                        self.last_method = "smina"
                        return s
                finally:
                    Path(lig_tmp.name).unlink(missing_ok=True)

        self.last_method = "heuristic"
        return _score_heuristic(smiles, target_protein)
