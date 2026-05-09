# data/build_drug_index.py
"""
Run this ONCE on Day 1. Takes 4-6 hours on MI300X.
Builds the drug embedding index used by RepurposingAgent.

Usage: from repo root: python data/build_drug_index.py

Note: The default ``compute_esm2_embedding`` path assumes **fair-esm ESM2-3B** (alphabet batch
converter). If you ran ``4_integrate.py``, replace that helper with ``get_embedding`` from
``models.esm2_loader`` so index dimension matches ``EMBEDDING_DIM``.
"""
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import CHEMBL_API
from models.esm2_loader import load_esm2


def fetch_chembl_drugs(limit=10000):
    """Fetch approved drugs with target protein sequences from ChEMBL."""
    drugs = []
    offset = 0
    batch_size = 100

    while len(drugs) < limit:
        url = f"{CHEMBL_API}/molecule.json"
        params = {
            "molecule_type": "Small molecule",
            "max_phase": 4,
            "limit": batch_size,
            "offset": offset,
            "format": "json",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        molecules = data.get("molecules", [])
        if not molecules:
            break

        for mol in molecules:
            smiles = mol.get("molecule_structures", {})
            if smiles and smiles.get("canonical_smiles"):
                drugs.append(
                    {
                        "chembl_id": mol["molecule_chembl_id"],
                        "name": mol.get("pref_name", "Unknown"),
                        "smiles": smiles["canonical_smiles"],
                        "indication": mol.get("indication_class", "Unknown"),
                        "max_phase": mol.get("max_phase", 0),
                    }
                )
        offset += batch_size
        print(f"Fetched {len(drugs)} drugs so far...")

    return drugs[:limit]


def compute_esm2_embedding(sequence, model, alphabet):
    """Compute ESM2-3B embedding for a protein sequence."""
    dev = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
    batch_tokens = batch_tokens.to(dev)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36], return_contacts=False)
    token_representations = results["representations"][36]
    embedding = token_representations[0, 1 : len(sequence) + 1].mean(0)
    return embedding.cpu().numpy()


def embed_drug_smiles_via_target(drug, esm2_model, esm2_alphabet):
    """
    For each drug, fetch its primary target protein from ChEMBL,
    then compute ESM2 embedding of that target protein.
    """
    try:
        url = f"{CHEMBL_API}/activity.json"
        params = {"molecule_chembl_id": drug["chembl_id"], "limit": 1}
        resp = requests.get(url, params=params, timeout=10).json()
        activities = resp.get("activities", [])
        if activities:
            target_id = activities[0].get("target_chembl_id")
            if target_id:
                t_url = f"{CHEMBL_API}/target/{target_id}.json"
                t_resp = requests.get(t_url, timeout=10).json()
                components = t_resp.get("target_components", [])
                if components:
                    uniprot_id = components[0].get("accession")
                    if uniprot_id:
                        seq_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
                        seq_resp = requests.get(seq_url, timeout=10).text
                        sequence = "".join(seq_resp.split("\n")[1:])
                        sequence = sequence[:1022]
                        if sequence:
                            return compute_esm2_embedding(sequence, esm2_model, esm2_alphabet)
    except Exception:
        pass
    return None


def build_index():
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ESM2-3B model on AMD MI300X...")
    esm2_model, esm2_alphabet = load_esm2()

    print("Fetching drugs from ChEMBL...")
    drugs = fetch_chembl_drugs(10000)

    embeddings = []
    valid_drugs = []

    print("Computing embeddings (this will take several hours)...")
    for drug in tqdm(drugs):
        embedding = embed_drug_smiles_via_target(drug, esm2_model, esm2_alphabet)
        if embedding is not None:
            embeddings.append(embedding)
            valid_drugs.append(drug)

    if not embeddings:
        print("No valid embeddings — check ChEMBL/API connectivity.")
        return

    embeddings_array = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings_array)

    dim = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_array)

    print(f"Built FAISS index with {index.ntotal} drugs")
    faiss.write_index(index, str(data_dir / "drug_index.faiss"))

    with open(data_dir / "drug_metadata.json", "w", encoding="utf-8") as f:
        json.dump(valid_drugs, f, indent=2)

    print("Drug index built successfully.")
    print(f"   - {index.ntotal} drugs embedded")
    print(f"   - Saved to {data_dir / 'drug_index.faiss'}")
    print(f"   - Metadata saved to {data_dir / 'drug_metadata.json'}")


if __name__ == "__main__":
    build_index()
