# main.py
"""
ALCHEMY — Entry point: ensure demo drug index, optional vLLM, launch Gradio UI.
"""
import json
import os
import subprocess
import sys
import time
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.chdir(ROOT)


def create_demo_index():
    """Minimal deterministic FAISS index for demos when full ChEMBL index is not built."""
    import faiss
    import numpy as np

    from models.esm2_loader import EMBEDDING_DIM

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    demo_drugs = [
        {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O", "indication": "Pain/Inflammation", "chembl_id": "CHEMBL25"},
        {"name": "Metformin", "smiles": "CN(C)C(=N)NC(N)=N", "indication": "Type 2 Diabetes", "chembl_id": "CHEMBL1431"},
        {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "indication": "Pain/Inflammation", "chembl_id": "CHEMBL521"},
        {"name": "Benznidazole", "smiles": "O=C(Cn1ccnc1[N+](=O)[O-])NCc1ccccc1", "indication": "Chagas disease", "chembl_id": "CHEMBL572"},
        {"name": "Miltefosine", "smiles": "CCCCCCCCCCCCCCCCOP(=O)([O-])OCC[N+](C)(C)C", "indication": "Leishmaniasis", "chembl_id": "CHEMBL1742"},
        {"name": "Amoxicillin", "smiles": "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", "indication": "Bacterial Infection", "chembl_id": "CHEMBL1082"},
    ]

    dim = EMBEDDING_DIM

    def demo_embedding(drug: dict) -> np.ndarray:
        text = f"{drug['name']} {drug['smiles']} {drug['indication']} {drug['chembl_id']}"
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(dim).astype("float32")

    embeddings = [demo_embedding(drug) for drug in demo_drugs]
    emb_array = np.vstack(embeddings)
    faiss.normalize_L2(emb_array)
    index = faiss.IndexFlatIP(dim)
    index.add(emb_array)
    faiss.write_index(index, str(data_dir / "drug_index.faiss"))
    for drug in demo_drugs:
        drug["demo_index"] = True
        drug["embedding_note"] = "Deterministic demo fallback; build the real ChEMBL/ESM2 index for research use."

    with open(data_dir / "drug_metadata.json", "w", encoding="utf-8") as f:
        json.dump(demo_drugs, f, indent=2)
    print(f"Demo-only fallback index created with {len(demo_drugs)} drugs at {data_dir} (dim={dim})")
    print("Build the real ChEMBL/ESM2 index before using repurposing results for research.")


def check_drug_index():
    idx = ROOT / "data" / "drug_index.faiss"
    meta = ROOT / "data" / "drug_metadata.json"
    if not idx.is_file() or not meta.is_file():
        print("Drug index not found — creating deterministic demo-only fallback index.")
        print("For production: python data/build_drug_index.py on MI300X.")
        create_demo_index()


def start_vllm():
    print("Starting vLLM server for Qwen2.5-72B...")
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Qwen2.5-72B-Instruct",
        "--dtype",
        "float16",
        "--max-model-len",
        "8192",
        "--gpu-memory-utilization",
        "0.75",
        "--port",
        "8000",
        "--api-key",
        "token-alchemy",
    ]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Waiting for vLLM (60s)...")
    time.sleep(60)


if __name__ == "__main__":
    print("=" * 60)
    print("  ALCHEMY — Drug Discovery Operating System")
    print("  AMD MI300X | ROCm 6.1 | Qwen2.5-72B · ESM2-3B · MolT5")
    print("=" * 60)

    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "9.4.2")

    if "--demo" in sys.argv:
        create_demo_index()
        sys.exit(0)

    if "--offline-demo" in sys.argv:
        os.environ["ALCHEMY_OFFLINE_DEMO"] = "1"

    check_drug_index()

    if not (ROOT / "data" / "pubmed_index.faiss").is_file():
        print(
            "Tip: optional PubMed+BioBERT RAG — set ENTREZ_EMAIL in .env, then:\n"
            "     python data/build_pubmed_index.py --limit 3000"
        )

    if "--with-vllm" in sys.argv:
        start_vllm()

    print("\nLaunching ALCHEMY UI...")
    from config.settings import GRADIO_PORT, GRADIO_SHARE
    from ui.app import demo

    demo.launch(server_port=GRADIO_PORT, share=GRADIO_SHARE)
