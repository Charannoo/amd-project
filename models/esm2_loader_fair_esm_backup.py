# models/esm2_loader.py
"""ESM2-3B via fair-esm for protein embeddings (hackathon default).

After fine-tuning with ``3_train.py`` + ``4_integrate.py``, this module is replaced
with a Hugging Face ESM2-650M binding predictor; ``EMBEDDING_DIM`` becomes 1280.
Rebuild ``data/drug_index.faiss`` so FAISS dimension matches query embeddings.
"""
import torch
import esm

# Mean-pooled layer-36 representation length for esm2_t36_3B_UR50D
EMBEDDING_DIM = 2560


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_esm2():
    """Load ESM2-3B on AMD MI300X via ROCm, or CPU for dev."""
    dev = _device()
    print(f"Loading ESM2-3B on {dev}...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.eval().to(dev)
    if dev.type == "cuda":
        print(f"ESM2-3B loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    else:
        print("ESM2-3B loaded on CPU (slow but works for demos).")
    return model, alphabet


def get_embedding(sequence: str, model, alphabet) -> list:
    """Get ESM2 embedding for a protein sequence. Returns 2560-dim vector."""
    dev = next(model.parameters()).device
    batch_converter = alphabet.get_batch_converter()
    sequence = sequence[:1022]
    _, _, batch_tokens = batch_converter([("seq", sequence)])
    batch_tokens = batch_tokens.to(dev)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[36])
    embedding = results["representations"][36][0, 1 : len(sequence) + 1].mean(0)
    return embedding.cpu().numpy().tolist()
