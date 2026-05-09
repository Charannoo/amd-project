# ALCHEMY — Drug Discovery Operating System

Agent-based ligand and chemistry engine for medicinal discovery (**ALCHEMY**) with **PharmaDuel** adversarial refinement. Built from the lablab.ai / AMD hackathon specification: multi-agent pipeline (targets, repurposing, de novo molecules, ADMET, literature) on **AMD MI300X (ROCm)** with optional **Qwen2.5-72B** via vLLM.

## Quick start

```bash
cd alchemy
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # set ENTREZ_EMAIL for PubMed builds; optional VLLM_HOST
# ROCm / MI300X: see setup.sh for PyTorch ROCm wheel + vLLM
python main.py              # creates demo drug FAISS index if missing, opens Gradio UI
```

## Data indices (run on GPU / workstation)

| Script | Purpose |
|--------|---------|
| `python data/build_drug_index.py` | ChEMBL → ESM2 target protein embeddings → `data/drug_index.faiss` + `drug_metadata.json` (long job). |
| `python data/build_pubmed_index.py --limit 3000` | PubMed abstracts → BioBERT embeddings → `data/pubmed_index.faiss` + `pubmed_metadata.json`. Set **ENTREZ_EMAIL** in `.env` (NCBI policy). |

Without a PubMed index, literature mode still uses **live PubMed** search; with an index, **LiteratureAgent** merges FAISS hits + live hits for RAG summaries.

## AMD Developer Cloud — fine-tune ESM2 on BindingDB (MI300X)

Provision a GPU instance from [AMD Developer Cloud](https://devcloud.amd.com/gpus?i=91d29b) (ROCm). From the ``alchemy`` directory on the instance:

| Step | Command | Notes |
|------|---------|--------|
| 1 | ``bash 1_setup.sh`` | ROCm PyTorch, deps, downloads BindingDB into ``~/alchemy_training/data/`` |
| 2 | ``python 2_prepare_data.py`` | Builds ``train.jsonl`` / ``val.jsonl`` (~10 min) |
| 3 | ``python 3_train.py`` | Fine-tunes ESM2-650M (~6–8 hr on MI300X); checkpoints under ``~/alchemy_training/checkpoints/`` |
| 4 | Copy checkpoints to your machine (or run in-repo) then ``python 4_integrate.py`` | Replaces ``models/esm2_loader.py`` with HF fine-tuned weights |

After integration, embedding size is **1280** (ESM2-650M) instead of **2560** (ESM2-3B). Remove ``data/drug_index.faiss`` and recreate with ``python main.py --demo`` or rebuild the full ChEMBL index so FAISS dimension matches ``models.esm2_loader.EMBEDDING_DIM``.

## Other services

- **Demo drug index only**: `python main.py --demo`
- **vLLM** (separate terminal, for Qwen-backed reports and optimiser JSON):

  `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-72B-Instruct --dtype float16 --port 8000 --api-key token-alchemy`

  Set `VLLM_HOST` in `.env` if not localhost.

- **REST API**: `uvicorn api.server:app --host 0.0.0.0 --port 8001`
- **vLLM helper**: `bash scripts/start_vllm.sh` (after installing vLLM on the GPU host).
- **Docker (CPU)**: `docker build -t alchemy . && docker run -p 7860:7860 alchemy` — for production GPU/ROCm, install PyTorch on the host per `setup.sh`.
- **Offline pitch**: `python main.py --offline-demo` then run Disease mode with “Chagas disease” to load `data/demo_results/chagas_pipeline.json` (replace that file with your own exported JSON after a real run).
- **Structure docking**: Set `BINDING_USE_SMINA=1`, `VINA_RECEPTOR_PDBQT`, `VINA_BOX_CENTER` (see `.env.example`). Requires a `smina` binary on `PATH`. Falls back to the RDKit heuristic if docking fails.
- **Tests**: `pip install -r requirements-dev.txt && pytest`
- **Submission**: see `SUBMISSION_CHECKLIST.md`

## Integrations

- **Open Targets** (`agents/opentargets_client.py`): repurposing and disease mode enrich ChEMBL candidates with **clinical trial / indication** overlap (`drugAndClinicalCandidates`) and optional **mechanism-of-action targets** per drug.
- **PharmaDuel defenders**: RDKit **PAINS** + **BRENK** catalogs, hERG heuristic, synthesis (SA + optional ASKCOS).

## Layout

- `config/settings.py` — ports, model names, thresholds, Entrez/Open Targets URLs
- `models/biobert_embedder.py` — shared BioBERT pooling for PubMed RAG
- `agents/` — target, repurposing, molecule, ADMET, literature, **opentargets_client**, **pharma_duel/** (attacker/defender)
- `modes/` — disease, molecule chat, pandemic, superbug, repurposing
- `ui/app.py` — Gradio UI
- `main.py` — Gradio entry + demo drug index bootstrap

## License

MIT — see `LICENSE`. Research / hackathon use only — not for clinical decisions.
