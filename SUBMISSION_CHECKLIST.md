# Hackathon submission checklist (AMD / lablab.ai)

Use this as a copy-paste task list. Code deliverables in this repo are marked ✅ where implemented.

## Run + record

- [ ] **Hardware path**: On MI300X, `bash setup.sh`, then `python main.py` (optionally `python main.py --with-vllm` or `scripts/start_vllm.sh` in a second shell).
- [ ] **Indices**: Run `python data/build_drug_index.py` and `python data/build_pubmed_index.py` (set `ENTREZ_EMAIL`) before the final demo.
- [ ] **Structure docking (optional)**: Install [smina](https://sourceforge.net/projects/smina/), set `BINDING_USE_SMINA=1`, `VINA_RECEPTOR_PDBQT`, `VINA_BOX_CENTER`, `VINA_BOX_SIZE` in `.env`.
- [ ] **Pitch fallback**: `python main.py --offline-demo` loads `data/demo_results/chagas_pipeline.json` for Disease mode when the query mentions Chagas (no live APIs).
- [ ] **2-minute video**: Screen record Disease mode + Molecule Chat + one line on MI300X / PharmaDuel.
- [ ] **Public GitHub**: Push repo; ensure `LICENSE` is visible.
- [ ] **Hugging Face Space** (if required): Add `Dockerfile` or Space config; set secrets for `VLLM_HOST` if API is external; document CPU vs GPU limits.
- [ ] **lablab.ai form**: Link repo + Space + video; describe agents (Target, Repurposing, MolT5, PharmaDuel, Open Targets, PubMed RAG).
- [ ] **Social**: 3 posts with `#AMDDevHackathon` (or event tags).

## Repo hygiene

- [ ] `pip install -r requirements-dev.txt && pytest` on CI or locally (some tests skip without `rdkit`/`esm`).
- [ ] Replace `data/demo_results/chagas_pipeline.json` with a JSON export from a real successful run for the pitch.

## Disclaimer

All scores and candidates are **research / demo only** — not for clinical or regulatory use.
