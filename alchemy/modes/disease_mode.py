# modes/disease_mode.py
"""Mode 1: disease name → targets, repurposing, novel generation, PharmaDuel."""
from agents.admet_agent import ADMETAgent
from agents.literature_agent import LiteratureAgent
from agents.molecule_agent import get_shared_molecule_agent
from agents.opentargets_client import enrich_candidates_with_opentargets
from agents.pharma_duel.pharma_duel import PharmaDuel
from agents.repurposing_agent import RepurposingAgent
from agents.target_agent import TargetAgent
from modes.async_utils import run_coro_sync
from modes.offline_demo import load_offline_disease_result
from models.qwen_client import qwen_chat


class DiseaseMode:
    def __init__(self, esm2_model, esm2_alphabet):
        self.target_agent = TargetAgent(esm2_model, esm2_alphabet)
        self.repurposing_agent = RepurposingAgent()
        self._molecule_agent = None
        self.admet_agent = ADMETAgent()
        self.literature_agent = LiteratureAgent()
        self.pharma_duel = PharmaDuel(esm2_model, esm2_alphabet)

    @property
    def molecule_agent(self):
        if self._molecule_agent is None:
            self._molecule_agent = get_shared_molecule_agent()
        return self._molecule_agent

    def run(self, disease_name: str, progress_callback=None) -> dict:
        print(f"\n[DiseaseMode] Starting pipeline for: {disease_name}")
        cached = load_offline_disease_result(disease_name)
        if cached is not None:
            print("[DiseaseMode] Using ALCHEMY_OFFLINE_DEMO cached result.")
            if progress_callback:
                progress_callback("Loaded offline demo cache.", 100)
            return cached

        steps = {}

        if progress_callback:
            progress_callback("Identifying protein targets...", 10)
        target_result = self.target_agent.run(disease_name)
        steps["targets"] = target_result

        if not target_result.get("proteins"):
            return {"error": f"No protein targets found for {disease_name}"}

        primary_target = target_result["primary_target"]

        if progress_callback:
            progress_callback("Searching FDA drug database...", 25)
        repurposing_result = self.repurposing_agent.run(target_result)
        if repurposing_result.get("candidates"):
            repurposing_result = {
                **repurposing_result,
                "candidates": enrich_candidates_with_opentargets(
                    repurposing_result["candidates"],
                    disease_name=disease_name,
                    drug_name=None,
                ),
            }
        steps["repurposing"] = repurposing_result

        if progress_callback:
            progress_callback("Generating novel drug candidates...", 45)
        novel_result = self.molecule_agent.run(primary_target["name"], num_candidates=5)
        steps["novel"] = novel_result

        all_candidates = []
        for drug in repurposing_result.get("candidates", [])[:5]:
            if drug.get("smiles"):
                all_candidates.append(drug["smiles"])
        all_candidates.extend(novel_result.get("novel_candidates", []))

        if not all_candidates:
            aspirin = "CC(=O)Oc1ccccc1C(=O)O"
            all_candidates = [aspirin]

        if progress_callback:
            progress_callback("Running PharmaDuel adversarial battle...", 60)
        battle_result = run_coro_sync(self.pharma_duel.run(all_candidates, primary_target))
        steps["pharma_duel"] = battle_result

        if progress_callback:
            progress_callback("Computing final ADMET profiles...", 80)
        survivors = battle_result.get("survivors", [])
        for s in survivors:
            if s.get("final_smiles"):
                s["admet"] = self.admet_agent.run(s["final_smiles"])

        if progress_callback:
            progress_callback("Searching PubMed evidence...", 90)
        lit_result = self.literature_agent.search(disease_name, primary_target["name"])
        steps["literature"] = lit_result

        if progress_callback:
            progress_callback("Generating hypothesis report...", 95)
        report = self._generate_report(disease_name, steps)

        if progress_callback:
            progress_callback("Complete!", 100)

        return {
            "disease": disease_name,
            "primary_target": primary_target["name"],
            "survivors": survivors,
            "eliminated": battle_result.get("eliminated", []),
            "literature": lit_result,
            "report": report,
            "pipeline_steps": steps,
        }

    def _generate_report(self, disease_name: str, steps: dict) -> str:
        survivors = steps.get("pharma_duel", {}).get("survivors", [])
        literature = steps.get("literature", {}).get("papers", [])

        system = "You are an expert pharmaceutical scientist. Write a concise drug discovery hypothesis report."
        user = f"""Disease: {disease_name}
Primary target: {steps['targets'].get('primary_target', {}).get('name', 'Unknown')}
Battle-hardened survivors: {len(survivors)} molecules survived PharmaDuel
Top survivor SMILES: {survivors[0].get('final_smiles', 'N/A') if survivors else 'None'}
Top survivor binding score: {survivors[0].get('best_binding_score', 'N/A') if survivors else 'None'}
Supporting literature: {len(literature)} PubMed papers found

Write a 3-paragraph hypothesis report:
1. Disease context
2. Candidates and mechanism
3. Next steps"""

        return qwen_chat(system, user, max_tokens=600)
