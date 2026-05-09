# agents/pharma_duel/pharma_duel.py
import asyncio

from agents.pharma_duel.attacker.binding_predictor import BindingPredictor
from agents.pharma_duel.attacker.optimiser import Optimiser
from agents.pharma_duel.defender.resistance_agent import ResistanceDefender
from agents.pharma_duel.defender.synthesis_agent import SynthesisDefender
from agents.pharma_duel.defender.toxicity_agent import ToxicityDefender
from config.settings import PHARMA_DUEL_CANDIDATES, PHARMA_DUEL_ROUNDS


class PharmaDuel:
    def __init__(self, esm2_model=None, esm2_alphabet=None, rounds: int = PHARMA_DUEL_ROUNDS):
        self.rounds = rounds
        self.binder = BindingPredictor()
        self.optimiser = Optimiser()
        self.toxicity = ToxicityDefender()
        self.resistance = ResistanceDefender(esm2_model, esm2_alphabet)
        self.synthesis = SynthesisDefender()

    async def run_defenders(self, smiles: str, target_protein: dict) -> dict:
        loop = asyncio.get_event_loop()
        tox_task = loop.run_in_executor(None, self.toxicity.evaluate, smiles)
        res_task = loop.run_in_executor(None, self.resistance.evaluate, smiles, target_protein)
        syn_task = loop.run_in_executor(None, self.synthesis.evaluate, smiles)

        tox_result, res_result, syn_result = await asyncio.gather(tox_task, res_task, syn_task)

        objections = []
        passed = 0
        total = 3

        if tox_result["passed"]:
            passed += 1
        else:
            objections.append(
                {"agent": "ToxicityDefender", "issue": tox_result["reason"], "severity": "high"}
            )

        if res_result["passed"]:
            passed += 1
        else:
            objections.append(
                {"agent": "ResistanceDefender", "issue": res_result["reason"], "severity": "medium"}
            )

        if syn_result["passed"]:
            passed += 1
        else:
            objections.append(
                {"agent": "SynthesisDefender", "issue": syn_result["reason"], "severity": "medium"}
            )

        return {
            "passed": passed,
            "total": total,
            "survival_rate": passed / total,
            "objections": objections,
            "toxicity": tox_result,
            "resistance": res_result,
            "synthesis": syn_result,
            "molecule_survives": passed >= 2,
        }

    async def battle_one_candidate(self, initial_smiles: str, target_protein: dict) -> dict:
        battle_log = []
        current_smiles = initial_smiles
        best_smiles = initial_smiles
        best_binding = -999.0

        for round_num in range(1, self.rounds + 1):
            print(f"\n  Round {round_num}/{self.rounds}")

            binding_score = self.binder.score(current_smiles, target_protein)
            print(f"    Attacker: binding score {binding_score:.2f}")

            defense_result = await self.run_defenders(current_smiles, target_protein)
            print(f"    Defender: {defense_result['passed']}/{defense_result['total']} passed")

            round_result = {
                "round": round_num,
                "smiles": current_smiles,
                "binding_score": binding_score,
                "defense": defense_result,
                "survived": defense_result["molecule_survives"],
            }
            battle_log.append(round_result)

            if defense_result["molecule_survives"]:
                if binding_score > best_binding:
                    best_binding = binding_score
                    best_smiles = current_smiles

                if round_num < self.rounds:
                    optimisation = self.optimiser.get_improvements(
                        current_smiles,
                        defense_result["objections"],
                        target_protein.get("name", "protein"),
                    )
                    if optimisation.get("modified_smiles"):
                        current_smiles = optimisation["modified_smiles"]
                        print("    Optimiser: improved molecule")
            else:
                fix_instruction = self.optimiser.address_objections(
                    current_smiles,
                    defense_result["objections"],
                    target_protein.get("name", "protein"),
                )
                if fix_instruction.get("modified_smiles"):
                    current_smiles = fix_instruction["modified_smiles"]
                    print(f"    Optimiser: fixing — {fix_instruction.get('reason', '')}")

        final_defense = battle_log[-1]["defense"]
        survived = any(r["survived"] for r in battle_log)

        return {
            "initial_smiles": initial_smiles,
            "final_smiles": best_smiles if survived else None,
            "survived": survived,
            "best_binding_score": best_binding if survived else None,
            "rounds_survived": sum(1 for r in battle_log if r["survived"]),
            "battle_log": battle_log,
            "elimination_reason": None
            if survived
            else [o["issue"] for o in final_defense.get("objections", [])],
        }

    async def run(self, candidates: list, target_protein: dict, progress_callback=None) -> dict:
        print(f"\nPharmaDuel starting — {len(candidates)} candidates, {self.rounds} rounds")
        print(f"   Target: {target_protein.get('name', 'Unknown')}")

        top_candidates = candidates[:PHARMA_DUEL_CANDIDATES]
        battle_results = []

        for i, smiles in enumerate(top_candidates):
            print(f"\n--- Battling candidate {i+1}/{len(top_candidates)} ---")
            result = await self.battle_one_candidate(smiles, target_protein)
            battle_results.append(result)
            if progress_callback:
                progress_callback(i + 1, len(top_candidates), result)

        survivors = [r for r in battle_results if r["survived"]]
        survivors.sort(key=lambda x: x.get("best_binding_score") or -999, reverse=True)

        print(f"\nPharmaDuel complete: {len(survivors)}/{len(top_candidates)} molecules survived")

        return {
            "survivors": survivors,
            "eliminated": [r for r in battle_results if not r["survived"]],
            "winner": survivors[0] if survivors else None,
            "total_rounds": self.rounds,
            "survival_rate": len(survivors) / max(1, len(top_candidates)),
            "battle_results": battle_results,
        }
