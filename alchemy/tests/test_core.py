import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_binding_predictor_heuristic():
    pytest.importorskip("rdkit")
    from agents.pharma_duel.attacker.binding_predictor import BindingPredictor

    b = BindingPredictor()
    tgt = {"name": "Test", "sequence": "M" * 200}
    s = b.score("CC(=O)Oc1ccccc1C(=O)O", tgt)
    assert isinstance(s, float)
    assert s > -500


def test_offline_demo_loads_chagas():
    from modes.offline_demo import load_offline_disease_result
    import os

    os.environ["ALCHEMY_OFFLINE_DEMO"] = "1"
    r = load_offline_disease_result("Chagas disease")
    assert r is not None
    assert r.get("offline_demo") is True
    assert "survivors" in r
    del os.environ["ALCHEMY_OFFLINE_DEMO"]


def test_opentargets_graphql_mock(monkeypatch):
    from agents import opentargets_client as ot

    def fake_post(url, json=None, timeout=45, **kwargs):
        class R:
            status_code = 200

            def raise_for_status(self):
                pass

            @property
            def text(self):
                return "{}"

            def json(self):
                q = (json or {}).get("query", "")
                if "SearchDisease" in q:
                    return {"data": {"search": {"hits": [{"id": "MONDO_0004979", "name": "asthma"}]}}}
                if "ClinCand" in q:
                    return {
                        "data": {
                            "disease": {
                                "drugAndClinicalCandidates": {
                                    "count": 1,
                                    "rows": [
                                        {
                                            "maxClinicalStage": "PHASE_3",
                                            "drug": {"id": "CHEMBL25", "name": "Aspirin", "drugType": "Small molecule"},
                                        }
                                    ],
                                }
                            }
                        }
                    }
                if "Mech" in q:
                    return {
                        "data": {
                            "drug": {
                                "mechanismsOfAction": {
                                    "rows": [{"targets": [{"id": "ENSG1", "approvedSymbol": "GENE1"}]}]
                                }
                            }
                        }
                    }
                return {"data": {}}

        return R()

    monkeypatch.setattr(ot.requests, "post", fake_post)
    hits = ot.search_disease("asthma", page_size=2)
    assert hits and hits[0]["id"].startswith("MONDO")
    rows = ot.clinical_candidates_for_disease("MONDO_0004979", max_rows=5)
    assert rows and rows[0]["chembl_id"] == "CHEMBL25"


def test_target_agent_parse():
    pytest.importorskip("esm")
    from agents.target_agent import TargetAgent

    ta = TargetAgent(esm2_model=None, esm2_alphabet=None)
    data = {
        "results": [
            {
                "primaryAccession": "P0",
                "sequence": {"value": "M" + "A" * 120},
                "proteinDescription": {
                    "recommendedName": {"fullName": {"value": "Test protein"}}
                },
                "genes": [{"geneName": {"value": "TEST"}}],
            }
        ]
    }
    proteins = ta._parse_uniprot_results(data)
    assert len(proteins) == 1
    assert proteins[0]["uniprot_id"] == "P0"
