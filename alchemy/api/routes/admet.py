from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class AdmetBody(BaseModel):
    smiles: str


@router.post("/admet")
def admet(body: AdmetBody):
    from agents.admet_agent import ADMETAgent

    return ADMETAgent().run(body.smiles)
