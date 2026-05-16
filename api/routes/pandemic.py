from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class PandemicBody(BaseModel):
    genome_sequence: str


@router.post("/pandemic")
def pandemic(body: PandemicBody):
    from models.esm2_loader import load_esm2
    from modes.pandemic_mode import PandemicMode

    model, alphabet = load_esm2()
    return PandemicMode(model, alphabet).run(body.genome_sequence)
