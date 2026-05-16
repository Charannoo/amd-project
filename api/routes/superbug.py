from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class SuperbugBody(BaseModel):
    bacteria: str
    resistance_mechanism: str


@router.post("/superbug")
def superbug(body: SuperbugBody):
    from models.esm2_loader import load_esm2
    from modes.superbug_mode import SuperbugMode

    model, alphabet = load_esm2()
    return SuperbugMode(model, alphabet).run(body.bacteria, body.resistance_mechanism)
