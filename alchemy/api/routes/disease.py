from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class DiseaseBody(BaseModel):
    disease_name: str


# Lazy deps — import heavy stack only when route is called
@router.post("/disease")
def disease_pipeline(body: DiseaseBody):
    from models.esm2_loader import load_esm2
    from modes.disease_mode import DiseaseMode

    model, alphabet = load_esm2()
    mode = DiseaseMode(model, alphabet)
    return mode.run(body.disease_name)
