from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter()


class RepurposeBody(BaseModel):
    query: str
    query_type: str = "Disease → Drug"


@router.post("/repurpose")
def repurpose(body: RepurposeBody):
    from models.esm2_loader import load_esm2
    from modes.repurposing_mode import RepurposingMode

    model, alphabet = load_esm2()
    return RepurposingMode(model, alphabet).run(body.query, body.query_type)
