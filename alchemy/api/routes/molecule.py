from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatBody(BaseModel):
    smiles_or_name: str
    message: str
    current_smiles: Optional[str] = None


@router.post("/molecule/chat")
def molecule_chat(body: ChatBody):
    from modes.molecule_chat_mode import MoleculeChatMode

    m = MoleculeChatMode()
    if body.current_smiles:
        m.current_smiles = body.current_smiles
        from agents.admet_agent import ADMETAgent

        ad = ADMETAgent().run(body.current_smiles)
        m.admet_history = [ad]
    else:
        load = m.load_molecule(body.smiles_or_name)
        if "error" in load:
            return load
    return m.chat(body.message)
