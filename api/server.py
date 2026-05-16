"""FastAPI application entry."""
from fastapi import FastAPI

from api.routes import admet, disease, health, molecule, pandemic, repurpose, superbug

app = FastAPI(title="ALCHEMY API", version="1.0.0")

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(disease.router, prefix="/api/v1", tags=["disease"])
app.include_router(molecule.router, prefix="/api/v1", tags=["molecule"])
app.include_router(pandemic.router, prefix="/api/v1", tags=["pandemic"])
app.include_router(superbug.router, prefix="/api/v1", tags=["superbug"])
app.include_router(repurpose.router, prefix="/api/v1", tags=["repurpose"])
app.include_router(admet.router, prefix="/api/v1", tags=["admet"])
