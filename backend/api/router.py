from fastapi import APIRouter
from backend.api.v1.endpoints import experiments, providers, datasets

api_router = APIRouter()

api_router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
api_router.include_router(providers.router, prefix="/providers", tags=["Providers"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
