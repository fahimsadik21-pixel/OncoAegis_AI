from fastapi import APIRouter
from app.specialists.model_router import model_router

router = APIRouter(
    prefix="/routing",
    tags=["Model Routing"]
)


@router.post("/select-model")
def select_model(
    modality: str,
    organ: str,
    task: str
):
    result = model_router.route(
        modality=modality,
        organ=organ,
        task=task
    )

    return result