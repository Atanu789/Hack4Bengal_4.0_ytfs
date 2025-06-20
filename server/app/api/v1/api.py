# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import youtube

api_router = APIRouter()

api_router.include_router(
    youtube.router,
    prefix="/youtube",
    tags=["YouTube"]
)