"""FastAPI entry point for the recommendation experience and analytics engine."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import RecommendationService

app = FastAPI(title="MovieMingle Recommender API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Rating(BaseModel):
    item_id: int
    rating: float = Field(ge=1, le=5)


class NewUserRequest(BaseModel):
    ratings: list[Rating]
    model: str = "item_cf"
    n: int = Field(default=10, ge=1, le=30)


@lru_cache
def service() -> RecommendationService:
    return RecommendationService()


def client_error(call):
    try:
        return call()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/", tags=["service"])
def root():
    """Point local developers to the UI and API documentation."""
    return {
        "message": "MovieMingle Recommender API is active.",
        "frontend": "http://localhost:5173",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ready",
        "models": [
            "svd",
            "item_cf_cosine",
            "item_cf_adj",
            "user_cf_pearson",
            "user_cf_cosine",
            "popularity",
        ],
    }


@app.get("/api/users/{user_id}/history")
def user_history(user_id: int):
    return client_error(lambda: {"items": service().history(user_id)})


@app.get("/api/recommend/{user_id}")
def recommend(user_id: int, model: str = "svd", n: int = Query(default=10, ge=1, le=30)):
    return client_error(lambda: {"items": service().recommend(user_id, model, n)})


@app.post("/api/recommend/new-user")
def new_user(request: NewUserRequest):
    return client_error(lambda: {"items": service().new_user([item.model_dump() for item in request.ratings], request.model, request.n)})


@app.get("/api/movies")
def movies(query: str = "", limit: int = Query(default=20, ge=1, le=50)):
    return {"items": service().search_movies(query, limit)}


@app.get("/api/metrics")
def metrics():
    return {"items": service().metrics()}


@app.get("/api/analytics/summary")
def analytics_summary():
    """Return aggregated dataset metrics, rating distributions, and top movies."""
    return service().analytics_summary()


@app.get("/api/export/ratings")
def export_ratings():
    """Download the complete MovieLens 100K ratings dataset as CSV."""
    content = service().export_ratings_csv()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="movielens_100k_ratings.csv"'},
    )


@app.get("/api/export/recommendations")
def export_recommendations(user_id: int = Query(default=1), model: str = Query(default="svd"), n: int = Query(default=10)):
    """Download current user recommendations as CSV."""
    content = client_error(lambda: service().export_recommendations_csv(user_id, model, n))
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="recommendations_user_{user_id}_{model}.csv"'},
    )


@app.get("/api/export/metrics")
def export_metrics():
    """Download model evaluation metrics as CSV."""
    content = service().export_metrics_csv()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="model_evaluation_metrics.csv"'},
    )


@app.get("/api/export/analytics")
def export_analytics():
    """Download aggregated dataset analytics summary as CSV."""
    content = service().export_analytics_csv()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="dataset_analytics_summary.csv"'},
    )


frontend = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
