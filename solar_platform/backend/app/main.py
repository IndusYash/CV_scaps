from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

app = FastAPI(
    title="Universal Solar Physics Platform API",
    description="Backend API for automated SCAPS & PV data parsing, ML model training, and predictions.",
    version="1.0.0",
)

# Enable CORS for frontend Vite server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
def startup_event():
    from app.database.db_manager import initialize_database
    initialize_database()


@app.get("/")
def read_root():
    return {"status": "online", "message": "Universal Solar Physics Platform API Server"}
