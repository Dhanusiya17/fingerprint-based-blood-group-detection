from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.services.routes import router as api_router

app = FastAPI(title="Fingerprint Blood Group Detection API", version="0.1.0")

# CORS (adjust origins as needed)
app.add_middleware(	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Static files
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

app.include_router(api_router)

@app.get("/health")
def health_check():
	return {"status": "ok"}

@app.get("/version")
def version():
	return {"version": app.version}