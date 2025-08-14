from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.services.routes import router as api_router

app = FastAPI(title="Fingerprint Blood Group Detection API", version="0.1.0")

# CORS (adjust origins as needed)
app.add_middleware(	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(api_router)

@app.get("/health")
def health_check():
	return {"status": "ok"}

@app.get("/version")
def version():
	return {"version": app.version}