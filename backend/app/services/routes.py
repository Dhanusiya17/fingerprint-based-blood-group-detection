from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.config import settings
from app.schemas import PredictResponse, TrainResponse

router = APIRouter(prefix="/api")


@router.get("/classes")
async def get_classes():
	from app.ml.dataset import discover_class_names
	return {"class_names": discover_class_names(settings.data_root)}


@router.get("/models")
async def get_models():
	return {"available": ["resnet50", "vgg16", "mobilenet_v2", "lenet"]}


@router.get("/config")
async def get_config():
	return {
		"data_root": str(settings.data_root),
		"models_dir": str(settings.models_dir),
		"image_size": settings.image_size,
		"batch_size": settings.batch_size,
		"num_epochs": settings.num_epochs,
		"learning_rate": settings.learning_rate,
	}


@router.get("/runs")
async def list_runs():
	root = settings.models_dir
	root.mkdir(parents=True, exist_ok=True)
	runs = []
	for p in root.iterdir():
		if p.is_dir():
			runs.append({"name": p.name, "mtime": p.stat().st_mtime})
	runs = sorted(runs, key=lambda r: r["mtime"], reverse=True)
	return {"runs": runs}


@router.get("/runs/{run_name}/history")
async def get_run_history(run_name: str):
	path = settings.models_dir / run_name / "history.json"
	if not path.exists():
		raise HTTPException(status_code=404, detail="history not found")
	import json
	return {"history": json.loads(path.read_text())}


@router.post("/train", response_model=TrainResponse)
async def train(model: str = Query("resnet50", enum=["resnet50", "vgg16", "mobilenet_v2", "lenet"])):
	# Lazy import to avoid heavy deps during app startup
	from app.ml.train import train_and_save
	result = train_and_save(model_name=model)
	return result


@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), model_dir: Optional[str] = None):
	if file.content_type not in ("image/jpeg", "image/png", "image/bmp"):
		raise HTTPException(status_code=400, detail="Unsupported file type")
	image_bytes = await file.read()
	dir_path = Path(model_dir) if model_dir else Path(settings.models_dir)
	# If directory contains multiple trained runs, pick the latest
	if not dir_path.exists():
		raise HTTPException(status_code=400, detail="Model directory does not exist")
	if (dir_path / "model.h5").exists():
		chosen_dir = dir_path
	else:
		runs = sorted([p for p in dir_path.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
		if not runs:
			raise HTTPException(status_code=400, detail="No trained model found")
		chosen_dir = runs[0]
	from app.ml.infer import predict_image_bytes
	return predict_image_bytes(image_bytes, chosen_dir)