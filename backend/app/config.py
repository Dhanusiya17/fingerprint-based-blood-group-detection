import os
from pathlib import Path
from typing import List


class Settings:
	project_name: str = "Fingerprint Blood Group Detection API"
	version: str = "0.1.0"
	# Data paths
	data_root: Path = Path(os.getenv("DATA_ROOT", "/workspace/dataset/dataset_blood_group"))
	models_dir: Path = Path(os.getenv("MODELS_DIR", "/workspace/backend/app/storage/models"))
	artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", "/workspace/backend/app/storage/artifacts"))
	# Training params (defaults can be tuned later)
	image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
	batch_size: int = int(os.getenv("BATCH_SIZE", "32"))
	num_epochs: int = int(os.getenv("NUM_EPOCHS", "10"))
	learning_rate: float = float(os.getenv("LEARNING_RATE", "1e-3"))
	seed: int = int(os.getenv("SEED", "42"))
	train_split: float = float(os.getenv("TRAIN_SPLIT", "0.8"))
	val_split: float = float(os.getenv("VAL_SPLIT", "0.1"))
	test_split: float = float(os.getenv("TEST_SPLIT", "0.1"))
	# Allowed image extensions
	image_extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]


settings = Settings()