from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from app.config import settings
from app.ml.dataset import discover_class_names


def _resolve_model_path(path: Path) -> Path:
	if path.is_file() and path.suffix.lower() == ".h5":
		return path
	if path.is_dir():
		default = path / "model.h5"
		if default.exists():
			return default
		# Fallback: pick first .h5
		candidates = sorted(path.glob("*.h5"))
		if candidates:
			return candidates[0]
	raise FileNotFoundError(f"Could not find a .h5 model file at {path}")


def load_model_bundle(model_path_or_dir: Path) -> Tuple[tf.keras.Model, List[str]]:
	model_path = _resolve_model_path(model_path_or_dir)
	labels_path = model_path.parent / "labels.json"
	model = tf.keras.models.load_model(model_path)
	class_names: List[str]
	if labels_path.exists():
		class_names = json.loads(labels_path.read_text()).get("class_names", [])
		if not class_names:
			class_names = discover_class_names(settings.data_root)
	else:
		# Fallback: derive class names from dataset directory
		class_names = discover_class_names(settings.data_root)
	return model, class_names


def preprocess_image(img: tf.Tensor, image_size: int) -> tf.Tensor:
	img = tf.image.resize(img, (image_size, image_size))
	img = tf.cast(img, tf.float32) / 255.0
	return img


def load_image_from_bytes(image_bytes: bytes) -> tf.Tensor:
	image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
	return image


def predict_image_bytes(image_bytes: bytes, model_path_or_dir: Path, image_size: int | None = None) -> Dict:
	model, class_names = load_model_bundle(model_path_or_dir)
	img = load_image_from_bytes(image_bytes)
	img = preprocess_image(img, image_size or settings.image_size)
	batch = tf.expand_dims(img, axis=0)
	probs = tf.squeeze(model.predict(batch), axis=0).numpy().astype(float)
	pred_idx = int(np.argmax(probs))
	return {
		"predicted_class": class_names[pred_idx],
		"predicted_index": pred_idx,
		"probabilities": {cls: float(probs[i]) for i, cls in enumerate(class_names)},
	}


def predict_image_path(image_path: Path, model_path_or_dir: Path, image_size: int | None = None) -> Dict:
	return predict_image_bytes(Path(image_path).read_bytes(), model_path_or_dir, image_size)