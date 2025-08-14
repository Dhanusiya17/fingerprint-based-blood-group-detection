from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import json
import tensorflow as tf

from app.config import settings
from app.ml.dataset import create_image_datasets
from app.ml.models import build_cnn


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(
		optimizer=optimizer,
		loss=tf.keras.losses.SparseCategoricalCrossentropy(),
		metrics=["accuracy"],
	)


def train_and_save(
	model_name: str = "resnet50",
	output_dir: Optional[Path] = None,
) -> Dict:
	splits = create_image_datasets()
	num_classes = len(splits.class_names)
	model = build_cnn(model_name=model_name, input_shape=(settings.image_size, settings.image_size, 3), num_classes=num_classes)
	compile_model(model, settings.learning_rate)

	run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
	checkpoint_dir = (output_dir or settings.models_dir) / f"{model_name}_{run_id}"
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	weights_path = checkpoint_dir / "model.h5"
	history_path = checkpoint_dir / "history.json"
	labels_path = checkpoint_dir / "labels.json"
	metadata_path = checkpoint_dir / "metadata.json"

	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(filepath=str(weights_path), monitor="val_accuracy", save_best_only=True, save_weights_only=False),
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
	]

	history = model.fit(
		splits.train,
		validation_data=splits.val,
		epochs=settings.num_epochs,
		callbacks=callbacks,
	)

	# Save history and labels
	with open(history_path, "w") as f:
		json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f)
	with open(labels_path, "w") as f:
		json.dump({"class_names": splits.class_names}, f)
	with open(metadata_path, "w") as f:
		json.dump({
			"run_id": run_id,
			"model_name": model_name,
			"image_size": settings.image_size,
			"batch_size": settings.batch_size,
			"num_epochs": settings.num_epochs,
			"learning_rate": settings.learning_rate,
			"class_names": splits.class_names,
		}, f)

	# Evaluate on test
	test_metrics = {}
	if splits.test is not None:
		test_metrics = model.evaluate(splits.test, return_dict=True)

	return {
		"model_dir": str(checkpoint_dir),
		"weights_path": str(weights_path),
		"history_path": str(history_path),
		"labels_path": str(labels_path),
		"test_metrics": test_metrics,
	}