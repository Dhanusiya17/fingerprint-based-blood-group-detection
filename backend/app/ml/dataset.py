from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

from app.config import settings


@dataclass
class DatasetSplits:
	train: tf.data.Dataset
	val: tf.data.Dataset
	test: tf.data.Dataset
	class_names: List[str]


def set_global_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def discover_class_names(data_root: Path) -> List[str]:
	class_dirs = [p.name for p in sorted(data_root.iterdir()) if p.is_dir()]
	if not class_dirs:
		raise RuntimeError(f"No class directories found under {data_root}")
	return class_dirs


def create_image_datasets(
	data_root: Path | None = None,
	image_size: int | None = None,
	batch_size: int | None = None,
	seed: int | None = None,
	train_split: float | None = None,
	val_split: float | None = None,
	test_split: float | None = None,
	augment: bool | None = None,
) -> DatasetSplits:
	root = data_root or settings.data_root
	img_size = image_size or settings.image_size
	batch = batch_size or settings.batch_size
	s = seed or settings.seed
	train_p = train_split or settings.train_split
	val_p = val_split or settings.val_split
	test_p = test_split or settings.test_split
	use_aug = settings and (augment if augment is not None else True)

	if abs((train_p + val_p + test_p) - 1.0) > 1e-6:
		raise ValueError("Splits must sum to 1.0")

	set_global_seed(s)

	class_names = discover_class_names(root)

	# Use Keras utility to load from directories with a validation split
	total_val = val_p + test_p
	if total_val <= 0.0:
		train_ds = tf.keras.preprocessing.image_dataset_from_directory(
			root, labels="inferred", image_size=(img_size, img_size), batch_size=batch,
			seed=s, validation_split=None, shuffle=True
		)
		val_ds = test_ds = None
		return DatasetSplits(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)

	# First create a train and temp split
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
		root,
		labels="inferred",
		image_size=(img_size, img_size),
		batch_size=batch,
		seed=s,
		validation_split=total_val,
		subset="training",
		shuffle=True,
	)
	temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
		root,
		labels="inferred",
		image_size=(img_size, img_size),
		batch_size=batch,
		seed=s,
		validation_split=total_val,
		subset="validation",
		shuffle=True,
	)

	# Now split temp into val and test according to proportions
	val_ratio = val_p / max(total_val, 1e-9)

	cardinality = tf.data.experimental.cardinality(temp_ds).numpy()
	val_batches = int(round(cardinality * val_ratio))
	val_ds = temp_ds.take(val_batches)
	test_ds = temp_ds.skip(val_batches)

	# Performance options
	autotune = tf.data.AUTOTUNE
	cache_and_prefetch = lambda ds: ds.cache().prefetch(buffer_size=autotune)
	train_ds = train_ds.shuffle(1000)
	val_ds = cache_and_prefetch(val_ds)
	test_ds = cache_and_prefetch(test_ds)

	# Optional augmentation (applied only to training data)
	if use_aug:
		augmenter = tf.keras.Sequential([
			tf.keras.layers.RandomFlip("horizontal"),
			tf.keras.layers.RandomRotation(0.05),
			tf.keras.layers.RandomZoom(0.1),
			tf.keras.layers.RandomContrast(0.1),
		], name="data_augmentation")
		train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=autotune)

	# Normalization layer
	normalize = tf.keras.layers.Rescaling(1.0 / 255)
	train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune).prefetch(buffer_size=autotune)
	val_ds = val_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune)
	test_ds = test_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=autotune)

	return DatasetSplits(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)