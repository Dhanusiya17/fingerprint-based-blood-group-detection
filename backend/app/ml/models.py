from __future__ import annotations

from typing import Literal

import tensorflow as tf


def build_cnn(
	model_name: Literal["resnet50", "vgg16", "mobilenet_v2", "lenet"] = "resnet50",
	input_shape: tuple[int, int, int] = (224, 224, 3),
	num_classes: int = 8,
	train_base: bool = False,
) -> tf.keras.Model:
	if model_name == "lenet":
		model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=input_shape),
			tf.keras.layers.Conv2D(32, 5, activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Conv2D(64, 5, activation="relu"),
			tf.keras.layers.MaxPooling2D(),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(256, activation="relu"),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(num_classes, activation="softmax"),
		])
		return model

	inputs = tf.keras.Input(shape=input_shape)

	if model_name == "resnet50":
		base = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
	elif model_name == "vgg16":
		base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
	elif model_name == "mobilenet_v2":
		base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)
	else:
		raise ValueError(f"Unknown model_name {model_name}")

	base.trainable = train_base
	x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
	x = tf.keras.layers.Dropout(0.3)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs)
	return model