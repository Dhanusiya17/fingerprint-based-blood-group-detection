from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TrainResponse(BaseModel):
	model_dir: str
	weights_path: str
	history_path: str
	labels_path: str
	test_metrics: Optional[Dict[str, float]] = Field(default=None)


class PredictResponse(BaseModel):
	predicted_class: str
	predicted_index: int
	probabilities: Dict[str, float]