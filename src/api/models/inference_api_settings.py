from pydantic import BaseModel, Field
from typing import Optional


class InferenceApiSettings(BaseModel):
    psg_config: Optional[str] = Field(default=None)
    model_pertrained_name_or_path: Optional[str] = Field(default=None)
    scores_threshold: Optional[float] = Field(default=None)
