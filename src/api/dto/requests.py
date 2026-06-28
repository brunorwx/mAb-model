from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    exp: int | None = Field(None, description="Optional experiment identifier")
    features: dict[str, float] = Field(
        ..., description="Numeric feature values for the prediction request"
    )
