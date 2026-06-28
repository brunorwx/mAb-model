from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class PredictResponse(BaseModel):
    exp: int | None
    titer: float
    details: str | None = None
