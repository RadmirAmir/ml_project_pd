"""Pydantic schemas for the scoring API.

The payload mirrors the assignment example (eight core fields) but accepts the
full 23-column UCI record as well: the remaining fields are optional and get
documented neutral defaults, so a client can send either a compact request or a
raw dataset row.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ClientData(BaseModel):
    """Raw features of one credit card client."""

    model_config = ConfigDict(
        extra="ignore",
        json_schema_extra={
            "examples": [
                {
                    "LIMIT_BAL": 20000,
                    "SEX": 2,
                    "EDUCATION": 2,
                    "MARRIAGE": 1,
                    "AGE": 24,
                    "PAY_0": 2,
                    "BILL_AMT1": 3913,
                    "PAY_AMT1": 0,
                }
            ]
        },
    )

    # --- core fields (required, identical to the assignment example) ------
    LIMIT_BAL: float = Field(..., ge=0, description="Amount of the given credit (NT dollar)")
    SEX: int = Field(..., ge=0, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(..., ge=0, le=6, description="1 = graduate, 2 = university, 3 = high school, 4 = other")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1 = married, 2 = single, 3 = other")
    AGE: int = Field(..., ge=18, le=100, description="Age in years")
    PAY_0: int = Field(..., ge=-2, le=9, description="Repayment status in September 2005")
    BILL_AMT1: float = Field(0.0, description="Bill statement amount in September 2005")
    PAY_AMT1: float = Field(0.0, description="Amount paid in September 2005")

    # --- optional history (defaults keep single-row payloads valid) -------
    PAY_2: int | None = Field(None, ge=-2, le=9)
    PAY_3: int | None = Field(None, ge=-2, le=9)
    PAY_4: int | None = Field(None, ge=-2, le=9)
    PAY_5: int | None = Field(None, ge=-2, le=9)
    PAY_6: int | None = Field(None, ge=-2, le=9)
    BILL_AMT2: float | None = Field(None, description="Bill statement amount in August 2005")
    BILL_AMT3: float | None = None
    BILL_AMT4: float | None = None
    BILL_AMT5: float | None = None
    BILL_AMT6: float | None = None
    PAY_AMT2: float | None = None
    PAY_AMT3: float | None = None
    PAY_AMT4: float | None = None
    PAY_AMT5: float | None = None
    PAY_AMT6: float | None = None

    @field_validator("BILL_AMT1", "PAY_AMT1", mode="before")
    @classmethod
    def _allow_none_core_amounts(cls, value: Any) -> Any:
        return 0.0 if value is None else value

    def to_feature_dict(self) -> dict:
        """Drop unset history fields so the feature builder applies defaults."""
        return {key: value for key, value in self.model_dump().items() if value is not None}


class PredictionResponse(BaseModel):
    """Single prediction returned by ``POST /predict``."""

    default_prediction: int = Field(..., description="1 = predicted default, 0 = no default")
    default_probability: float = Field(..., ge=0, le=1, description="Estimated probability of default (PD)")
    risk_band: str = Field(..., description="low / medium / high / critical")
    threshold: float = Field(..., description="Decision threshold applied to the PD")
    model_name: str | None = Field(None, description="Classifier used behind the model")


class BatchPredictionRequest(BaseModel):
    """Batch scoring payload."""

    records: list[ClientData] = Field(..., min_length=1, max_length=5000)


class BatchPredictionResponse(BaseModel):
    """Batch scoring response."""

    predictions: list[PredictionResponse]
    n_records: int
    mean_probability: float
    n_flagged: int


class HealthResponse(BaseModel):
    """Liveness probe payload."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None


class ModelInfoResponse(BaseModel):
    """Model metadata exposed to consumers and monitoring jobs."""

    model_name: str | None
    threshold: float
    trained_at: str | None
    dataset_hash: str | None
    sklearn_version: str | None
    features: dict[str, list[str]]
    test_metrics: dict[str, Any]
