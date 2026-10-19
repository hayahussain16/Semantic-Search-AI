from typing import Literal
from pydantic import BaseModel, Field


class CustomerAnalysis(BaseModel):
    sentiment: Literal["positive", "neutral", "negative"]
    churn_risk: float = Field(ge=0, le=1)
    reason: str
    recommended_action: str

from typing import Optional
from pydantic import BaseModel, Field


class Metric(BaseModel):
    name: str
    value: Optional[float] = None
    unit: Optional[str] = None


class DocumentAnalysis(BaseModel):
    document_type: str
    company: Optional[str] = None
    reporting_period: Optional[str] = None
    summary: str
    key_points: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    key_metrics: list[Metric] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)