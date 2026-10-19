import pytest
from pydantic import ValidationError

from app.schemas import CustomerAnalysis


def test_valid_analysis():
    a = CustomerAnalysis(
        sentiment="negative",
        churn_risk=0.8,
        reason="late orders",
        recommended_action="call"
    )

    assert a.churn_risk == 0.8


def test_risk_out_of_range():
    with pytest.raises(ValidationError):
        CustomerAnalysis(
            sentiment="negative",
            churn_risk=1.7,
            reason="x",
            recommended_action="y"
        )


def test_bad_sentiment():
    with pytest.raises(ValidationError):
        CustomerAnalysis(
            sentiment="angry",
            churn_risk=0.5,
            reason="x",
            recommended_action="y"
        )