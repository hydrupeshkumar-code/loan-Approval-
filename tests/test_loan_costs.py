from __future__ import annotations

import pytest

from src.loan_approval.loan_costs import estimate_loan_cost


def test_estimates_fixed_rate_loan_repayment() -> None:
    estimate = estimate_loan_cost(1_000_000, 12, 12)

    assert estimate.monthly_payment == pytest.approx(88_848.7887, rel=1e-8)
    assert estimate.total_repayment == pytest.approx(1_066_185.464, rel=1e-8)
    assert estimate.total_interest == pytest.approx(66_185.464, rel=1e-8)


def test_zero_interest_rate_splits_principal_evenly() -> None:
    estimate = estimate_loan_cost(120_000, 0, 12)

    assert estimate.monthly_payment == pytest.approx(10_000)
    assert estimate.total_repayment == pytest.approx(120_000)
    assert estimate.total_interest == pytest.approx(0)


@pytest.mark.parametrize(
    ("principal", "rate", "term"),
    [
        (0, 10, 12),
        (-1, 10, 12),
        (100, -1, 12),
        (100, 10, 0),
        (100, 10, 1.5),
        (float("inf"), 10, 12),
    ],
)
def test_rejects_invalid_loan_inputs(principal: float, rate: float, term: int) -> None:
    with pytest.raises(ValueError):
        estimate_loan_cost(principal, rate, term)
