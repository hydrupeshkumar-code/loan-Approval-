"""Transparent loan repayment calculations for user-supplied rate assumptions."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LoanCostEstimate:
    monthly_payment: float
    total_repayment: float
    total_interest: float


def estimate_loan_cost(
    principal: float,
    annual_interest_rate_percent: float,
    term_months: int,
) -> LoanCostEstimate:
    """Estimate fixed monthly payments; excludes fees, taxes, and rate changes."""
    if not math.isfinite(principal) or principal <= 0:
        raise ValueError("Principal must be a finite amount greater than zero.")
    if not math.isfinite(annual_interest_rate_percent) or annual_interest_rate_percent < 0:
        raise ValueError("Annual interest rate must be finite and non-negative.")
    if isinstance(term_months, bool) or not isinstance(term_months, int) or term_months <= 0:
        raise ValueError("Term must be a positive whole number of months.")

    monthly_rate = annual_interest_rate_percent / 1200
    if monthly_rate == 0:
        payment = principal / term_months
    else:
        factor = (1 + monthly_rate) ** term_months
        payment = principal * monthly_rate * factor / (factor - 1)

    total = payment * term_months
    return LoanCostEstimate(
        monthly_payment=payment,
        total_repayment=total,
        total_interest=total - principal,
    )
