"""
Canonical promotion economics for Nudge.

Every phase (deadweight-loss accounting, budget optimisation, dashboard ROI)
reads these constants, so dollar figures are comparable across the pipeline.

Why the cost model looks like this
----------------------------------
An earlier version charged a flat $10 per targeted customer, which made every
scenario unprofitable. That was an artefact of the assumption, not a finding:
the treatment in the Hillstrom RCT is *an email*, and an email does not cost
$10. Charging discount face value to every recipient also double-counts, since
a discount is only paid out when someone actually redeems it.

The cost of promoting one customer is therefore modelled in two parts:

    send cost            paid on every email, always
    expected redemption  discount rate x order value, paid only when a
                         treated customer converts

Both benefit and cost are expressed in contribution margin, not gross revenue.
Crediting a campaign with the full order value would overstate its return by
roughly 3x, since most of that revenue pays for the goods.

Every figure below is either measured in the data or a documented assumption
about the business wrapped around it. Change them here and the notebooks,
optimiser, and dashboard all follow.
"""

from __future__ import annotations

# ── Measured in the Hillstrom RCT ────────────────────────────────────────────

# Mean `spend` among converters.
REVENUE_PER_CONVERSION: float = 116.36

# Conversion rate among *treated* customers inside the segment a CATE model
# actually targets (~1.5-2% across the practical budget range). This drives how
# often a discount is redeemed, so it belongs in the cost, not just the benefit.
TREATED_CONVERSION_RATE: float = 0.02

# ── Business assumptions ─────────────────────────────────────────────────────

# Marginal cost of sending one email. Generous: commercial ESPs bill far less
# per send, but this absorbs list hygiene and creative amortisation.
EMAIL_SEND_COST: float = 0.10

# Face value of the offer, as a fraction of order value.
DISCOUNT_RATE: float = 0.15

# Fraction of revenue left after COGS. 0.35 is a conservative e-commerce figure;
# set to 1.0 to report on gross revenue instead.
GROSS_MARGIN: float = 0.35


def margin_per_conversion() -> float:
    """Contribution margin earned on one incremental conversion."""
    return REVENUE_PER_CONVERSION * GROSS_MARGIN


def promo_cost() -> float:
    """
    Expected cost of promoting one customer.

    Send cost is certain; the discount is only paid when a treated customer
    converts, so it enters weighted by the treated conversion rate.
    """
    expected_redemption = TREATED_CONVERSION_RATE * DISCOUNT_RATE * REVENUE_PER_CONVERSION
    return EMAIL_SEND_COST + expected_redemption


# Convenience aliases so callers can import a value rather than call a function.
PROMO_COST: float = promo_cost()
MARGIN_PER_CONVERSION: float = margin_per_conversion()

# Budget levels for the Phase 5A sweep and the dashboard slider bounds.
# Sized against this cost model: the largest scenario reaches most of the 64K
# base, so the sweep spans the profitable range *and* the point past which
# targeting stops paying for itself.
BUDGET_SCENARIOS: list[float] = [500, 1_000, 2_500, 5_000, 10_000, 25_000]


def break_even_uplift() -> float:
    """
    Incremental conversion rate a targeted segment must clear to pay for itself.

    Below this, promoting the segment destroys margin no matter how well the
    model ranks within it.
    """
    return promo_cost() / margin_per_conversion()
