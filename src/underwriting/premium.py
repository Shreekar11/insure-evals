"""
Illustrative premium calculator.

IMPORTANT: All constants (base_rate, multipliers, cost-of-failure) are PLACEHOLDERS
pending real actuarial loss data. This is the structure of AI underwriting, not a
statistically calibrated pricing model. The formula shape is defensible; the numbers
are not. Every premium output must be labelled as illustrative.
"""
from dataclasses import dataclass


# ── Placeholder constants — label clearly in any UI ───────────────────────
BASE_ANNUAL_PREMIUM = 5_000       # USD; placeholder
MEDICAL_DOMAIN_MULTIPLIER = 1.5   # higher stakes domain; placeholder
COST_OF_FAILURE_PER_CLAIM = 50_000  # illustrative cost per harmful incident; placeholder


@dataclass
class PremiumResult:
    annual_premium_usd: float
    risk_class: str             # Low / Medium / High / Critical
    hallucination_mult: float
    bias_mult: float
    safety_mult: float
    volume_mult: float
    domain_mult: float
    sensitivity: dict           # {"cut_hal_30pct_saves": float, ...}
    caveat: str


def _rate_mult(rate: float, axis: str) -> float:
    """
    Convert a failure rate (0.0–1.0) to a risk multiplier.
    Scale is illustrative — placeholder pending loss data.
    """
    if rate < 0.05:
        return 1.0
    elif rate < 0.15:
        return 1.3
    elif rate < 0.30:
        return 1.7
    elif rate < 0.50:
        return 2.5
    else:
        return 4.0


def _risk_class(total_mult: float) -> str:
    if total_mult < 1.5:
        return "Low"
    elif total_mult < 2.5:
        return "Medium"
    elif total_mult < 4.0:
        return "High"
    return "Critical"


def calculate(
    hallucination_rate: float,
    bias_rate: float,
    safety_rate: float,
    deployment_volume: str = "medium",   # low / medium / high / enterprise
    domain: str = "medical",
) -> PremiumResult:
    """
    Compute the illustrative annual premium.

    Parameters
    ----------
    hallucination_rate : float  0.0–1.0 fraction flagged
    bias_rate          : float  0.0–1.0
    safety_rate        : float  0.0–1.0
    deployment_volume  : str    low / medium / high / enterprise
    domain             : str    currently only 'medical' is tuned
    """
    volume_mults = {"low": 0.7, "medium": 1.0, "high": 1.5, "enterprise": 2.5}
    volume_mult = volume_mults.get(deployment_volume, 1.0)
    domain_mult = MEDICAL_DOMAIN_MULTIPLIER if domain == "medical" else 1.0

    hal_mult = _rate_mult(hallucination_rate, "hallucination")
    bias_mult = _rate_mult(bias_rate, "bias")
    safety_mult = _rate_mult(safety_rate, "safety")

    total_mult = hal_mult * bias_mult * safety_mult * volume_mult * domain_mult
    premium = BASE_ANNUAL_PREMIUM * total_mult

    # Sensitivity: what saving 30% on hallucination rate would do
    hal_30 = _rate_mult(hallucination_rate * 0.70, "hallucination")
    premium_30 = BASE_ANNUAL_PREMIUM * hal_30 * bias_mult * safety_mult * volume_mult * domain_mult
    hal_saving = premium - premium_30

    caveat = (
        "⚠️ Illustrative only. All multipliers and base rates are placeholders "
        "pending real actuarial loss data. Premium reflects the structure of AI "
        "underwriting, not a statistically calibrated price. "
        "Probe N is small (~20–30 per axis) — directional only."
    )

    return PremiumResult(
        annual_premium_usd=round(premium, 2),
        risk_class=_risk_class(total_mult),
        hallucination_mult=round(hal_mult, 2),
        bias_mult=round(bias_mult, 2),
        safety_mult=round(safety_mult, 2),
        volume_mult=round(volume_mult, 2),
        domain_mult=round(domain_mult, 2),
        sensitivity={"cut_hallucination_30pct_saves_usd": round(hal_saving, 2)},
        caveat=caveat,
    )
