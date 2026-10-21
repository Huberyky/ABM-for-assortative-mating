"""Pure equations transcribed from ``simulation_model/model.nlogo``.

Keeping the numerical kernel free of Mesa state makes source-line review and
unit testing possible without changing the model's stochastic process.
"""

from __future__ import annotations

import math


def mate_value(
    *,
    ego_education: int,
    partner_education: int,
    partner_income: int,
    partner_age: int,
    ideal_partner_age: int,
    education_weight: float,
    income_weight: float,
    age_weight: float,
    max_education: int,
    max_income: int,
    max_age: int,
) -> float:
    """NetLogo mate value ``v_ij`` (source lines 773--809, 831--867)."""
    education = (max_education - abs(ego_education - partner_education)) / max_education
    age = (max_age - abs(partner_age - ideal_partner_age)) / max_age
    income = partner_income / max_income
    return (education**education_weight) * (age**age_weight) * (income**income_weight)


def search_probability(*, duration: int, commitment: float) -> float:
    """Probability of seeking a partner, Eq. 2 / NetLogo line 735."""
    return math.exp(-duration * commitment)


def dating_probability(
    *, age: int, age_pressure: float, value: float, duration: int, commitment: float
) -> float:
    """Probability of accepting a date, Eq. 4 / NetLogo lines 775 and 788."""
    return (1 - math.exp(-age * age_pressure * value)) * math.exp(-duration * commitment)


def marriage_probability(
    *, age: int, age_pressure: float, value: float, duration: int, commitment: float
) -> float:
    """Probability of proposing marriage, Eq. 5 / NetLogo line 943."""
    return ((1 - math.exp(-age * age_pressure * value))
            * (1 - math.exp(-duration * commitment)))


def death_probability(
    *, age: int, max_age: int, shape: float, maximum_probability: float, steps_per_year: int
) -> float:
    """Per-step death probability, Eq. 6 / NetLogo lines 990 and 1010."""
    return (((max_age - abs(age - max_age)) / max_age) ** shape
            * (maximum_probability / steps_per_year))
