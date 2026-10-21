from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Mapping


def _netlogo_value(value: Any) -> Any:
    """Convert values as stored in the original experiment-design CSV files."""
    if isinstance(value, str):
        text = value.strip().strip('"')
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            number = float(text)
            return int(number) if number.is_integer() else number
        except ValueError:
            return text
    return value


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """NetLogo interface parameters; ages are measured in simulation steps."""

    condition_no: int = 1
    run_no: int = 1
    country: str = "BE"
    ws_f: float = 0.385
    ws_m: float = 0.934
    wy_f: float = 1.201
    wy_m: float = 1.025
    wa_f: float = 10.833
    wa_m: float = 5.009
    delta: float = 0.9
    burnin_steps: int = 800
    start_year: float = 1921.0
    end_year: float = 2012.9
    steps_per_year: int = 10
    F: int = 250
    M: int = 250
    A_marr: int = 160
    A_max: int = 800
    S_max: int = 4
    Y_max: int = 5
    a_0_19_f: float = 0.3
    a_20_39_f: float = 0.35
    a_40_59_f: float = 0.25
    a_60_79_f: float = 0.1
    a_0_19_m: float = 0.3
    a_20_39_m: float = 0.35
    a_40_59_m: float = 0.25
    a_60_79_m: float = 0.1
    a_begin_s_2: int = 60
    a_begin_s_3: int = 100
    a_begin_s_4: int = 190
    a_end_s_1: int = 60
    a_end_s_2: int = 160
    a_end_s_3: int = 190
    a_end_s_4: int = 240
    beta_f: float = 0.015
    beta_m: float = 0.015
    sigma_f: float = 0.003
    sigma_m: float = 0.0015
    alpha_a_f: int = 25
    alpha_a_m: int = 240
    d_f: float = 0.1
    d_m: float = 0.1
    wd_f: float = 6.0
    wd_m: float = 6.0
    write_to_file: bool = True
    seed_to_use: str = "random"
    own_seed: int = 42

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], **overrides: Any) -> "ModelConfig":
        allowed = {field.name for field in fields(cls)}
        values = {key: _netlogo_value(value) for key, value in row.items() if key in allowed}
        values.update(overrides)
        return cls(**values)

    def with_run(self, run_no: int, seed: int) -> "ModelConfig":
        return replace(self, run_no=run_no, own_seed=seed, seed_to_use="own")

    def validate(self) -> None:
        if not 0 <= self.delta <= 1:
            raise ValueError("delta must be in [0, 1]")
        if min(self.F, self.M, self.steps_per_year, self.A_max, self.A_marr) <= 0:
            raise ValueError("population sizes, steps_per_year and A_max must be positive")
        if self.A_marr >= self.A_max:
            raise ValueError("A_marr must be smaller than A_max")
        if self.S_max != 4 or self.Y_max != 5:
            raise ValueError("the source equations require S_max=4 and Y_max=5")
        if self.seed_to_use not in {"own", "random"}:
            raise ValueError("seed_to_use must be 'own' or 'random'")
        if self.burnin_steps < 0 or self.end_year < self.start_year:
            raise ValueError("burn-in must be nonnegative and end_year must not precede start_year")
        horizon_steps = (self.end_year - self.start_year) * self.steps_per_year
        if abs(horizon_steps - round(horizon_steps)) > 1e-8:
            raise ValueError("end_year must lie on the steps_per_year time grid")
        weights = (self.ws_f, self.ws_m, self.wy_f, self.wy_m, self.wa_f, self.wa_m)
        rates = (self.beta_f, self.beta_m, self.sigma_f, self.sigma_m,
                 self.d_f, self.d_m, self.wd_f, self.wd_m)
        if min(weights) < 0 or min(rates) < 0:
            raise ValueError("preference weights, rates and mortality shapes must be nonnegative")
        for gender, shares in {
            "female": (self.a_0_19_f, self.a_20_39_f, self.a_40_59_f, self.a_60_79_f),
            "male": (self.a_0_19_m, self.a_20_39_m, self.a_40_59_m, self.a_60_79_m),
        }.items():
            if min(shares) < 0 or sum(shares) <= 0:
                raise ValueError(f"{gender} initial-age shares must be nonnegative with positive total")
