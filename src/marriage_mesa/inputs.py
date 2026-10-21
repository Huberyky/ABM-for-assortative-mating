from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def _rows(path: Path) -> np.ndarray:
    """Read NetLogo's nested numeric-list format without executing it."""
    text = path.read_text(encoding="utf-8")
    rows = [[float(x) for x in re.findall(NUMBER, match)]
            for match in re.findall(r"\[([^\[\]]+)\]", text)]
    rows = [row for row in rows if row]
    if not rows:
        raise ValueError(f"no numeric rows found in {path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged NetLogo matrix in {path}: widths={sorted(widths)}")
    return np.asarray(rows, dtype=float)


def _vector(path: Path) -> tuple[float, ...]:
    values = tuple(float(x) for x in re.findall(NUMBER, path.read_text(encoding="utf-8")))
    if not values:
        raise ValueError(f"no numeric values found in {path}")
    return values


def _check_matrix(name: str, matrix: np.ndarray, categories: int) -> None:
    if matrix.ndim != 2 or matrix.shape[1] != categories + 1:
        raise ValueError(f"{name}: expected year + {categories} probabilities, got {matrix.shape}")
    probabilities = matrix[:, 1:]
    if np.any(probabilities < 0) or np.any(probabilities.sum(axis=1) <= 0):
        raise ValueError(f"{name}: negative or zero-total probability row")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError(f"{name}: source probability rows must sum to one")


@dataclass(frozen=True, slots=True)
class CohortInputs:
    male_s: np.ndarray
    female_s: np.ndarray
    male_y_low: np.ndarray
    male_y_high: np.ndarray
    female_y_low: np.ndarray
    female_y_high: np.ndarray
    measurement_points: tuple[float, ...]

    @classmethod
    def from_netlogo(cls, input_dir: str | Path, country: str) -> "CohortInputs":
        directory = Path(input_dir)
        obj = cls(
            male_s=_rows(directory / f"{country}_input_male_s.txt"),
            female_s=_rows(directory / f"{country}_input_female_s.txt"),
            male_y_low=_rows(directory / f"{country}_input_male_y_s_low.txt"),
            male_y_high=_rows(directory / f"{country}_input_male_y_s_high.txt"),
            female_y_low=_rows(directory / f"{country}_input_female_y_s_low.txt"),
            female_y_high=_rows(directory / f"{country}_input_female_y_s_high.txt"),
            measurement_points=_vector(directory / "measurement_points.txt"),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        _check_matrix("male_s", self.male_s, 4)
        _check_matrix("female_s", self.female_s, 4)
        for name in ("male_y_low", "male_y_high", "female_y_low", "female_y_high"):
            _check_matrix(name, getattr(self, name), 5)
        matrices = (self.male_s, self.female_s, self.male_y_low, self.male_y_high,
                    self.female_y_low, self.female_y_high)
        if len({matrix.shape[0] for matrix in matrices}) != 1:
            raise ValueError("input matrices have different cohort counts")
        base_years = self.male_s[:, 0]
        if np.any(np.diff(base_years) <= 0):
            raise ValueError("cohort years must be strictly increasing")
        for matrix in matrices[1:]:
            if not np.array_equal(matrix[:, 0], base_years):
                raise ValueError("cohort-year columns differ across input matrices")

    @property
    def n_cohorts(self) -> int:
        return self.male_s.shape[0]

    def cohort_index(self, year: float) -> int:
        years = self.male_s[:, 0]
        if year < years[0]:
            return 0
        return min(int(np.searchsorted(years, year, side="right") - 1), self.n_cohorts - 1)

    @classmethod
    def from_arrays(
        cls,
        years: Iterable[float],
        male_s: np.ndarray,
        female_s: np.ndarray,
        male_y_low: np.ndarray,
        male_y_high: np.ndarray,
        female_y_low: np.ndarray,
        female_y_high: np.ndarray,
        measurement_points: Iterable[float],
    ) -> "CohortInputs":
        year_column = np.asarray(list(years), dtype=float)[:, None]
        add_year = lambda values: np.column_stack([year_column, np.asarray(values, dtype=float)])
        obj = cls(add_year(male_s), add_year(female_s), add_year(male_y_low), add_year(male_y_high),
                  add_year(female_y_low), add_year(female_y_high), tuple(measurement_points))
        obj.validate()
        return obj

