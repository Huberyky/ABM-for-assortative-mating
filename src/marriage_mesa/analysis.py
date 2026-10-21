from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


COHORT_BREAKS = [1940, 1950, 1960, 1970, 1980]


def collapse_education(values: pd.Series) -> pd.Series:
    return values.map({1: 1, 2: 1, 3: 2, 4: 3}).astype("Int64")


def summarize_census(records: Iterable[dict], cohort_breaks: Sequence[int] = COHORT_BREAKS) -> pd.DataFrame:
    """Reproduce R scripts 3.1--3.5 at run x ten-year-cohort level."""
    data = pd.DataFrame(records)
    required = {"l_i", "g_i", "year_of_birth_i", "s_i", "s_j", "a_i", "a_j"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"census is missing columns: {sorted(missing)}")
    breaks = list(cohort_breaks)
    labels = [f"({left},{right}]" for left, right in zip(breaks[:-1], breaks[1:])]
    # The source R code retains both spouses and cohorts each record by the
    # focal person's birth year.  A cross-decade couple can therefore enter two
    # different cohort cells; do not deduplicate to one female row.
    data = data.loc[data["l_i"] == "married"].copy()
    data = data.loc[data["year_of_birth_i"].between(breaks[0] + 1, breaks[-1] + 0.9)]
    data["ten_cohort"] = pd.cut(
        np.floor(pd.to_numeric(data["year_of_birth_i"])),
        breaks,
        labels=labels,
        right=True,
    )
    data = data.dropna(subset=["ten_cohort"])
    own_edu = collapse_education(data["s_i"])
    partner_edu = collapse_education(data["s_j"])
    male = data["g_i"].eq("male")
    female = data["g_i"].eq("female")
    data["hyper"] = ((male & own_edu.gt(partner_edu))
                     | (female & own_edu.lt(partner_edu))).astype(int)
    data["homo"] = own_edu.eq(partner_edu).astype(int)
    data["hypo"] = ((male & own_edu.lt(partner_edu))
                    | (female & own_edu.gt(partner_edu))).astype(int)
    own_age = np.floor(pd.to_numeric(data["a_i"]) / 10)
    partner_age = np.floor(pd.to_numeric(data["a_j"]) / 10)
    data["a_diff_sim"] = np.where(male, own_age - partner_age, partner_age - own_age)
    keys = [name for name in (
        "country", "condition_no", "run_no", "ten_cohort", "delta",
        "ws_f", "ws_m", "wy_f", "wy_m", "wa_f", "wa_m",
    ) if name in data.columns]
    output_columns = keys + [
        "count_hyper_sim", "count_homo_sim", "count_hypo_sim",
        "perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim", "a_diff_sim", "n_records",
    ]
    if data.empty:
        return pd.DataFrame(columns=output_columns)
    result = data.groupby(keys, observed=True).agg(
        count_hyper_sim=("hyper", "sum"),
        count_homo_sim=("homo", "sum"),
        count_hypo_sim=("hypo", "sum"),
        a_diff_sim=("a_diff_sim", "mean"),
        n_records=("hyper", "size"),
    ).reset_index()
    total = result[["count_hyper_sim", "count_homo_sim", "count_hypo_sim"]].sum(axis=1)
    result["perc_hyper_sim"] = result["count_hyper_sim"] / total
    result["perc_homo_sim"] = result["count_homo_sim"] / total
    result["perc_hypo_sim"] = result["count_hypo_sim"] / total
    return result[output_columns]


def aggregate_conditions(run_results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim", "a_diff_sim", "n_records"]
    keys = [name for name in (
        "condition_no", "country", "ten_cohort", "delta", "ws_f", "ws_m",
        "wy_f", "wy_m", "wa_f", "wa_m",
    ) if name in run_results.columns]
    return run_results.groupby(keys, observed=True)[metrics].mean().reset_index()


def calibration_loss(simulated: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Exact loss order from original R script ``3_6_calibration_analysis.r``."""
    cohort = simulated.merge(observed, on=["country", "ten_cohort"], validate="many_to_one")
    pairs = {
        "DEV_HYPER": ("perc_hyper_sim", "perc_hyper_obs"),
        "DEV_HOMO": ("perc_homo_sim", "perc_homo_obs"),
        "DEV_HYPO": ("perc_hypo_sim", "perc_hypo_obs"),
        "DEV_AGE": ("a_diff_sim", "a_diff_obs"),
    }
    for output, (sim, obs) in pairs.items():
        cohort[output] = (cohort[sim] - cohort[obs]).abs()
    keys = [name for name in (
        "condition_no", "country", "delta", "ws_f", "ws_m", "wy_f", "wy_m", "wa_f", "wa_m",
    ) if name in cohort.columns]
    condition = cohort.groupby(keys, observed=True)[list(pairs)].sum().reset_index()
    condition["deviation"] = np.prod(condition[list(pairs)], axis=1) ** 0.25
    condition["fit"] = -np.log(condition["deviation"])
    return condition


def response_surface_frame(data: pd.DataFrame, parameters: Sequence[str]) -> pd.DataFrame:
    x = pd.DataFrame(index=data.index)
    for parameter in parameters:
        x[parameter] = pd.to_numeric(data[parameter])
    for parameter in parameters:
        x[f"{parameter}_sq"] = x[parameter] ** 2
    for left, right in combinations(parameters, 2):
        x[f"{left}_x_{right}"] = x[left] * x[right]
    return x


def fit_response_surface(data: pd.DataFrame, parameters: Sequence[str], outcome: str = "fit", weights=None):
    import statsmodels.api as sm
    x = sm.add_constant(response_surface_frame(data, parameters), has_constant="add")
    y = pd.to_numeric(data[outcome])
    return (sm.OLS(y, x, missing="drop").fit() if weights is None
            else sm.WLS(y, x, weights=pd.to_numeric(weights), missing="drop").fit())


NETLOGO_COLUMNS = [
    "agent_i", "tick_of_birth_i", "year_of_birth_i", "g_i", "s_i", "y_i", "a_i", "l_i",
    "agent_j", "tick_of_birth_j", "year_of_birth_j", "g_j", "s_j", "y_j", "a_j", "l_j",
    "country", "year", "wa_m", "wa_f", "wy_m", "wy_f", "ws_m", "ws_f", "delta",
    "run_rec", "condition_no", "run_no",
]


def write_netlogo_compatible_census(records: Iterable[dict], model, path: str | Path) -> Path:
    """Write 39 metadata lines + source header so the original R ``fread`` works."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    c = model.config
    metadata = [
        ("seed_to_use", c.seed_to_use), ("my-seed", model.actual_seed),
        ("burnin_steps", c.burnin_steps), ("start_year", c.start_year), ("end_year", c.end_year),
        ("steps_per_year", c.steps_per_year), ("A_max", c.A_max), ("S_max", c.S_max),
        ("Y_max", c.Y_max), ("A_marr", c.A_marr), ("d_m", c.d_m), ("d_f", c.d_f),
        ("wd_m", c.wd_m), ("wd_f", c.wd_f), ("F", c.F), ("M", c.M),
        ("sigma_m", c.sigma_m), ("sigma_f", c.sigma_f), ("beta_m", c.beta_m),
        ("beta_f", c.beta_f), ("alpha_a_m", c.alpha_a_m), ("alpha_a_f", c.alpha_a_f),
        ("a_0_19_f", c.a_0_19_f), ("a_20_39_f", c.a_20_39_f),
        ("a_40_59_f", c.a_40_59_f), ("a_60_79_f", c.a_60_79_f),
        ("a_0_19_m", c.a_0_19_m), ("a_20_39_m", c.a_20_39_m),
        ("a_40_59_m", c.a_40_59_m), ("a_60_79_m", c.a_60_79_m),
        ("a_begin_s_2", c.a_begin_s_2), ("a_begin_s_3", c.a_begin_s_3),
        ("a_begin_s_4", c.a_begin_s_4), ("a_end_s_1", c.a_end_s_1),
        ("a_end_s_2 ", c.a_end_s_2), ("a_end_s_3", c.a_end_s_3),
        ("a_end_s_4", c.a_end_s_4), ("run_rec", model.run_rec),
    ]
    frame = pd.DataFrame(records, columns=NETLOGO_COLUMNS)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for key, value in metadata:
            separator = " " if key == "a_end_s_2 " else ": "
            handle.write(f"{key}{separator}{value}\n")
        handle.write("=========================================\n")
        handle.write(" ".join(NETLOGO_COLUMNS) + "\n")
        frame.to_csv(handle, sep=" ", index=False, header=False, na_rep="NA", lineterminator="\n")
    return target


def read_netlogo_outputs(directory: str | Path, pattern: str) -> pd.DataFrame:
    files = sorted(Path(directory).glob(pattern))
    if not files:
        raise FileNotFoundError(f"no NetLogo files matched {pattern} in {directory}")
    return pd.concat(
        [pd.read_csv(path, skiprows=39, sep=r"\s+", na_values="NA") for path in files],
        ignore_index=True,
    )
