"""China-context data adapter and source-faithful experiment designs.

This module changes population inputs and observation targets only.  The Mesa
scientific kernel remains the audited Grow--Van Bavel NetLogo replication.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import ModelConfig
from .inputs import CohortInputs


EDUCATION_MAP = {
    "未上过学": 1,
    "未受教育": 1,
    "小学": 2,
    "中学": 3,
    "高等教育": 4,
}
CHINA_COHORT_BREAKS = (1940, 1950, 1960, 1970, 1980, 1990, 2000)
CALIBRATION_PARAMETERS = ("ws_f", "ws_m", "wy_f", "wy_m", "wa_f", "wa_m")
CALIBRATION_BOUNDS = {
    "ws_f": (0.0, 2.0),
    "ws_m": (0.0, 2.0),
    "wy_f": (0.0, 2.0),
    "wy_m": (0.0, 2.0),
    "wa_f": (0.0, 20.0),
    "wa_m": (0.0, 20.0),
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cohort_start(value: object) -> int:
    years = [int(item) for item in re.findall(r"\d{4}", str(value))]
    if not years:
        raise ValueError(f"cannot parse cohort label: {value!r}")
    return years[0]


def _expand_probabilities(
    table: pd.DataFrame,
    target_years: np.ndarray,
    name: str,
    *,
    allow_boundary_hold: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    years = table.index.to_numpy(dtype=float)
    if not allow_boundary_hold and (
        target_years.min() < years.min() or target_years.max() > years.max()
    ):
        raise ValueError(
            f"{name}: target years extend outside observed "
            f"[{years.min():.0f}, {years.max():.0f}]"
        )
    values = np.column_stack([
        np.interp(target_years, years, table[column].to_numpy(dtype=float))
        for column in table.columns
    ])
    values = np.clip(values, 0.0, None)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError(f"{name}: interpolation produced a zero-total row")
    values /= totals
    status = np.where(
        np.isin(target_years, years),
        "observed",
        np.where(
            (target_years < years.min()) | (target_years > years.max()),
            "extrapolated_boundary_hold",
            "interpolated",
        ),
    )
    audit = pd.DataFrame({
        "input": name,
        "cohort_year": target_years.astype(int),
        "status": status,
        "source_first_year": int(years.min()),
        "source_last_year": int(years.max()),
    })
    return values, audit


def build_china_inputs(
    education_xlsx: str | Path,
    income_xlsx: str | Path,
    *,
    target_years: Iterable[int] = range(1960, 2021, 5),
    measurement_points: Iterable[float] = (2021.0,),
    male_code: int = 1,
    female_code: int = 0,
    allow_boundary_hold: bool = True,
) -> tuple[CohortInputs, pd.DataFrame, pd.DataFrame]:
    """Map the user-prepared workbooks to the NetLogo input contract.

    The earnings files distinguish the source model's education levels 1--3
    (``low``) from level 4 (``high``).  Low-level distributions are pooled with
    cell counts ``n``; an unweighted average would change the empirical input.
    Rows whose education group is missing are audited and excluded because they
    cannot be assigned to either source-model stratum.
    """
    target = np.asarray(list(target_years), dtype=float)
    if target.ndim != 1 or not len(target) or np.any(np.diff(target) <= 0):
        raise ValueError("target_years must be a non-empty increasing sequence")

    education = pd.read_excel(education_xlsx, sheet_name="教育比例校准")
    required_education = {"队列", "教育阶段", "男性比例", "女性比例"}
    missing = required_education.difference(education.columns)
    if missing:
        raise ValueError(f"education workbook lacks columns: {sorted(missing)}")
    education["cohort_year"] = education["队列"].map(_cohort_start)
    education["education"] = education["教育阶段"].map(EDUCATION_MAP)
    if education["education"].isna().any():
        labels = education.loc[education["education"].isna(), "教育阶段"].unique()
        raise ValueError(f"unmapped education labels: {labels.tolist()}")

    def education_table(column: str) -> pd.DataFrame:
        table = education.pivot(
            index="cohort_year", columns="education", values=column
        ).sort_index().reindex(columns=[1, 2, 3, 4])
        if table.isna().any().any():
            raise ValueError(f"education workbook has incomplete cells for {column}")
        if (table < 0).any().any():
            raise ValueError(f"education workbook has negative cells for {column}")
        return table.div(table.sum(axis=1), axis=0)

    male_s, audit_male_s = _expand_probabilities(
        education_table("男性比例"), target, "male_education",
        allow_boundary_hold=allow_boundary_hold,
    )
    female_s, audit_female_s = _expand_probabilities(
        education_table("女性比例"), target, "female_education",
        allow_boundary_hold=allow_boundary_hold,
    )

    income = pd.read_excel(income_xlsx)
    required_income = {"gender", "edu_group", "cohort_5", "y_i", "n", "prob_y"}
    missing = required_income.difference(income.columns)
    if missing:
        raise ValueError(f"income workbook lacks columns: {sorted(missing)}")
    if income.duplicated(["gender", "edu_group", "cohort_5", "y_i"]).any():
        raise ValueError("income workbook contains duplicate probability cells")
    if not income["gender"].dropna().isin([female_code, male_code]).all():
        raise ValueError("income workbook contains an unexpected gender code")
    if not income["y_i"].dropna().isin([1, 2, 3, 4, 5]).all():
        raise ValueError("income workbook contains an unexpected y_i category")
    cell_totals = income.groupby(
        ["gender", "edu_group", "cohort_5"], dropna=False, observed=True
    )["n"].transform("sum")
    implied_probability = np.divide(
        income["n"].to_numpy(dtype=float),
        cell_totals.to_numpy(dtype=float),
        out=np.zeros(len(income), dtype=float),
        where=cell_totals.to_numpy(dtype=float) > 0,
    )
    nonzero_cells = cell_totals.gt(0)
    if not np.allclose(
        implied_probability[nonzero_cells],
        income.loc[nonzero_cells, "prob_y"].to_numpy(dtype=float),
        atol=1e-10,
    ):
        raise ValueError("income prob_y is inconsistent with the supplied cell counts n")
    selected = income.loc[income["edu_group"].isin([1, 2, 3, 4])].copy()

    def income_table(gender_code: int, high: bool) -> pd.DataFrame:
        data = selected.loc[selected["gender"].eq(gender_code)].copy()
        data = data.loc[
            data["edu_group"].eq(4) if high else data["edu_group"].isin([1, 2, 3])
        ]
        table = data.groupby(["cohort_5", "y_i"], observed=True)["n"].sum().unstack(fill_value=0)
        table = table.reindex(columns=[1, 2, 3, 4, 5], fill_value=0).sort_index()
        if table.empty or (table.sum(axis=1) <= 0).any():
            raise ValueError("income workbook contains a zero-count selected stratum")
        return table.div(table.sum(axis=1), axis=0)

    arrays: dict[str, np.ndarray] = {}
    audits = [audit_male_s, audit_female_s]
    for sex, code in (("male", male_code), ("female", female_code)):
        for level, high in (("low", False), ("high", True)):
            name = f"{sex}_income_{level}"
            arrays[name], audit = _expand_probabilities(
                income_table(code, high), target, name,
                allow_boundary_hold=allow_boundary_hold,
            )
            audits.append(audit)

    inputs = CohortInputs.from_arrays(
        years=target,
        male_s=male_s,
        female_s=female_s,
        male_y_low=arrays["male_income_low"],
        male_y_high=arrays["male_income_high"],
        female_y_low=arrays["female_income_low"],
        female_y_high=arrays["female_income_high"],
        measurement_points=measurement_points,
    )
    input_audit = pd.concat(audits, ignore_index=True)
    exclusion_audit = pd.DataFrame([{
        "rule": "exclude_missing_education_group_from_income_conditionals",
        "rows_excluded": int(income["edu_group"].isna().sum()),
        "n_excluded": float(income.loc[income["edu_group"].isna(), "n"].sum()),
        "rows_retained": int(len(selected)),
        "n_retained": float(selected["n"].sum()),
    }])
    return inputs, input_audit, exclusion_audit


def write_netlogo_input_contract(
    inputs: CohortInputs, output_dir: str | Path, *, country: str = "CN"
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    matrices = {
        f"{country}_input_male_s.txt": inputs.male_s,
        f"{country}_input_female_s.txt": inputs.female_s,
        f"{country}_input_male_y_s_low.txt": inputs.male_y_low,
        f"{country}_input_male_y_s_high.txt": inputs.male_y_high,
        f"{country}_input_female_y_s_low.txt": inputs.female_y_low,
        f"{country}_input_female_y_s_high.txt": inputs.female_y_high,
    }
    paths: dict[str, Path] = {}
    for filename, matrix in matrices.items():
        path = target / filename
        lines = ["["] + [
            "[" + " ".join(f"{value:.15g}" for value in row) + "]"
            for row in matrix
        ] + ["]"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths[filename] = path
    measurement = target / "measurement_points.txt"
    measurement.write_text(
        "[" + " ".join(f"{value:.15g}" for value in inputs.measurement_points) + "]\n",
        encoding="utf-8",
    )
    paths[measurement.name] = measurement
    CohortInputs.from_netlogo(target, country)
    return paths


def recode_cgss_education(values: pd.Series) -> pd.Series:
    """Map CGSS schooling to the paper's three post-analysis categories."""
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [numeric.between(1, 3), numeric.between(4, 8), numeric.between(9, 13)],
            [1.0, 2.0, 3.0],
            default=np.nan,
        ),
        index=values.index,
    )


def extract_cgss_targets(
    cgss_dta: str | Path,
    *,
    cohort_breaks: Iterable[int] = CHINA_COHORT_BREAKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct weighted spouse-match and age-gap targets from CGSS 2021."""
    columns = ["A2", "A3_1", "A71a", "A69", "A7a", "A72", "weight"]
    raw = pd.read_stata(cgss_dta, columns=columns, convert_categoricals=False)
    audit_rows = [{"stage": "raw", "rows": int(len(raw))}]
    data = raw.loc[raw["A2"].isin([1, 2]) & raw["A69"].isin([3, 4])].copy()
    audit_rows.append({"stage": "currently_married_with_spouse", "rows": int(len(data))})
    data["self_edu"] = recode_cgss_education(data["A7a"])
    data["partner_edu"] = recode_cgss_education(data["A72"])
    required = ["A3_1", "A71a", "self_edu", "partner_edu", "weight"]
    data = data.dropna(subset=required)
    data = data.loc[
        data["A3_1"].between(1900, 2005)
        & data["A71a"].between(1900, 2005)
        & data["weight"].gt(0)
    ].copy()
    audit_rows.append({"stage": "valid_birth_education_weight", "rows": int(len(data))})

    male_respondent = data["A2"].eq(1)
    data["male_edu"] = np.where(male_respondent, data["self_edu"], data["partner_edu"])
    data["female_edu"] = np.where(male_respondent, data["partner_edu"], data["self_edu"])
    data["male_birth"] = np.where(male_respondent, data["A3_1"], data["A71a"])
    data["female_birth"] = np.where(male_respondent, data["A71a"], data["A3_1"])
    breaks = list(cohort_breaks)
    labels = [f"({left},{right}]" for left, right in zip(breaks[:-1], breaks[1:])]
    data["ten_cohort"] = pd.cut(
        np.floor(pd.to_numeric(data["A3_1"])), breaks, labels=labels, right=True
    )
    data["hyper"] = data["male_edu"].gt(data["female_edu"]).astype(float)
    data["homo"] = data["male_edu"].eq(data["female_edu"]).astype(float)
    data["hypo"] = data["male_edu"].lt(data["female_edu"]).astype(float)
    # At a common survey date, husband's age minus wife's age equals wife's
    # birth year minus husband's birth year.
    data["age_diff"] = data["female_birth"] - data["male_birth"]
    data = data.dropna(subset=["ten_cohort"])
    audit_rows.append({"stage": "within_requested_respondent_cohorts", "rows": int(len(data))})

    rows = []
    for cohort, group in data.groupby("ten_cohort", observed=True):
        weights = group["weight"].to_numpy(dtype=float)
        weight_sum = float(weights.sum())
        weight_sq_sum = float(np.square(weights).sum())
        effective_n = weight_sum**2 / weight_sq_sum
        hyper = float(np.average(group["hyper"], weights=weights))
        homo = float(np.average(group["homo"], weights=weights))
        hypo = float(np.average(group["hypo"], weights=weights))
        age = float(np.average(group["age_diff"], weights=weights))
        # Kish effective-n uncertainty is an explicit approximation because the
        # released variables used here do not encode the complete survey design.
        age_variance = float(
            np.average(np.square(group["age_diff"] - age), weights=weights)
        )
        rows.append({
            "country": "CN",
            "ten_cohort": str(cohort),
            "perc_hyper_obs": hyper,
            "perc_homo_obs": homo,
            "perc_hypo_obs": hypo,
            "a_diff_obs": age,
            "perc_hyper_obs_se_kish": float(np.sqrt(hyper * (1 - hyper) / effective_n)),
            "perc_homo_obs_se_kish": float(np.sqrt(homo * (1 - homo) / effective_n)),
            "perc_hypo_obs_se_kish": float(np.sqrt(hypo * (1 - hypo) / effective_n)),
            "a_diff_obs_se_kish": float(np.sqrt(age_variance / effective_n)),
            "n_unweighted": int(len(group)),
            "weight_sum": weight_sum,
            "kish_effective_n": float(effective_n),
        })
    targets = pd.DataFrame(rows)
    if len(targets) != len(labels):
        raise ValueError(f"CGSS targets cover {len(targets)} of {len(labels)} requested cohorts")
    shares = targets[["perc_hyper_obs", "perc_homo_obs", "perc_hypo_obs"]].sum(axis=1)
    if not np.allclose(shares, 1.0, atol=1e-10):
        raise ValueError("CGSS couple-type shares do not sum to one")
    return targets, pd.DataFrame(audit_rows)


def china_base_config(**overrides: object) -> ModelConfig:
    """Source configuration with China timing and fixed full-model delta."""
    base = ModelConfig(
        country="CN",
        start_year=1921.0,
        end_year=2021.0,
        delta=0.9,
        F=250,
        M=250,
        condition_no=1,
        write_to_file=False,
    )
    return replace(base, **overrides)


def fixed_delta_calibration_design(
    base: ModelConfig | None = None,
    *,
    bounds: Mapping[str, tuple[float, float]] = CALIBRATION_BOUNDS,
    center_points: int = 8,
) -> pd.DataFrame:
    """Six-factor rotatable CCD, matching the paper with delta fixed at 0.9.

    The source design is a seven-factor rotatable central-composite design.
    Fixing delta removes that factor, yielding 2^6 factorial points, 12 axial
    points, and eight center replicates (84 conditions).
    """
    base = china_base_config() if base is None else base
    names = list(CALIBRATION_PARAMETERS)
    if set(bounds) != set(names):
        raise ValueError(f"bounds must contain exactly {names}")
    alpha = (2 ** len(names)) ** 0.25
    factorial_level = 1.0 / alpha
    coded: list[tuple[float, ...]] = list(
        product((-factorial_level, factorial_level), repeat=len(names))
    )
    for index in range(len(names)):
        for sign in (-1.0, 1.0):
            point = [0.0] * len(names)
            point[index] = sign
            coded.append(tuple(point))
    coded.extend([(0.0,) * len(names)] * center_points)

    constants = asdict(base)
    rows = []
    for condition_no, point in enumerate(coded, start=1):
        row = constants.copy()
        row["condition_no"] = condition_no
        row["delta"] = 0.9
        for name, coded_value in zip(names, point):
            lower, upper = bounds[name]
            midpoint = (lower + upper) / 2
            half_range = (upper - lower) / 2
            row[name] = midpoint + coded_value * half_range
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_design(config: ModelConfig) -> pd.DataFrame:
    """Full China model and the paper's four one-component ablations."""
    variants = [
        (1, "full", {}),
        (2, "no_s", {"ws_m": 0.0, "ws_f": 0.0}),
        (3, "no_delta", {"delta": 0.5}),
        (4, "no_a", {"wa_m": 0.0, "wa_f": 0.0}),
        (5, "no_y", {"wy_m": 0.0, "wy_f": 0.0}),
    ]
    rows = []
    for condition_no, source, changes in variants:
        row = asdict(replace(config, condition_no=condition_no, **changes))
        row["source"] = source
        rows.append(row)
    return pd.DataFrame(rows)


def config_from_parameter_file(path: str | Path) -> ModelConfig:
    table = pd.read_csv(path)
    if len(table) != 1:
        raise ValueError("calibrated parameter file must contain exactly one row")
    values = table.iloc[0].to_dict()
    return ModelConfig.from_mapping(asdict(china_base_config()) | values)


def prepare_china_assets(
    *,
    education_xlsx: str | Path,
    income_xlsx: str | Path,
    cgss_dta: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    target = Path(output_dir)
    input_dir = target / "input"
    target.mkdir(parents=True, exist_ok=True)
    inputs, input_audit, exclusion_audit = build_china_inputs(
        education_xlsx, income_xlsx
    )
    targets, target_audit = extract_cgss_targets(cgss_dta)
    paths = write_netlogo_input_contract(inputs, input_dir)
    artifacts = {
        "input_dir": input_dir,
        "input_audit": target / "china_input_interpolation_audit.csv",
        "input_exclusion_audit": target / "china_income_exclusion_audit.csv",
        "targets": target / "cgss2021_targets.csv",
        "target_audit": target / "cgss2021_target_audit.csv",
        "calibration_design": target / "china_calibration_design_delta_0_9.csv",
        "transferred_ablation_design": target / "china_transferred_ablation_design.csv",
        "data_manifest": target / "china_data_manifest.json",
    }
    input_audit.to_csv(artifacts["input_audit"], index=False)
    exclusion_audit.to_csv(artifacts["input_exclusion_audit"], index=False)
    targets.to_csv(artifacts["targets"], index=False)
    target_audit.to_csv(artifacts["target_audit"], index=False)
    fixed_delta_calibration_design().to_csv(artifacts["calibration_design"], index=False)
    ablation_design(china_base_config()).to_csv(
        artifacts["transferred_ablation_design"], index=False
    )
    source_paths = {
        "education_workbook": Path(education_xlsx).resolve(),
        "income_workbook": Path(income_xlsx).resolve(),
        "cgss2021": Path(cgss_dta).resolve(),
    }
    manifest = {
        "schema": "grow-vanbavel-china-data/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
        "model_input_contract": {
            "country": "CN",
            "cohort_years": list(range(1960, 2021, 5)),
            "measurement_points": [2021.0],
            "education_levels": EDUCATION_MAP,
            "income_gender_codes": {"female": 0, "male": 1},
            "income_low_education_groups": [1, 2, 3],
            "income_high_education_group": 4,
            "interpolation": "linear probability interpolation followed by row normalization",
            "extrapolation": "nearest observed boundary held constant and explicitly audited",
            "missing_income_education_group": "excluded because it cannot be assigned to low/high model strata",
        },
        "cgss_target_contract": {
            "marital_status_codes": [3, 4],
            "respondent_sex_codes": {"male": 1, "female": 2},
            "education_recode": {"1-3": 1, "4-8": 2, "9-13": 3},
            "cohort_breaks": list(CHINA_COHORT_BREAKS),
            "weights": "CGSS variable weight",
            "age_difference": "husband age minus wife age = wife birth year minus husband birth year",
        },
        "simulation_scope": {
            "start_year": 1921.0,
            "end_year": 2021.0,
            "delta_full": 0.9,
            "scientific_kernel": "unchanged audited Mesa/NetLogo replication",
        },
    }
    artifacts["data_manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    artifacts.update({f"input:{name}": path for name, path in paths.items()})
    return artifacts
