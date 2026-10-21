"""Reusable simulation stage with a machine-readable provenance manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Iterable

import mesa
import numpy as np
import pandas as pd

from .config import ModelConfig
from .experiment import read_design, run_design


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    design_path: str | Path
    input_dir: str | Path
    output_csv: str | Path
    conditions: tuple[int, ...] | None = None
    repetitions: int = 50
    workers: int = 1
    base_seed: int = 20260817
    census_dir: str | Path | None = None
    manifest_path: str | Path | None = None
    source_model_path: str | Path | None = None
    cohort_breaks: tuple[int, ...] | None = None


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_results(results: pd.DataFrame, repetitions: int) -> dict:
    required = {
        "condition_no", "run_no", "ten_cohort", "perc_hyper_sim",
        "perc_homo_sim", "perc_hypo_sim", "a_diff_sim", "seed",
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"workflow output lacks columns: {sorted(missing)}")
    keys = ["condition_no", "run_no", "ten_cohort"]
    duplicates = int(results.duplicated(keys).sum())
    if duplicates:
        raise ValueError(f"workflow output contains {duplicates} duplicate run/cohort rows")
    shares = results[["perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim"]].sum(axis=1)
    if not np.allclose(shares, 1.0, atol=1e-10):
        raise ValueError("couple-type shares do not sum to one")
    run_counts = results.groupby("condition_no", observed=True)["run_no"].nunique()
    incomplete = run_counts.loc[run_counts != repetitions]
    if not incomplete.empty:
        raise ValueError(f"incomplete repetitions by condition: {incomplete.to_dict()}")
    return {
        "rows": int(len(results)),
        "duplicate_run_cohort_rows": duplicates,
        "conditions": int(results["condition_no"].nunique()),
        "runs_per_condition": {str(int(k)): int(v) for k, v in run_counts.items()},
        "share_sum_max_abs_error": float((shares - 1).abs().max()),
    }


def _input_hashes(input_dir: Path, countries: Iterable[str]) -> dict[str, str]:
    names = ["measurement_points.txt"]
    for country in sorted(set(countries)):
        names.extend([
            f"{country}_input_male_s.txt",
            f"{country}_input_female_s.txt",
            f"{country}_input_male_y_s_low.txt",
            f"{country}_input_male_y_s_high.txt",
            f"{country}_input_female_y_s_low.txt",
            f"{country}_input_female_y_s_high.txt",
        ])
    return {name: _sha256(input_dir / name) for name in names}


def run_workflow(spec: WorkflowSpec) -> tuple[pd.DataFrame, Path]:
    """Run the Mesa stage, validate its contract, and freeze provenance.

    The following calibration and effect-assessment stages intentionally stay
    in R; the manifest records their hand-off commands instead of reimplementing
    the paper's estimators in Python.
    """
    if spec.repetitions <= 0 or spec.workers <= 0:
        raise ValueError("repetitions and workers must be positive")
    design_path = Path(spec.design_path).resolve()
    input_dir = Path(spec.input_dir).resolve()
    output_csv = Path(spec.output_csv).resolve()
    design = read_design(design_path)
    if spec.conditions is not None:
        design = design.loc[design["condition_no"].isin(spec.conditions)]
    if design.empty:
        raise ValueError("workflow selected no design conditions")
    configs = [ModelConfig.from_mapping(row) for row in design.to_dict("records")]

    results = run_design(
        design_path=design_path,
        input_dir=input_dir,
        output_csv=output_csv,
        conditions=spec.conditions,
        repetitions=spec.repetitions,
        workers=spec.workers,
        base_seed=spec.base_seed,
        census_dir=spec.census_dir,
        cohort_breaks=spec.cohort_breaks,
    )
    contract = validate_run_results(results, spec.repetitions)
    manifest_path = (Path(spec.manifest_path).resolve() if spec.manifest_path
                     else output_csv.with_suffix(".manifest.json"))
    source = Path(spec.source_model_path).resolve() if spec.source_model_path else None
    manifest = {
        "schema": "grow-vanbavel-mesa-workflow/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "engine": {"python": platform.python_version(), "mesa": mesa.__version__},
        "spec": {key: str(value) if isinstance(value, Path) else value
                 for key, value in asdict(spec).items()},
        "selected_conditions": [int(value) for value in design["condition_no"]],
        "countries": sorted({config.country for config in configs}),
        "provenance": {
            "design_sha256": _sha256(design_path),
            "source_model_sha256": None if source is None else _sha256(source),
            "input_sha256": _input_hashes(input_dir, (config.country for config in configs)),
            "output_sha256": _sha256(output_csv),
        },
        "data_contract": contract,
        "stages": [
            {"name": "mesa_simulation", "status": "complete", "artifact": str(output_csv)},
            {"name": "r_prepare", "status": "manual_next",
             "script": "r_analysis/prepare_mesa_output.R"},
            {"name": "r_calibration", "status": "manual_next",
             "script": "r_analysis/fit_calibration.R"},
            {"name": "netlogo_parity", "status": "manual_next",
             "command": "marriage-mesa parity ..."},
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return results, manifest_path
