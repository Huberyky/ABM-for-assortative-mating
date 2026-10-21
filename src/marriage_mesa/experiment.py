from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .analysis import summarize_census, write_netlogo_compatible_census
from .config import ModelConfig
from .inputs import CohortInputs
from .model import MarriageModel


def read_design(path: str | Path) -> pd.DataFrame:
    design = pd.read_csv(path)
    design.columns = [str(column).strip() for column in design.columns]
    if "condition_no" not in design:
        raise ValueError(f"design file lacks condition_no: {path}")
    if design["condition_no"].duplicated().any():
        raise ValueError("condition_no must be unique within a design file")
    return design


def _one_run(
    row: dict,
    input_dir: str,
    run_no: int,
    seed: int,
    census_dir: str | None,
    cohort_breaks: tuple[int, ...] | None,
):
    config = ModelConfig.from_mapping(
        row,
        run_no=run_no,
        own_seed=seed,
        seed_to_use="own",
    )
    inputs = CohortInputs.from_netlogo(input_dir, config.country)
    # The batch runner always needs in-memory censuses for the R bridge.  This
    # infrastructure flag does not alter any agent behavior.
    model = MarriageModel(config=config, inputs=inputs, collect_census=True)
    records = model.run()
    errors = model.invariant_errors()
    if errors:
        raise RuntimeError("; ".join(errors[:10]))
    if census_dir:
        filename = f"{config.country}_c{config.condition_no}_r{run_no}_mesa_{seed}.txt"
        write_netlogo_compatible_census(records, model, Path(census_dir) / filename)
    summary = summarize_census(
        records, cohort_breaks=cohort_breaks or (1940, 1950, 1960, 1970, 1980)
    )
    summary["seed"] = seed
    return summary


def run_design(
    design_path: str | Path,
    input_dir: str | Path,
    output_csv: str | Path,
    conditions: Iterable[int] | None = None,
    repetitions: int = 50,
    workers: int = 1,
    base_seed: int = 20260817,
    census_dir: str | Path | None = None,
    cohort_breaks: Sequence[int] | None = None,
) -> pd.DataFrame:
    design = read_design(design_path)
    if conditions is not None:
        selected = set(conditions)
        design = design.loc[design["condition_no"].isin(selected)]
        absent = selected.difference(set(design["condition_no"]))
        if absent:
            raise ValueError(f"conditions not in design: {sorted(absent)}")
    jobs = []
    for row in design.to_dict("records"):
        condition = int(row["condition_no"])
        for run_no in range(1, repetitions + 1):
            seed = base_seed + condition * 100_000 + run_no
            jobs.append((row, str(input_dir), run_no, seed,
                         None if census_dir is None else str(census_dir),
                         None if cohort_breaks is None else tuple(cohort_breaks)))
    results = []
    if workers == 1:
        results = [_one_run(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_one_run, *job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(target, index=False)
    return final
