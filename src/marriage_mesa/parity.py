from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import read_netlogo_outputs, summarize_census


# Historical engineering diagnostics retained for backward-compatible files.
# These values do not occur in Grow & Van Bavel (2015), the NetLogo model, or
# the original R scripts and must not be interpreted as scientific acceptance
# thresholds.  Formal all-country reporting uses multiplicity-controlled
# inference and reports these columns under an explicit ``legacy`` label.
LEGACY_HEURISTIC_TOLERANCES = {
    "perc_hyper_sim": 0.04,
    "perc_homo_sim": 0.04,
    "perc_hypo_sim": 0.04,
    "a_diff_sim": 0.35,
}
METRIC_TOLERANCES = LEGACY_HEURISTIC_TOLERANCES


def compare_netlogo_mesa(
    netlogo_dir: str | Path,
    pattern: str,
    mesa_results: str | Path,
    output_csv: str | Path,
    output_json: str | Path | None = None,
    standardized_tolerance: float = 0.35,
) -> pd.DataFrame:
    """Compare independent repeated runs; pathwise equal-seed parity is invalid."""
    raw_netlogo = read_netlogo_outputs(netlogo_dir, pattern)
    netlogo = summarize_census(raw_netlogo.to_dict("records"))
    mesa = pd.read_csv(mesa_results)
    # A reusable all-country workflow may pass one combined Mesa result file.
    # The NetLogo glob identifies a single country/condition, so constrain the
    # replica to the same keys before computing any distributional statistic.
    for key in ("country", "condition_no"):
        if key in netlogo and key in mesa:
            source_values = pd.Series(netlogo[key]).dropna().astype(str).unique()
            if len(source_values) == 1:
                mesa = mesa.loc[mesa[key].astype(str).eq(source_values[0])]
    if mesa.empty:
        raise ValueError(f"no Mesa rows match NetLogo pattern {pattern!r}")
    rows = []
    cohorts = sorted(set(netlogo["ten_cohort"].astype(str)) & set(mesa["ten_cohort"].astype(str)))
    for cohort in cohorts:
        for metric, absolute_tolerance in LEGACY_HEURISTIC_TOLERANCES.items():
            source = pd.to_numeric(netlogo.loc[netlogo["ten_cohort"].astype(str).eq(cohort), metric]).dropna()
            replica = pd.to_numeric(mesa.loc[mesa["ten_cohort"].astype(str).eq(cohort), metric]).dropna()
            difference = replica.mean() - source.mean()
            pooled_sd = math_pooled_sd(source, replica)
            standard_difference = difference / pooled_sd if pooled_sd else 0.0
            hedges_g = hedges_standardized_mean_difference(replica, source)
            test = stats.ttest_ind(replica, source, equal_var=False)
            distribution_test = stats.cramervonmises_2samp(replica, source)
            variance_test = stats.levene(replica, source, center="median")
            se = np.sqrt(replica.var(ddof=1) / len(replica) + source.var(ddof=1) / len(source))
            df = welch_df(replica, source)
            critical = stats.t.ppf(0.975, df) if np.isfinite(df) else 1.96
            rows.append({
                "ten_cohort": cohort,
                "metric": metric,
                "netlogo_mean": source.mean(),
                "mesa_mean": replica.mean(),
                "difference": difference,
                "difference_ci95_low": difference - critical * se,
                "difference_ci95_high": difference + critical * se,
                "pooled_standardized_difference": standard_difference,
                "hedges_g": hedges_g,
                "difference_standard_error": se,
                "welch_t_statistic": test.statistic,
                "welch_degrees_freedom": df,
                "netlogo_run_sd": source.std(ddof=1),
                "mesa_run_sd": replica.std(ddof=1),
                "welch_p_value": test.pvalue,
                "cramer_von_mises_statistic": distribution_test.statistic,
                "cramer_von_mises_p_value": distribution_test.pvalue,
                "brown_forsythe_statistic": variance_test.statistic,
                "brown_forsythe_p_value": variance_test.pvalue,
                "legacy_absolute_tolerance": absolute_tolerance,
                "legacy_standardized_tolerance": standardized_tolerance,
                "legacy_pass_absolute": abs(difference) <= absolute_tolerance,
                "legacy_pass_standardized": abs(standard_difference) <= standardized_tolerance,
                "n_netlogo": len(source),
                "n_mesa": len(replica),
            })
    result = pd.DataFrame(rows)
    result["legacy_pass"] = (
        result["legacy_pass_absolute"] & result["legacy_pass_standardized"]
    )
    # Compatibility aliases for callers created before the methodology audit.
    result["absolute_tolerance"] = result["legacy_absolute_tolerance"]
    result["standardized_tolerance"] = result["legacy_standardized_tolerance"]
    result["pass_absolute"] = result["legacy_pass_absolute"]
    result["pass_standardized"] = result["legacy_pass_standardized"]
    result["pass"] = result["legacy_pass"]
    target = Path(output_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(target, index=False)
    if output_json:
        report = {
            "comparison": "independent repeated-run distributional parity",
            "pathwise_seed_identity_expected": False,
            "n_checks": int(len(result)),
            "methodological_status": (
                "legacy heuristic only; not a paper-defined equivalence test"
            ),
            "n_legacy_passed": int(result["legacy_pass"].sum()),
            "all_legacy_passed": bool(result["legacy_pass"].all()),
            "max_absolute_standardized_difference": float(result["pooled_standardized_difference"].abs().max()),
            "legacy_heuristic_thresholds": {
                "standardized_difference": standardized_tolerance,
                "absolute_by_metric": METRIC_TOLERANCES,
            },
        }
        json_target = Path(output_json)
        json_target.parent.mkdir(parents=True, exist_ok=True)
        json_target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def math_pooled_sd(left: pd.Series, right: pd.Series) -> float:
    return float(np.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2))


def hedges_standardized_mean_difference(left: pd.Series, right: pd.Series) -> float:
    """Hedges g for two independent samples (left mean minus right mean)."""
    n_left, n_right = len(left), len(right)
    degrees_freedom = n_left + n_right - 2
    if degrees_freedom <= 0:
        return float("nan")
    pooled_variance = (
        (n_left - 1) * left.var(ddof=1) + (n_right - 1) * right.var(ddof=1)
    ) / degrees_freedom
    if pooled_variance <= 0:
        return 0.0 if left.mean() == right.mean() else float("inf")
    cohen_d = (left.mean() - right.mean()) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * degrees_freedom - 1.0)
    return float(correction * cohen_d)


def welch_df(left: pd.Series, right: pd.Series) -> float:
    a = left.var(ddof=1) / len(left)
    b = right.var(ddof=1) / len(right)
    denominator = (a * a) / (len(left) - 1) + (b * b) / (len(right) - 1)
    return float((a + b) ** 2 / denominator) if denominator else float("inf")
