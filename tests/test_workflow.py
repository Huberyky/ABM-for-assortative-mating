import pandas as pd
import pytest

from marriage_mesa.workflow import validate_run_results


def valid_results():
    return pd.DataFrame({
        "condition_no": [1, 1],
        "run_no": [1, 2],
        "ten_cohort": ["(1940,1950]", "(1940,1950]"],
        "perc_hyper_sim": [0.2, 0.3],
        "perc_homo_sim": [0.5, 0.4],
        "perc_hypo_sim": [0.3, 0.3],
        "a_diff_sim": [2.0, 2.1],
        "seed": [11, 12],
    })


def test_workflow_contract_accepts_complete_unique_results():
    contract = validate_run_results(valid_results(), repetitions=2)
    assert contract["rows"] == 2
    assert contract["runs_per_condition"] == {"1": 2}


def test_workflow_contract_rejects_invalid_share_sum():
    results = valid_results()
    results.loc[0, "perc_hypo_sim"] = 0.2
    with pytest.raises(ValueError, match="sum to one"):
        validate_run_results(results, repetitions=2)
