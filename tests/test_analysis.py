from dataclasses import replace

import numpy as np
import pandas as pd

from marriage_mesa.analysis import calibration_loss, summarize_census, write_netlogo_compatible_census
from marriage_mesa.config import ModelConfig
from marriage_mesa.inputs import CohortInputs
from marriage_mesa.model import MarriageModel


def tiny_inputs():
    return CohortInputs.from_arrays(
        years=[2000],
        male_s=np.array([[0.25] * 4]), female_s=np.array([[0.25] * 4]),
        male_y_low=np.array([[0.2] * 5]), male_y_high=np.array([[0.2] * 5]),
        female_y_low=np.array([[0.2] * 5]), female_y_high=np.array([[0.2] * 5]),
        measurement_points=[2000],
    )


def test_netlogo_text_contract_has_header_after_39_metadata_lines(tmp_path):
    config = replace(ModelConfig(), M=5, F=5, burnin_steps=0, start_year=2000,
                     end_year=2000, own_seed=3, seed_to_use="own", d_m=0, d_f=0)
    model = MarriageModel(config=config, inputs=tiny_inputs())
    records = model.run()
    target = write_netlogo_compatible_census(records, model, tmp_path / "run.txt")
    frame = pd.read_csv(target, skiprows=39, sep=r"\s+", na_values="NA")
    assert len(frame) == 10
    assert {"agent_i", "year_of_birth_i", "condition_no", "run_no"}.issubset(frame.columns)


def test_calibration_loss_sums_cohorts_before_geometric_mean():
    simulated = pd.DataFrame({
        "condition_no": [1, 1], "country": ["BE", "BE"],
        "ten_cohort": ["(1940,1950]", "(1950,1960]"],
        "delta": [0.9, 0.9], "ws_f": [1, 1], "ws_m": [1, 1],
        "wy_f": [1, 1], "wy_m": [1, 1], "wa_f": [1, 1], "wa_m": [1, 1],
        "perc_hyper_sim": [0.2, 0.3], "perc_homo_sim": [0.5, 0.4],
        "perc_hypo_sim": [0.3, 0.3], "a_diff_sim": [1.0, 2.0],
    })
    observed = pd.DataFrame({
        "country": ["BE", "BE"], "ten_cohort": ["(1940,1950]", "(1950,1960]"],
        "perc_hyper_obs": [0.1, 0.1], "perc_homo_obs": [0.4, 0.2],
        "perc_hypo_obs": [0.2, 0.1], "a_diff_obs": [0.5, 1.0],
    })
    result = calibration_loss(simulated, observed).iloc[0]
    expected = (0.3 * 0.3 * 0.3 * 1.5) ** 0.25
    assert np.isclose(result["deviation"], expected)


def test_summary_keeps_both_spouses_and_cohorts_each_by_own_birth_year():
    common = {
        "l_i": "married", "country": "BE", "condition_no": 1, "run_no": 1,
        "delta": 0.9, "ws_f": 1, "ws_m": 1, "wy_f": 1, "wy_m": 1,
        "wa_f": 1, "wa_m": 1,
    }
    records = [
        dict(common, g_i="female", year_of_birth_i=1949.5, s_i=3, s_j=4, a_i=500, a_j=530),
        dict(common, g_i="male", year_of_birth_i=1951.5, s_i=4, s_j=3, a_i=530, a_j=500),
    ]
    result = summarize_census(records)
    assert list(result["ten_cohort"].astype(str)) == ["(1940,1950]", "(1950,1960]"]
    assert list(result["perc_hyper_sim"]) == [1.0, 1.0]
    assert list(result["a_diff_sim"]) == [3.0, 3.0]
