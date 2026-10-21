from dataclasses import replace

import numpy as np

from marriage_mesa.config import ModelConfig
from marriage_mesa.inputs import CohortInputs
from marriage_mesa.model import MarriageModel


def simple_inputs():
    return CohortInputs.from_arrays(
        years=[2000],
        male_s=np.array([[0.1, 0.2, 0.5, 0.2]]),
        female_s=np.array([[0.1, 0.2, 0.4, 0.3]]),
        male_y_low=np.array([[0.2] * 5]),
        male_y_high=np.array([[0.2] * 5]),
        female_y_low=np.array([[0.2] * 5]),
        female_y_high=np.array([[0.2] * 5]),
        measurement_points=[2000.1],
    )


def test_timeline_mesa_registration_and_relationship_invariants():
    config = replace(
        ModelConfig(), M=30, F=30, burnin_steps=2, start_year=2000,
        end_year=2000.1, own_seed=123, seed_to_use="own", d_m=0, d_f=0,
    )
    model = MarriageModel(config=config, inputs=simple_inputs())
    assert len(model.agents) == 60
    census = model.run()
    assert model.stopped
    assert model.tick == 3
    assert len(census) == 60
    assert model.invariant_errors() == []
    assert len(model.datacollector.get_model_vars_dataframe()) == 5
    assert len(model.census_datacollector.get_agent_vars_dataframe()) == 60


def test_deterministic_repetition_for_same_mesa_seed():
    config = replace(
        ModelConfig(), M=20, F=20, burnin_steps=2, start_year=2000,
        end_year=2000.1, own_seed=987, seed_to_use="own", d_m=0, d_f=0,
    )
    first = MarriageModel(config=config, inputs=simple_inputs()).run()
    second = MarriageModel(config=config, inputs=simple_inputs()).run()
    assert first == second


def test_cohort_selection_uses_source_year_column():
    inputs = simple_inputs()
    assert inputs.cohort_index(1900) == 0
    assert inputs.cohort_index(2000) == 0
