from dataclasses import replace
import math

import numpy as np

from marriage_mesa.config import ModelConfig
from marriage_mesa.equations import (
    dating_probability,
    death_probability,
    marriage_probability,
    search_probability,
)
from marriage_mesa.inputs import CohortInputs
from marriage_mesa.model import DATING, MARRIED, MarriageModel


def source_inputs(measurement_points=(1999,)):
    return CohortInputs.from_arrays(
        years=[1921],
        male_s=np.array([[0.25] * 4]),
        female_s=np.array([[0.25] * 4]),
        male_y_low=np.array([[0.2] * 5]),
        male_y_high=np.array([[0.2] * 5]),
        female_y_low=np.array([[0.2] * 5]),
        female_y_high=np.array([[0.2] * 5]),
        measurement_points=measurement_points,
    )


def source_config(**overrides):
    base = replace(
        ModelConfig(),
        M=2,
        F=2,
        burnin_steps=0,
        start_year=2000,
        end_year=2000.1,
        seed_to_use="own",
        own_seed=17,
        d_m=0,
        d_f=0,
    )
    return replace(base, **overrides)


def test_male_age_utility_uses_absolute_age_240_and_female_uses_relative_age():
    config = source_config(
        ws_m=0,
        wy_m=0,
        wa_m=1,
        ws_f=0,
        wy_f=0,
        wa_f=1,
        alpha_a_m=240,
        alpha_a_f=25,
    )
    model = MarriageModel(config=config, inputs=source_inputs())
    male = next(agent for agent in model.agents if agent.gender == "male")
    female = next(agent for agent in model.agents if agent.gender == "female")

    male.age, female.age = 600, 240
    assert math.isclose(model._mate_value(male, female), 1.0)

    female.age, male.age = 300, 325
    assert math.isclose(model._mate_value(female, male), 1.0)


def test_source_probability_equations():
    assert math.isclose(search_probability(duration=10, commitment=0.015), math.exp(-0.15))
    expected_date = (1 - math.exp(-240 * 0.003 * 0.75)) * math.exp(-10 * 0.015)
    assert math.isclose(
        dating_probability(age=240, age_pressure=0.003, value=0.75,
                           duration=10, commitment=0.015),
        expected_date,
    )
    expected_marriage = ((1 - math.exp(-240 * 0.003 * 0.75))
                         * (1 - math.exp(-10 * 0.015)))
    assert math.isclose(
        marriage_probability(age=240, age_pressure=0.003, value=0.75,
                             duration=10, commitment=0.015),
        expected_marriage,
    )
    assert math.isclose(
        death_probability(age=400, max_age=800, shape=6,
                          maximum_probability=0.1, steps_per_year=10),
        (0.5**6) * 0.01,
    )


def test_partner_switch_preserves_source_proposal_state_and_resets_abandoned_partners():
    model = MarriageModel(config=source_config(), inputs=source_inputs())
    males = [agent for agent in model.agents if agent.gender == "male"]
    females = [agent for agent in model.agents if agent.gender == "female"]
    male_a, male_b = males
    female_a, female_b = females

    for left, right in ((male_a, female_a), (male_b, female_b)):
        left.relation = right.relation = MARRIED
        left.partner, right.partner = right.source_id, left.source_id
        left.marriage_proposal, right.marriage_proposal = right.source_id, left.source_id

    stale_male = male_a.marriage_proposal
    stale_female = female_b.marriage_proposal
    model._form_dating(male_a, female_b)

    assert male_a.relation == female_b.relation == DATING
    assert male_a.marriage_proposal == stale_male
    assert female_b.marriage_proposal == stale_female
    assert female_a.marriage_proposal is None
    assert male_b.marriage_proposal is None
    assert model.invariant_errors() == []


def test_step_uses_exact_netlogo_phase_order():
    model = MarriageModel(config=source_config(), inputs=source_inputs())
    calls = []
    names = [
        "do_make_new_agents",
        "do_update_schooling_status",
        "do_update_p_seek_i",
        "do_meeting_and_dating",
        "do_marriage_proposals",
        "do_wed_agents",
        "do_kill_agents",
        "do_update_age",
        "do_update_relation_times",
        "do_update_year",
    ]
    for name in names:
        setattr(model, name, lambda name=name: calls.append(name))
    model.step()
    assert calls == names


def test_agentset_activation_has_netlogo_snapshot_semantics():
    model = MarriageModel(config=source_config(), inputs=source_inputs())
    initial_count = len(model.agents)
    seen = []

    def action(agent):
        seen.append(agent.source_id)
        if len(seen) == 1:
            model._new_agent("male", 0, model.year)

    model.activation.ask(model.agents, action)
    assert len(seen) == initial_count
    assert len(model.agents) == initial_count + 1
