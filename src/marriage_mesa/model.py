from __future__ import annotations

import math
import secrets
from typing import Any

import mesa
from mesa.datacollection import DataCollector

from .activation import NetLogoActivation
from .config import ModelConfig
from .equations import (
    dating_probability,
    death_probability,
    marriage_probability,
    mate_value,
    search_probability,
)
from .inputs import CohortInputs


SINGLE = "single"
DATING = "dating_relation"
MARRIED = "married"


class MarriageAgent(mesa.Agent):
    """One NetLogo turtle, registered automatically with its Mesa model."""

    def __init__(
        self,
        model: "MarriageModel",
        source_id: int,
        gender: str,
        age: int | None,
        year_of_birth: float,
    ) -> None:
        super().__init__(model)
        self.source_id = source_id
        self.gender = gender
        self.age = age
        self.education: int | None = None
        self.income: int | None = None
        self.relation = SINGLE
        self.partner: int | None = None
        self.duration = 0
        self.schooling = 1
        self.p_seek = 0.0
        self.tick_of_birth = model.tick
        self.year_of_birth = year_of_birth
        self.marriage_proposal: int | None = None


class MarriageModel(mesa.Model):
    """Process-faithful Mesa translation of ``simulation_model/model.nlogo``.

    Every source procedure is represented by one ``do_*`` method and ``step``
    invokes them in the exact order of NetLogo's ``go`` procedure.  Equal seeds
    do not imply pathwise identity because Mesa/Python and NetLogo use different
    random engines and agent-set implementations; validation is distributional.
    """

    def __init__(
        self,
        *,
        config: ModelConfig,
        inputs: CohortInputs,
        collect_census: bool | None = None,
    ) -> None:
        config.validate()
        inputs.validate()
        actual_seed = config.own_seed if config.seed_to_use == "own" else secrets.randbelow(2**31 - 1)
        super().__init__(rng=actual_seed)
        self.config = config
        self.inputs = inputs
        self.actual_seed = actual_seed
        self.tick = 0
        self.year = 0.0
        self.next_source_id = 0
        self.people: dict[int, MarriageAgent] = {}
        self.male_replacements = 0
        self.female_replacements = 0
        self.censuses: list[dict[str, Any]] = []
        self.collect_census = config.write_to_file if collect_census is None else collect_census
        self.stopped = False
        self.run_rec = f"{config.country}_c{config.condition_no}_r{config.run_no}_seed{actual_seed}"
        self.activation = NetLogoActivation(self)
        self.datacollector = DataCollector(
            model_reporters={
                "year": lambda m: m.year,
                "population": lambda m: len(m.people),
                "single": lambda m: sum(a.relation == SINGLE for a in m.people.values()),
                "dating": lambda m: sum(a.relation == DATING for a in m.people.values()),
                "married": lambda m: sum(a.relation == MARRIED for a in m.people.values()),
            }
        )
        self.census_datacollector = DataCollector(
            agent_reporters={
                "source_id": "source_id",
                "gender": "gender",
                "age": "age",
                "education": "education",
                "income": "income",
                "relation": "relation",
                "partner": "partner",
                "duration": "duration",
                "schooling": "schooling",
                "p_seek": "p_seek",
            },
        )
        self.do_update_year()
        self.do_initialize_agents()
        self.do_update_schooling_status()
        self.datacollector.collect(self)

    def _new_agent(self, gender: str, age: int | None, year_of_birth: float) -> MarriageAgent:
        source_id = self.next_source_id
        self.next_source_id += 1
        # Source setxy consumes two random draws, even though position is unused.
        self.random.random()
        self.random.random()
        agent = MarriageAgent(self, source_id, gender, age, year_of_birth)
        self.people[source_id] = agent
        return agent

    def _cohort_rows(self) -> tuple[Any, ...]:
        index = self.inputs.cohort_index(self.year)
        return (
            self.inputs.male_s[index],
            self.inputs.female_s[index],
            self.inputs.male_y_low[index],
            self.inputs.male_y_high[index],
            self.inputs.female_y_low[index],
            self.inputs.female_y_high[index],
        )

    def _categorical(self, row: Any) -> int:
        weights = row[1:]
        pick = self.random.random() * float(sum(weights))
        upper = 0.0
        for category, weight in enumerate(weights, start=1):
            upper += float(weight)
            if pick < upper:
                return category
        return len(weights)

    def do_initialize_agents(self) -> None:
        for _ in range(self.config.M):
            self._new_agent("male", None, self.year)
        for _ in range(self.config.F):
            self._new_agent("female", None, self.year)
        self._assign_new_attributes(initial=True)

    def _assign_education(self, agent: MarriageAgent, row: Any) -> None:
        agent.education = self._categorical(row)

    def _assign_income(self, agent: MarriageAgent, low: Any, high: Any) -> None:
        row = high if agent.education == self.config.S_max else low
        agent.income = self._categorical(row)

    @staticmethod
    def _set_newborn_age(agent: MarriageAgent) -> None:
        agent.age = 0

    def _assign_initial_age(self, agent: MarriageAgent, shares: list[float]) -> None:
        pick = self.random.random() * sum(shares)
        cumulative = 0.0
        band = 3
        for index, share in enumerate(shares):
            cumulative += share
            if pick < cumulative:
                band = index
                break
        agent.age = self.random.randrange(200) + 200 * band

    def _assign_new_attributes(self, initial: bool) -> None:
        male_s, female_s, male_y_low, male_y_high, female_y_low, female_y_high = self._cohort_rows()
        new_agents = self.activation.select(lambda agent: agent.age is None)
        for gender, row in (("male", male_s), ("female", female_s)):
            agents = (agent for agent in new_agents if agent.gender == gender)
            self.activation.ask(agents, self._assign_education, row)
        for gender, low, high in (
            ("male", male_y_low, male_y_high),
            ("female", female_y_low, female_y_high),
        ):
            agents = (agent for agent in new_agents if agent.gender == gender)
            self.activation.ask(agents, self._assign_income, low, high)
        if not initial:
            self.activation.ask(new_agents, self._set_newborn_age)
            return
        for gender in ("male", "female"):
            shares = (
                [self.config.a_0_19_m, self.config.a_20_39_m, self.config.a_40_59_m, self.config.a_60_79_m]
                if gender == "male"
                else [self.config.a_0_19_f, self.config.a_20_39_f, self.config.a_40_59_f, self.config.a_60_79_f]
            )
            agents = (agent for agent in new_agents if agent.gender == gender)
            self.activation.ask(agents, self._assign_initial_age, shares)

    def do_make_new_agents(self) -> None:
        for _ in range(self.male_replacements):
            self._new_agent("male", None, self.year)
        for _ in range(self.female_replacements):
            self._new_agent("female", None, self.year)
        if self.male_replacements or self.female_replacements:
            self._assign_new_attributes(initial=False)
        self.male_replacements = self.female_replacements = 0

    def _update_schooling_status(self, agent: MarriageAgent) -> None:
        assert agent.age is not None and agent.education is not None
        if agent.age < self.config.a_begin_s_2:
            agent.schooling = 1
        elif agent.age < self.config.a_begin_s_3:
            agent.schooling = 2
        elif agent.age < self.config.a_begin_s_4:
            agent.schooling = 3
        else:
            agent.schooling = 4
        end_age = {
            1: self.config.a_end_s_1,
            2: self.config.a_end_s_2,
            3: self.config.a_end_s_3,
            4: self.config.a_end_s_4,
        }[agent.education]
        if agent.age >= end_age:
            agent.schooling = 5

    def do_update_schooling_status(self) -> None:
        self.activation.ask(self.agents, self._update_schooling_status)

    @staticmethod
    def _by_gender(agent: MarriageAgent, female: float, male: float) -> float:
        return female if agent.gender == "female" else male

    def _mate_value(self, ego: MarriageAgent, alter: MarriageAgent) -> float:
        assert ego.education and alter.education and alter.income
        assert ego.age is not None and alter.age is not None
        ws = self._by_gender(ego, self.config.ws_f, self.config.ws_m)
        wy = self._by_gender(ego, self.config.wy_f, self.config.wy_m)
        wa = self._by_gender(ego, self.config.wa_f, self.config.wa_m)
        # Source asymmetry is intentional: men prefer an absolute partner age
        # of 240 steps, while women prefer own age + 25 steps.
        target_age = ego.age + self.config.alpha_a_f if ego.gender == "female" else self.config.alpha_a_m
        return mate_value(
            ego_education=ego.education,
            partner_education=alter.education,
            partner_income=alter.income,
            partner_age=alter.age,
            ideal_partner_age=target_age,
            education_weight=ws,
            income_weight=wy,
            age_weight=wa,
            max_education=self.config.S_max,
            max_income=self.config.Y_max,
            max_age=self.config.A_max,
        )

    def _accept_date(self, ego: MarriageAgent, alter: MarriageAgent) -> bool:
        assert ego.age is not None
        sigma = self._by_gender(ego, self.config.sigma_f, self.config.sigma_m)
        beta = self._by_gender(ego, self.config.beta_f, self.config.beta_m)
        new_value = self._mate_value(ego, alter)
        if ego.relation == SINGLE:
            probability = dating_probability(
                age=ego.age,
                age_pressure=sigma,
                value=new_value,
                duration=ego.duration,
                commitment=beta,
            )
        else:
            current = self.people.get(ego.partner) if ego.partner is not None else None
            if current is None or new_value <= self._mate_value(ego, current):
                return False
            probability = dating_probability(
                age=ego.age,
                age_pressure=sigma,
                value=new_value,
                duration=ego.duration,
                commitment=beta,
            )
        return self.random.random() < probability

    def _break_relation(self, uid: int, *, reset_focal_proposal: bool = True) -> None:
        agent = self.people.get(uid)
        if agent is None:
            return
        partner = self.people.get(agent.partner) if agent.partner is not None else None
        agent.relation, agent.partner = SINGLE, None
        agent.duration = 0
        if reset_focal_proposal:
            agent.marriage_proposal = None
        if partner is not None and partner.partner == uid:
            partner.relation, partner.partner = SINGLE, None
            partner.duration, partner.marriage_proposal = 0, None

    def _form_dating(self, ego: MarriageAgent, alter: MarriageAgent) -> None:
        if ego.partner is not None:
            self._break_relation(ego.source_id, reset_focal_proposal=False)
        if alter.partner is not None:
            self._break_relation(alter.source_id, reset_focal_proposal=False)
        ego.relation = alter.relation = DATING
        ego.partner, alter.partner = alter.source_id, ego.source_id
        ego.duration = alter.duration = 0

    def _update_p_seek(self, agent: MarriageAgent) -> None:
        beta = self._by_gender(agent, self.config.beta_f, self.config.beta_m)
        agent.p_seek = search_probability(duration=agent.duration, commitment=beta)

    def do_update_p_seek_i(self) -> None:
        eligible = self.activation.select(
            lambda agent: agent.age is not None and agent.age >= self.config.A_marr
        )
        self.activation.ask(eligible, self._update_p_seek)

    def _one_without_partner(
        self,
        pool: list[int],
        excluded: int | None,
        excluded_is_member: bool,
    ) -> MarriageAgent | None:
        if not pool:
            return None
        if excluded is None or not excluded_is_member:
            return self.people[pool[self.random.randrange(len(pool))]]
        if len(pool) == 1:
            return None
        index = self.random.randrange(len(pool) - 1)
        uid = pool[index] if pool[index] != excluded else pool[-1]
        return self.people[uid]

    def _meeting_activation(
        self,
        ego: MarriageAgent,
        by_school: dict[tuple[str, int], list[int]],
        different_school: dict[tuple[str, int], list[int]],
    ) -> None:
        if self.random.random() >= ego.p_seek:
            return
        same_school = self.random.random() < self.config.delta
        opposite = "female" if ego.gender == "male" else "male"
        pool = (by_school.get((opposite, ego.schooling), [])
                if same_school else different_school[(opposite, ego.schooling)])
        current = self.people.get(ego.partner) if ego.partner is not None else None
        excluded_is_member = current is not None and ((current.schooling == ego.schooling) == same_school)
        alter = self._one_without_partner(pool, ego.partner, excluded_is_member)
        if alter is not None and self._accept_date(ego, alter) and self._accept_date(alter, ego):
            self._form_dating(ego, alter)

    def do_meeting_and_dating(self) -> None:
        eligible = self.activation.select(
            lambda agent: (agent.age is not None
                           and agent.age >= self.config.A_marr
                           and agent.p_seek > 0)
        )
        adults = {"male": [], "female": []}
        by_school: dict[tuple[str, int], list[int]] = {}
        for agent in self.agents:
            if agent.age is not None and agent.age >= self.config.A_marr:
                adults[agent.gender].append(agent.source_id)
                by_school.setdefault((agent.gender, agent.schooling), []).append(agent.source_id)
        different_school = {
            (gender, school): [uid for uid in adults[gender] if self.people[uid].schooling != school]
            for gender in ("male", "female") for school in range(1, 6)
        }
        self.activation.ask(eligible, self._meeting_activation, by_school, different_school)

    def _marriage_proposal_activation(self, ego: MarriageAgent) -> None:
        if ego.relation != DATING or ego.partner is None:
            return
        partner = self.people.get(ego.partner)
        if partner is None:
            return
        assert ego.age is not None
        sigma = self._by_gender(ego, self.config.sigma_f, self.config.sigma_m)
        beta = self._by_gender(ego, self.config.beta_f, self.config.beta_m)
        value = self._mate_value(ego, partner)
        probability = marriage_probability(
            age=ego.age,
            age_pressure=sigma,
            value=value,
            duration=ego.duration,
            commitment=beta,
        )
        if self.random.random() < probability:
            ego.marriage_proposal = partner.source_id

    def do_marriage_proposals(self) -> None:
        eligible = self.activation.select(
            lambda agent: (agent.age is not None
                           and agent.age >= self.config.A_marr
                           and agent.relation == DATING)
        )
        self.activation.ask(eligible, self._marriage_proposal_activation)

    def do_wed_agents(self) -> None:
        # Each undirected dating link is considered once, in random link order.
        links = [(a.source_id, a.partner) for a in self.agents
                 if a.relation == DATING and a.partner is not None and a.source_id < a.partner]
        self.random.shuffle(links)
        for uid, partner_id in links:
            agent, partner = self.people.get(uid), self.people.get(partner_id)
            if (agent is not None and partner is not None
                    and agent.relation == partner.relation == DATING
                    and agent.marriage_proposal == partner.source_id
                    and partner.marriage_proposal == agent.source_id):
                agent.relation = partner.relation = MARRIED

    def _death_activation(self, agent: MarriageAgent, gender: str, dead: dict[str, int]) -> None:
        if agent.age is None:
            return
        d = self.config.d_f if gender == "female" else self.config.d_m
        wd = self.config.wd_f if gender == "female" else self.config.wd_m
        probability = death_probability(
            age=agent.age,
            max_age=self.config.A_max,
            shape=wd,
            maximum_probability=d,
            steps_per_year=self.config.steps_per_year,
        )
        draw = self.random.random()
        if draw < probability or agent.age >= self.config.A_max:
            uid = agent.source_id
            self._break_relation(uid)
            del self.people[uid]
            agent.remove()
            dead[gender] += 1

    def do_kill_agents(self) -> None:
        dead = {"male": 0, "female": 0}
        for gender in ("male", "female"):
            agents = self.activation.select(lambda agent: agent.gender == gender)
            self.activation.ask(agents, self._death_activation, gender, dead)
        self.male_replacements = dead["male"]
        self.female_replacements = dead["female"]

    def _census_row(self, agent: MarriageAgent) -> dict[str, Any]:
        partner = self.people.get(agent.partner) if agent.partner is not None else None
        return {
            "agent_i": agent.source_id,
            "tick_of_birth_i": agent.tick_of_birth,
            "year_of_birth_i": agent.year_of_birth,
            "g_i": agent.gender,
            "s_i": agent.education,
            "y_i": agent.income,
            "a_i": agent.age,
            "l_i": agent.relation,
            "agent_j": None if partner is None else partner.source_id,
            "tick_of_birth_j": None if partner is None else partner.tick_of_birth,
            "year_of_birth_j": None if partner is None else partner.year_of_birth,
            "g_j": None if partner is None else partner.gender,
            "s_j": None if partner is None else partner.education,
            "y_j": None if partner is None else partner.income,
            "a_j": None if partner is None else partner.age,
            "l_j": None if partner is None else partner.relation,
            "country": self.config.country,
            "year": self.year,
            "wa_m": self.config.wa_m,
            "wa_f": self.config.wa_f,
            "wy_m": self.config.wy_m,
            "wy_f": self.config.wy_f,
            "ws_m": self.config.ws_m,
            "ws_f": self.config.ws_f,
            "delta": self.config.delta,
            "run_rec": self.run_rec,
            "condition_no": self.config.condition_no,
            "run_no": self.config.run_no,
        }

    def _record_census_agent(self, agent: MarriageAgent) -> None:
        self.censuses.append(self._census_row(agent))

    def do_write_census(self) -> None:
        related = self.activation.select(lambda agent: agent.relation != SINGLE)
        single = self.activation.select(lambda agent: agent.relation == SINGLE)
        self.activation.ask(related, self._record_census_agent)
        self.activation.ask(single, self._record_census_agent)
        self.census_datacollector.collect(self)

    @staticmethod
    def _increment_age(agent: MarriageAgent) -> None:
        assert agent.age is not None
        agent.age += 1

    def do_update_age(self) -> None:
        self.activation.ask(self.agents, self._increment_age)

    @staticmethod
    def _increment_relation_time(agent: MarriageAgent) -> None:
        agent.duration += 1

    def do_update_relation_times(self) -> None:
        related = self.activation.select(lambda agent: agent.relation != SINGLE)
        self.activation.ask(related, self._increment_relation_time)

    def do_update_year(self) -> None:
        self.year = (0.0 if self.tick < self.config.burnin_steps else
                     self.config.start_year
                     + (self.tick - self.config.burnin_steps) / self.config.steps_per_year)

    def step(self) -> None:
        if self.stopped:
            return
        self.do_make_new_agents()
        self.do_update_schooling_status()
        self.do_update_p_seek_i()
        self.do_meeting_and_dating()
        self.do_marriage_proposals()
        self.do_wed_agents()
        self.do_kill_agents()
        if (self.collect_census
                and any(math.isclose(self.year, point, abs_tol=1e-9)
                        for point in self.inputs.measurement_points)):
            self.do_write_census()
        self.datacollector.collect(self)
        if math.isclose(self.year, self.config.end_year, abs_tol=1e-9):
            self.stopped = True
            self.running = False
            return
        self.tick += 1
        self.do_update_age()
        self.do_update_relation_times()
        self.do_update_year()

    def run(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        expected = (self.config.burnin_steps
                    + round((self.config.end_year - self.config.start_year) * self.config.steps_per_year)
                    + 1)
        limit = expected if max_steps is None else max_steps
        for _ in range(limit):
            self.step()
            if self.stopped:
                break
        if max_steps is None and not self.stopped:
            raise RuntimeError(f"model did not reach end_year in {limit} steps; year={self.year}")
        return self.censuses

    def invariant_errors(self) -> list[str]:
        errors: list[str] = []
        for agent in self.agents:
            if agent.partner is None and agent.relation != SINGLE:
                errors.append(f"agent {agent.source_id}: non-single without partner")
            if agent.partner is not None:
                partner = self.people.get(agent.partner)
                if partner is None or partner.partner != agent.source_id:
                    errors.append(f"agent {agent.source_id}: asymmetric partner link")
                elif partner.relation != agent.relation:
                    errors.append(f"agent {agent.source_id}: relation-status mismatch")
            if agent.education not in {1, 2, 3, 4} or agent.income not in {1, 2, 3, 4, 5}:
                errors.append(f"agent {agent.source_id}: invalid education/income category")
        genders = {gender: sum(a.gender == gender for a in self.agents) for gender in ("male", "female")}
        accounted = {
            "male": genders["male"] + self.male_replacements,
            "female": genders["female"] + self.female_replacements,
        }
        if accounted != {"male": self.config.M, "female": self.config.F}:
            errors.append(f"population accounting failed: live={genders}, accounted={accounted}")
        return errors
