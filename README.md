# ABM for assortative mating 婚姻匹配模拟

This repository contains a Python Mesa replication of the agent-based marriage-market model in:

> Grow, A., & Van Bavel, J. (2015). Assortative Mating and the Reversal of Gender Inequality in Education in Europe: An Agent-Based Model. *PLOS ONE, 10*(6), e0127806. https://doi.org/10.1371/journal.pone.0127806

本仓库只发布复现模型的程序代码，不包含原始调查数据、论文作者的数据文件、个人整理数据、模拟输出或本地研究文档。

## What is reproduced

The Mesa engine preserves the original NetLogo model's:

- agent attributes and education states;
- education, earnings-prospect, and age mate-value equations;
- relationship search, bilateral dating, partner switching, proposal, marriage, mortality, and replacement rules;
- within-tick phase order and randomized `ask` semantics;
- experimental-design parameters and source-compatible census output;
- R aggregation, response-surface calibration, validation, and ablation workflow.

Mesa 3 `AgentSet.shuffle_do` is used through a snapshot activation adapter. This keeps each NetLogo-style phase independently randomized without changing the scientific order of events.

## Repository layout

```text
src/marriage_mesa/
  model.py          # agent state and source-faithful process
  equations.py      # pure numerical equations
  activation.py     # NetLogo ask -> Mesa AgentSet adapter
  inputs.py         # cohort input contract
  experiment.py     # repeated experimental runs
  analysis.py       # census and cohort summaries
  workflow.py       # validation, hashes, provenance manifests
  parity.py         # NetLogo-Mesa distributional comparison
  ablation.py       # mechanism-removal experiments
  china.py          # optional China input/calibration utilities
  visualization.py # optional SolaraViz dashboard
r_analysis/         # R continuation of calibration/effect assessment
tests/              # data-free unit and workflow tests
```

## Installation

Python 3.10 or later is required.

```bash
python -m venv .venv
python -m pip install -e ".[test]"
pytest
```

For the optional Mesa/Solara visualization:

```bash
python -m pip install -e ".[visualization]"
```

## Input contract

Data are intentionally not distributed. To run a country, supply a directory containing:

```text
<CC>_input_male_s.txt
<CC>_input_female_s.txt
<CC>_input_male_y_s_low.txt
<CC>_input_male_y_s_high.txt
<CC>_input_female_y_s_low.txt
<CC>_input_female_y_s_high.txt
measurement_points.txt
```

Each education matrix contains a cohort year plus four probabilities. Each earnings-prospect matrix contains a cohort year plus five probabilities. Probability rows must be non-negative and sum to one. Experimental-design CSV files must include `condition_no`, `country`, and the model parameters used by `ModelConfig`.

## Batch simulation

After installation:

```bash
marriage-mesa run-workflow \
  --design path/to/design_data_validation.csv \
  --input-dir path/to/input \
  --output outputs/validation.csv \
  --repetitions 50 \
  --workers 4 \
  --manifest outputs/validation.manifest.json
```

Use `marriage-mesa --help` for parity comparison, ablation, visualization, and China-specific preparation commands.

## R continuation

`r_analysis/prepare_mesa_output.R` parameterizes the repeated logic of the original country-specific preparation scripts. `r_analysis/fit_calibration.R` retains the original full second-order response surface, OLS, residual variance model, and inverse-variance WLS estimator. R requires the `rsm` package for calibration.

## Reproducibility scope

Python/Mesa and NetLogo use different random-number and agent-set implementations. Equal integer seeds therefore do not imply pathwise-identical trajectories. The relevant replication target is equation/process fidelity plus distributional agreement across independent repeated runs.

The code preserves one easily mistranslated source rule: men evaluate how close a woman's absolute age is to 24 years, while women evaluate how close a man's age is to their own age plus 2.5 years.

## Data and licensing note

No empirical data or original model files are included here. Obtain any original supporting material under its own terms and cite Grow and Van Bavel (2015). No separate license is granted for this repository unless a license file is added by the owner.

