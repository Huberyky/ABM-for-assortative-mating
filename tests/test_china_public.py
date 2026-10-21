from dataclasses import asdict

import numpy as np
import pandas as pd

from marriage_mesa.china import (
    ablation_design,
    china_base_config,
    fixed_delta_calibration_design,
    recode_cgss_education,
)


def test_cgss_education_mapping_matches_model_output_collapse():
    source = pd.Series([1, 2, 3, 4, 8, 9, 13, 14, 98, np.nan])
    actual = recode_cgss_education(source).tolist()
    expected = [1, 1, 1, 2, 2, 3, 3, np.nan, np.nan, np.nan]
    for left, right in zip(actual, expected):
        assert (pd.isna(left) and pd.isna(right)) or left == right


def test_fixed_delta_design_is_six_factor_source_style_ccd():
    design = fixed_delta_calibration_design()
    assert len(design) == 2**6 + 2 * 6 + 8
    assert design["condition_no"].is_unique
    assert design["delta"].eq(0.9).all()


def test_ablation_rows_change_only_the_declared_mechanism():
    base = china_base_config(ws_f=.4, ws_m=.9, wy_f=1.2, wy_m=1.0, wa_f=10.8, wa_m=5.0)
    design = ablation_design(base).set_index("source")
    scientific = set(asdict(base)) - {"condition_no"}
    expected = {
        "no_s": {"ws_f", "ws_m"},
        "no_delta": {"delta"},
        "no_a": {"wa_f", "wa_m"},
        "no_y": {"wy_f", "wy_m"},
    }
    full = design.loc["full"]
    for source, allowed in expected.items():
        row = design.loc[source]
        changed = {name for name in scientific if row[name] != full[name]}
        assert changed == allowed

