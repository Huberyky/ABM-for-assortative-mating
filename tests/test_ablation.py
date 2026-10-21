import pandas as pd

from marriage_mesa.ablation import (
    classify_ablation,
    cluster_bootstrap_effects,
    compare_table3_engines,
    compare_with_paper,
    write_ablation_dashboard,
)


def test_ablation_classification_uses_parameters_not_condition_numbers():
    base = {"ws_m": .934, "ws_f": .385, "wy_m": 1.025, "wy_f": 1.201,
            "wa_m": 5.009, "wa_f": 10.833, "delta": .9}
    assert classify_ablation(base) == "full"
    assert classify_ablation(dict(base, ws_m=0, ws_f=0)) == "no_s"
    assert classify_ablation(dict(base, wy_m=0, wy_f=0)) == "no_y"
    assert classify_ablation(dict(base, wa_m=0, wa_f=0)) == "no_a"
    assert classify_ablation(dict(base, delta=.5)) == "no_delta"


def test_country_cluster_ablation_difference_preserves_sign():
    rows = []
    for source, shift in (("full", 0.0), ("no_s", 0.1), ("no_delta", 0.05),
                          ("no_a", -0.02), ("no_y", -0.04)):
        for country in ("BE", "DE"):
            for cohort in ("(1940,1950]", "(1950,1960]"):
                rows.append({
                    "country": country, "ten_cohort": cohort, "source": source,
                    "perc_hyper_sim": .15 + shift,
                    "perc_homo_sim": .70 - shift,
                    "perc_hypo_sim": .15,
                })
    result = cluster_bootstrap_effects(pd.DataFrame(rows), repetitions=100, seed=1)
    no_s_hyper = result.loc[
        result["source"].eq("no_s") & result["couple_type"].eq("hypergamic")
    ].iloc[0]
    assert abs(no_s_hyper["difference_vs_full"] - .1) < 1e-12
    assert no_s_hyper["country_cluster_ci95_low"] > 0


def test_paper_comparison_and_dashboard(tmp_path):
    rows = []
    values = {
        "ESS": (.14, .05), "full": (.13, .05), "no_s": (.18, .05),
        "no_delta": (.16, .05), "no_a": (.12, .04), "no_y": (.09, .04),
    }
    for source, (mean, sd) in values.items():
        rows.extend([
            {"source": source, "couple_type": "hypergamic", "mean": mean, "sd": sd},
            {"source": source, "couple_type": "homogamic", "mean": .71, "sd": .06},
            {"source": source, "couple_type": "hypogamic", "mean": .15, "sd": .06},
        ])
    table = pd.DataFrame(rows)
    comparison = compare_with_paper(table)
    effects = pd.DataFrame([{
        "source": "no_s", "couple_type": "hypergamic",
        "difference_percentage_points": 5.0,
        "country_cluster_ci95_low": .03, "country_cluster_ci95_high": .07,
    }])
    dashboard = write_ablation_dashboard(
        table, effects, comparison, tmp_path / "dashboard.html"
    )
    text = dashboard.read_text(encoding="utf-8")
    assert "原论文式 Table 3" in text
    assert "关闭教育偏好" in text
    engines = compare_table3_engines(table, table.copy())
    assert engines["same_after_two_decimal_rounding"].all()
    assert engines["mean_difference_mesa_minus_netlogo"].abs().max() == 0
