import pandas as pd

from marriage_mesa.reporting import write_markdown_report, write_parity_dashboard


def test_all_country_report_writers_include_run_counts_and_full_detail(tmp_path):
    summary = pd.DataFrame([{
        "country": "BE", "country_name": "比利时", "condition_no": 1,
        "checks": 2, "passed": 1, "passed_absolute": 2,
        "passed_standardized": 1, "max_abs_standardized_difference": 0.4,
        "max_absolute_tolerance_ratio": 0.5, "all_passed": False,
    }])
    common = {
        "country": "BE", "country_name": "比利时", "condition_no": 1,
        "ten_cohort": "(1940,1950]", "metric": "perc_hyper_sim",
        "metric_name": "教育上婚比例", "netlogo_mean": 0.2, "mesa_mean": 0.19,
        "difference": -0.01, "difference_ci95_low": -0.02,
        "difference_ci95_high": 0.0, "pooled_standardized_difference": -0.1,
        "welch_p_value": 0.5, "absolute_tolerance": 0.04,
        "standardized_tolerance": 0.35, "pass_absolute": True,
        "n_netlogo": 50, "n_mesa": 100, "absolute_difference": 0.01,
        "absolute_standardized_difference": 0.1, "absolute_tolerance_ratio": 0.25,
    }
    detail = pd.DataFrame([
        dict(common, pass_standardized=True, **{"pass": True}),
        dict(common, metric="perc_homo_sim", metric_name="教育同质婚比例",
             pooled_standardized_difference=0.4,
             absolute_standardized_difference=0.4,
             pass_standardized=False, **{"pass": False}),
    ])
    html = write_parity_dashboard(summary, detail, tmp_path / "dashboard.html")
    markdown = write_markdown_report(summary, detail, tmp_path / "report.md")

    html_text = html.read_text(encoding="utf-8")
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "原 NetLogo 50 次 vs 修正版 Mesa 100 次" in html_text
    assert "192 个比较单元" in html_text
    assert "总体结果：1/2" in markdown_text
    assert "教育同质婚比例" in markdown_text
