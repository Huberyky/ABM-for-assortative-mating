"""All-country NetLogo--Mesa parity reporting and self-contained HTML view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .parity import compare_netlogo_mesa


COUNTRY_NAMES = {
    "BE": "比利时", "DE": "德国", "DK": "丹麦", "ES": "西班牙",
    "FI": "芬兰", "FR": "法国", "GR": "希腊", "IE": "爱尔兰",
    "NL": "荷兰", "PT": "葡萄牙", "SE": "瑞典", "UK": "英国",
}

METRIC_NAMES = {
    "perc_hyper_sim": "教育上婚比例",
    "perc_homo_sim": "教育同质婚比例",
    "perc_hypo_sim": "教育下婚比例",
    "a_diff_sim": "夫妻年龄差（年）",
}


def read_design(path: str | Path) -> pd.DataFrame:
    """Read the validation design without importing the Mesa runtime."""
    design = pd.read_csv(path)
    design.columns = [str(column).strip() for column in design.columns]
    if "condition_no" not in design:
        raise ValueError(f"design file lacks condition_no: {path}")
    if design["condition_no"].duplicated().any():
        raise ValueError("condition_no must be unique within a design file")
    return design


def holm_adjust(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm (1979) step-down adjusted p-values; valid under dependence."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ordered = values[order]
    m = len(values)
    adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered
    rejected = adjusted <= alpha
    return rejected, adjusted


def add_familywise_inference(
    detail: pd.DataFrame, *, family_alpha: float = 0.05, power: float = 0.80
) -> pd.DataFrame:
    """Add transparent, multiplicity-controlled evidence to all 192 cells.

    The null tests compare NetLogo and Mesa run-level means.  Holm-adjusted
    p-values control the familywise error rate under arbitrary dependence.
    Bonferroni intervals are conservative simultaneous intervals.  MDE values
    describe resolution at the observed standard error; they are not
    equivalence margins.
    """
    result = detail.copy()
    m = len(result)
    reject, p_adjusted = holm_adjust(
        result["welch_p_value"].to_numpy(), alpha=family_alpha
    )
    result["holm_adjusted_p_value"] = p_adjusted
    result["mean_difference_detected_fwer05"] = reject
    distribution_reject, distribution_p = holm_adjust(
        result["cramer_von_mises_p_value"].to_numpy(), alpha=family_alpha
    )
    variance_reject, variance_p = holm_adjust(
        result["brown_forsythe_p_value"].to_numpy(), alpha=family_alpha
    )
    result["cramer_von_mises_holm_p"] = distribution_p
    result["distribution_difference_detected_fwer05"] = distribution_reject
    result["brown_forsythe_holm_p"] = variance_p
    result["variance_difference_detected_fwer05"] = variance_reject
    per_comparison_alpha = family_alpha / m
    critical = np.array([
        stats.t.ppf(1.0 - per_comparison_alpha / 2.0, df)
        for df in result["welch_degrees_freedom"]
    ])
    result["simultaneous_ci95_low"] = (
        result["difference"] - critical * result["difference_standard_error"]
    )
    result["simultaneous_ci95_high"] = (
        result["difference"] + critical * result["difference_standard_error"]
    )
    # Normal approximation to 80%-power resolution at familywise alpha.
    result["mde80_familywise"] = (
        stats.norm.ppf(1.0 - per_comparison_alpha / 2.0)
        + stats.norm.ppf(power)
    ) * result["difference_standard_error"]
    result["within_netlogo_one_run_sd"] = (
        result["difference"].abs() <= result["netlogo_run_sd"]
    )
    result["inference_label"] = np.where(
        result["distribution_difference_detected_fwer05"],
        "检测到边际分布差异（Cramér–von Mises；Holm FWER 5%）",
        "未检测到边际分布差异（不等于已证明等价）",
    )
    return result


def run_all_country_parity(
    *,
    design_path: str | Path,
    netlogo_dir: str | Path,
    mesa_results: str | Path,
    output_dir: str | Path,
    standardized_tolerance: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Compare every validation condition and write CSV, JSON, and HTML."""
    design = read_design(design_path).sort_values("condition_no")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    mesa_frame = pd.read_csv(mesa_results, usecols=["condition_no", "run_no"])
    run_counts = mesa_frame.groupby("condition_no")["run_no"].nunique()
    if run_counts.nunique() != 1:
        raise ValueError(f"Mesa repetitions differ across conditions: {run_counts.to_dict()}")
    n_mesa_runs = int(run_counts.iloc[0])
    comparison_label = f"50v{n_mesa_runs}"

    for row in design.itertuples(index=False):
        country = str(row.country).strip().strip('"')
        condition = int(row.condition_no)
        stem = f"parity_{country}_condition{condition}_source_faithful_{comparison_label}"
        frame = compare_netlogo_mesa(
            netlogo_dir=netlogo_dir,
            pattern=f"{country}_c{condition}_*.txt",
            mesa_results=mesa_results,
            output_csv=target / f"{stem}.csv",
            output_json=target / f"{stem}.json",
            standardized_tolerance=standardized_tolerance,
        )
        frame.insert(0, "condition_no", condition)
        frame.insert(0, "country_name", COUNTRY_NAMES.get(country, country))
        frame.insert(0, "country", country)
        frame.to_csv(target / f"{stem}.csv", index=False)
        frames.append(frame)

    detail = add_familywise_inference(pd.concat(frames, ignore_index=True))
    detail["metric_name"] = detail["metric"].map(METRIC_NAMES)
    detail["absolute_difference"] = detail["difference"].abs()
    detail["absolute_standardized_difference"] = detail["pooled_standardized_difference"].abs()
    detail["absolute_hedges_g"] = detail["hedges_g"].abs()
    detail["absolute_tolerance_ratio"] = detail["absolute_difference"] / detail["absolute_tolerance"]
    detail_path = target / f"parity_ALL12_source_faithful_{comparison_label}_detail.csv"
    detail.to_csv(detail_path, index=False)

    summary = (
        detail.groupby(["country", "country_name", "condition_no"], as_index=False)
        .agg(
            checks=("legacy_pass", "size"),
            statistically_detected=("mean_difference_detected_fwer05", "sum"),
            distribution_differences=("distribution_difference_detected_fwer05", "sum"),
            variance_differences=("variance_difference_detected_fwer05", "sum"),
            within_netlogo_one_run_sd=("within_netlogo_one_run_sd", "sum"),
            legacy_passed=("legacy_pass", "sum"),
            max_abs_standardized_difference=("absolute_standardized_difference", "max"),
            max_abs_hedges_g=("absolute_hedges_g", "max"),
            min_holm_adjusted_p=("holm_adjusted_p_value", "min"),
            min_distribution_holm_p=("cramer_von_mises_holm_p", "min"),
            max_absolute_tolerance_ratio=("absolute_tolerance_ratio", "max"),
        )
    )
    summary["no_statistically_detected_cells"] = (
        summary["statistically_detected"].eq(0)
        & summary["distribution_differences"].eq(0)
        & summary["variance_differences"].eq(0)
    )
    # Compatibility columns are explicitly legacy, not the current conclusion.
    summary["passed"] = summary["legacy_passed"]
    summary["passed_absolute"] = (
        detail.groupby(["country", "country_name", "condition_no"])["legacy_pass_absolute"]
        .sum().to_numpy()
    )
    summary["passed_standardized"] = (
        detail.groupby(["country", "country_name", "condition_no"])["legacy_pass_standardized"]
        .sum().to_numpy()
    )
    summary["all_passed"] = summary["legacy_passed"].eq(summary["checks"])
    summary_path = target / f"parity_ALL12_source_faithful_{comparison_label}_summary.csv"
    summary.to_csv(summary_path, index=False)

    report = {
        "comparison": (
            f"all validation countries; independent 50 NetLogo vs {n_mesa_runs} Mesa runs"
        ),
        "countries": int(summary["country"].nunique()),
        "checks": int(len(detail)),
        "scientific_conclusion": (
            "Report separately Holm-controlled Welch mean, Cramer-von Mises "
            "distribution, and Brown-Forsythe variance diagnostics; a null result "
            "is not proof of practical equivalence."
        ),
        "statistically_detected_cells_fwer05": int(
            detail["mean_difference_detected_fwer05"].sum()
        ),
        "distribution_differences_detected_fwer05": int(
            detail["distribution_difference_detected_fwer05"].sum()
        ),
        "distribution_cells_with_unadjusted_p_below_05": int(
            (detail["cramer_von_mises_p_value"] < .05).sum()
        ),
        "minimum_distribution_holm_p_value": float(
            detail["cramer_von_mises_holm_p"].min()
        ),
        "variance_differences_detected_fwer05": int(
            detail["variance_difference_detected_fwer05"].sum()
        ),
        "variance_cells_with_unadjusted_p_below_05": int(
            (detail["brown_forsythe_p_value"] < .05).sum()
        ),
        "minimum_variance_holm_p_value": float(
            detail["brown_forsythe_holm_p"].min()
        ),
        "minimum_holm_adjusted_p_value": float(detail["holm_adjusted_p_value"].min()),
        "cells_with_unadjusted_p_below_05": int((detail["welch_p_value"] < .05).sum()),
        "legacy_heuristic_passed": int(detail["legacy_pass"].sum()),
        "legacy_heuristic_status": (
            "retained for provenance only; thresholds are absent from the paper and original R"
        ),
        "max_absolute_standardized_difference": float(
            detail["absolute_standardized_difference"].max()
        ),
        "statistically_detected_cells": detail.loc[
            detail["mean_difference_detected_fwer05"],
            ["country", "ten_cohort", "metric", "difference",
             "hedges_g", "welch_p_value", "holm_adjusted_p_value"],
        ].to_dict("records"),
    }
    (target / f"parity_ALL12_source_faithful_{comparison_label}_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    html_path = target / f"parity_ALL12_source_faithful_{comparison_label}_dashboard.html"
    write_parity_dashboard(summary, detail, html_path)
    write_markdown_report(
        summary,
        detail,
        target / f"parity_ALL12_source_faithful_{comparison_label}_report.md",
    )
    return summary, detail, html_path


def write_parity_dashboard(
    summary: pd.DataFrame, detail: pd.DataFrame, output_html: str | Path
) -> Path:
    """Write a dependency-free dashboard that also works from a file link."""
    summary_records = json.dumps(summary.to_dict("records"), ensure_ascii=False)
    detail_records = json.dumps(detail.to_dict("records"), ensure_ascii=False)
    n_netlogo = int(detail["n_netlogo"].iloc[0])
    n_mesa = int(detail["n_mesa"].iloc[0])
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Grow–Van Bavel NetLogo–Mesa 全 12 国统计验证</title>
<style>
:root{{--bg:#f4f7fb;--card:#fff;--ink:#172033;--muted:#64748b;--ok:#16845b;--bad:#c43d4b;--blue:#315efb}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1480px;margin:auto;padding:28px}} h1{{margin:0 0 4px;font-size:27px}} .sub{{color:var(--muted);margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px;margin:18px 0}} .card{{background:var(--card);padding:17px;border-radius:12px;box-shadow:0 3px 16px #1f293712}}
.value{{font-size:25px;font-weight:750}} .label{{color:var(--muted)}} .panel{{background:var(--card);padding:18px;border-radius:12px;margin:14px 0;box-shadow:0 3px 16px #1f293712}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px 10px;border-bottom:1px solid #e8edf4;text-align:right;white-space:nowrap}} th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}} th{{position:sticky;top:0;background:#f8fafc}}
.scroll{{max-height:620px;overflow:auto}} .ok{{color:var(--ok);font-weight:700}} .bad{{color:var(--bad);font-weight:700}} .bar{{height:8px;background:#e8edf4;border-radius:5px;overflow:hidden;min-width:120px}} .fill{{height:100%;background:var(--blue)}}
.filters{{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px}} select{{padding:7px 10px;border:1px solid #cbd5e1;border-radius:7px;background:white}} @media(max-width:760px){{.cards{{grid-template-columns:1fr 1fr}} main{{padding:14px}}}}
</style></head><body><main>
<h1>NetLogo–Mesa 全 12 国统计验证</h1><div class="sub">原 NetLogo {n_netlogo} 次 vs 修正版 Mesa {n_mesa} 次。主结论采用 Welch 均值检验 + Holm 192 项家族校正（FWER 5%）；“未检测到差异”不等于“已证明等价”。论文原口径的均值 ± 1 次运行标准差作为描述性参照。</div>
<div class="cards" id="cards"></div>
<section class="panel"><h2>如何阅读</h2><p>Welch 检验均值，Cramér–von Mises 检验整个边际分布，Brown–Forsythe 检验运行间方差；三组各自对 192 项作 Holm 校正并控制 FWER 5%。未检测到差异只能说明在当前 50 vs 100 次重复与检验能力下没有反证，不能证明实质等价。Hedges g 不设人为通过线；MDE80 是当前设计的识别精度，不是等价门槛。</p></section>
<section class="panel"><h2>逐国概览</h2><div class="scroll"><table><thead><tr><th>国家</th><th>条件</th><th>均值差异</th><th>分布差异</th><th>方差差异</th><th>在 NetLogo ±1 SD 内</th><th>最大 |Hedges g|</th><th>旧诊断（仅溯源）</th></tr></thead><tbody id="summary"></tbody></table></div></section>
<section class="panel"><h2>192 个比较单元</h2><div class="filters"><select id="country"><option value="">全部国家</option></select><select id="metric"><option value="">全部指标</option></select><select id="status"><option value="">全部状态</option><option value="detected">检测到分布差异</option><option value="not_detected">未检测到分布差异</option></select></div><div class="scroll"><table><thead><tr><th>国家</th><th>出生队列</th><th>指标</th><th>NetLogo</th><th>Mesa</th><th>Mesa−NetLogo</th><th>Hedges g</th><th>均值 Holm p</th><th>分布 Holm p</th><th>同时 95% CI</th><th>MDE80</th><th>结论</th></tr></thead><tbody id="detail"></tbody></table></div></section>
<script>
const summaries={summary_records}; const details={detail_records};
const fmt=x=>Number(x).toFixed(3); const total=details.length, detected=details.filter(x=>x.mean_difference_detected_fwer05).length;
const unadjusted=details.filter(x=>x.welch_p_value<.05).length; const maxG=Math.max(...details.map(x=>x.absolute_hedges_g));
const distDetected=details.filter(x=>x.distribution_difference_detected_fwer05).length;
document.getElementById('cards').innerHTML=[['国家',new Set(details.map(x=>x.country)).size],['比较单元',total],['均值差异（Holm）',detected+'/'+total],['分布差异（Holm）',distDetected+'/'+total]].map(x=>`<div class="card"><div class="value">${{x[1]}}</div><div class="label">${{x[0]}}</div></div>`).join('');
document.getElementById('summary').innerHTML=summaries.map(x=>`<tr><td>${{x.country_name}} (${{x.country}})</td><td>${{x.condition_no}}</td><td class="${{x.statistically_detected===0?'ok':'bad'}}">${{x.statistically_detected}}/${{x.checks}}</td><td class="${{x.distribution_differences===0?'ok':'bad'}}">${{x.distribution_differences}}/${{x.checks}}</td><td class="${{x.variance_differences===0?'ok':'bad'}}">${{x.variance_differences}}/${{x.checks}}</td><td>${{x.within_netlogo_one_run_sd}}/${{x.checks}}</td><td>${{fmt(x.max_abs_hedges_g)}}</td><td>${{x.legacy_passed}}/${{x.checks}}</td></tr>`).join('');
const country=document.getElementById('country'),metric=document.getElementById('metric'),status=document.getElementById('status');
[...new Set(details.map(x=>x.country))].forEach(v=>country.insertAdjacentHTML('beforeend',`<option>${{v}}</option>`)); [...new Set(details.map(x=>x.metric_name))].forEach(v=>metric.insertAdjacentHTML('beforeend',`<option>${{v}}</option>`));
function render(){{const rows=details.filter(x=>(!country.value||x.country===country.value)&&(!metric.value||x.metric_name===metric.value)&&(!status.value||(status.value==='detected')===x.distribution_difference_detected_fwer05)); document.getElementById('detail').innerHTML=rows.map(x=>`<tr><td>${{x.country_name}} (${{x.country}})</td><td>${{x.ten_cohort}}</td><td>${{x.metric_name}}</td><td>${{fmt(x.netlogo_mean)}}</td><td>${{fmt(x.mesa_mean)}}</td><td>${{fmt(x.difference)}}</td><td>${{fmt(x.hedges_g)}}</td><td>${{Number(x.holm_adjusted_p_value).toPrecision(3)}}</td><td>${{Number(x.cramer_von_mises_holm_p).toPrecision(3)}}</td><td>[${{fmt(x.simultaneous_ci95_low)}}, ${{fmt(x.simultaneous_ci95_high)}}]</td><td>${{fmt(x.mde80_familywise)}}</td><td class="${{x.distribution_difference_detected_fwer05?'bad':'ok'}}">${{x.distribution_difference_detected_fwer05?'检测到分布差异':'未检测到（非等价证明）'}}</td></tr>`).join('')}}
[country,metric,status].forEach(x=>x.addEventListener('change',render)); render();
</script></main></body></html>"""
    target = Path(output_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def write_markdown_report(
    summary: pd.DataFrame, detail: pd.DataFrame, output_md: str | Path
) -> Path:
    """Write the full 192-cell comparison in a readable audit document."""
    n_netlogo = int(detail["n_netlogo"].iloc[0])
    n_mesa = int(detail["n_mesa"].iloc[0])
    passed = int(detail["pass"].sum())
    absolute_passed = int(detail["pass_absolute"].sum())
    standardized_passed = int(detail["pass_standardized"].sum())
    has_inference = "holm_adjusted_p_value" in detail.columns
    detected = int(detail["mean_difference_detected_fwer05"].sum()) if has_inference else None
    lines = [
        "# 全 12 国 NetLogo–Mesa 统计验证报告",
        "",
        f"比较口径：每国原 NetLogo {n_netlogo} 次与修正版 Mesa {n_mesa} 次独立重复；",
        "主分析比较运行级均值，采用 Welch 检验并对全部 192 项使用 Holm 校正，"
        "将家族第一类错误率控制为 5%。",
        "未检测到差异不等于证明实质等价；论文和原始 R 脚本没有规定可用于代码移植的 SESOI。",
        "",
    ]
    if has_inference:
        lines.extend([
            f"正式结果：Holm 校正后检测到均值差异 {detected}/{len(detail)}；"
            f"未经校正 p<.05 为 {int((detail['welch_p_value'] < .05).sum())}/{len(detail)}。",
            f"Cramér–von Mises 边际分布差异为 "
            f"{int(detail['distribution_difference_detected_fwer05'].sum())}/{len(detail)}；"
            f"Brown–Forsythe 运行间方差差异为 "
            f"{int(detail['variance_difference_detected_fwer05'].sum())}/{len(detail)}。",
            f"最小 Holm 校正 p={detail['holm_adjusted_p_value'].min():.6g}；"
            f"最大 |Hedges g|={detail['absolute_hedges_g'].max():.6g}。",
            "",
            "旧工程阈值（比例 0.04、年龄 0.35 年、标准化差 0.35）只为保留历史溯源，"
            "不是论文规则，也不再产生正式的等价结论。",
        ])
    lines.extend([
        "",
        f"旧口径总体结果：{passed}/{len(detail)}；绝对差 {absolute_passed}/{len(detail)}；"
        f"标准化差 {standardized_passed}/{len(detail)}。",
        "",
        "## 逐国汇总",
        "",
        "| 国家 | 条件 | 双门槛 | 绝对差 | 标准化差 | 最大绝对标准化差 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.country_name} ({row.country}) | {row.condition_no} | "
            f"{int(row.passed)}/{int(row.checks)} | "
            f"{int(row.passed_absolute)}/{int(row.checks)} | "
            f"{int(row.passed_standardized)}/{int(row.checks)} | "
            f"{row.max_abs_standardized_difference:.6f} |"
        )

    failures = detail.loc[~detail["pass"]]
    lines.extend([
        "", "## 未通过单元", "",
        "| 国家 | 出生队列 | 指标 | NetLogo | Mesa | Mesa−NetLogo | 标准化差 | 绝对差通过 | 标准化通过 |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ])
    for row in failures.itertuples(index=False):
        lines.append(
            f"| {row.country_name} ({row.country}) | {row.ten_cohort} | {row.metric_name} | "
            f"{row.netlogo_mean:.6f} | {row.mesa_mean:.6f} | {row.difference:.6f} | "
            f"{row.pooled_standardized_difference:.6f} | "
            f"{'是' if row.pass_absolute else '否'} | {'是' if row.pass_standardized else '否'} |"
        )

    if has_inference:
        lines.extend([
            "", "## 正式的 192 项同时推断", "",
            "| 国家 | 出生队列 | 指标 | 差值 | Hedges g | Welch Holm p | C-vM Holm p | BF Holm p | 家族同时 95% CI | MDE80 | 结论 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in detail.itertuples(index=False):
            lines.append(
                f"| {row.country} | {row.ten_cohort} | {row.metric_name} | "
                f"{row.difference:.6f} | {row.hedges_g:.6f} | "
                f"{row.holm_adjusted_p_value:.6g} | {row.cramer_von_mises_holm_p:.6g} | "
                f"{row.brown_forsythe_holm_p:.6g} | "
                f"[{row.simultaneous_ci95_low:.6f}, {row.simultaneous_ci95_high:.6f}] | "
                f"{row.mde80_familywise:.6f} | {row.inference_label} |"
            )

    lines.extend([
        "", "## 全部 192 个比较单元", "",
        "| 国家 | 出生队列 | 指标 | NetLogo | Mesa | Mesa−NetLogo | 标准化差 | 绝对差通过 | 标准化通过 | 最终 |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ])
    for row in detail.to_dict("records"):
        lines.append(
            f"| {row['country']} | {row['ten_cohort']} | {row['metric_name']} | "
            f"{row['netlogo_mean']:.6f} | {row['mesa_mean']:.6f} | "
            f"{row['difference']:.6f} | {row['pooled_standardized_difference']:.6f} | "
            f"{'是' if row['pass_absolute'] else '否'} | "
            f"{'是' if row['pass_standardized'] else '否'} | "
            f"{'通过' if row['pass'] else '未通过'} |"
        )
    target = Path(output_md)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
