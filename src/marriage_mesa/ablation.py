"""Paper-faithful effect-assessment (ablation) reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import read_netlogo_outputs, summarize_census


SOURCE_ORDER = ["ESS", "full", "no_s", "no_delta", "no_a", "no_y"]
TYPE_ORDER = ["hypergamic", "homogamic", "hypogamic"]
SOURCE_NAMES = {
    "ESS": "ESS 实际数据",
    "full": "完整模型",
    "no_s": "关闭教育偏好（wsᵐ=wsᶠ=0）",
    "no_delta": "取消学校相遇结构（δ=0.5）",
    "no_a": "关闭年龄偏好（waᵐ=waᶠ=0）",
    "no_y": "关闭收入偏好（wyᵐ=wyᶠ=0）",
}
TYPE_NAMES = {
    "hypergamic": "教育上婚",
    "homogamic": "教育同质婚",
    "hypogamic": "教育下婚",
}
PAPER_TABLE3 = {
    "ESS": {"hypergamic": (.14, .05), "homogamic": (.71, .06), "hypogamic": (.15, .06)},
    "full": {"hypergamic": (.13, .05), "homogamic": (.71, .06), "hypogamic": (.16, .04)},
    "no_s": {"hypergamic": (.18, .05), "homogamic": (.60, .07), "hypogamic": (.22, .05)},
    "no_delta": {"hypergamic": (.16, .05), "homogamic": (.64, .07), "hypogamic": (.21, .05)},
    "no_a": {"hypergamic": (.12, .04), "homogamic": (.75, .05), "hypogamic": (.13, .03)},
    "no_y": {"hypergamic": (.09, .04), "homogamic": (.79, .04), "hypogamic": (.12, .04)},
}


def classify_ablation(row: pd.Series | dict) -> str:
    """Classify one original design row without relying on condition numbers."""
    get = row.get
    if float(get("ws_m")) == 0 and float(get("ws_f")) == 0:
        return "no_s"
    if float(get("wy_m")) == 0 and float(get("wy_f")) == 0:
        return "no_y"
    if float(get("wa_m")) == 0 and float(get("wa_f")) == 0:
        return "no_a"
    if float(get("delta")) == 0.5:
        return "no_delta"
    return "full"


def extract_netlogo_ablation_runs(
    *,
    validation_design: str | Path,
    effect_design: str | Path,
    validation_output_dir: str | Path,
    effect_output_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Convert all archived NetLogo effect runs to the common run-level CSV."""
    designs = [
        (pd.read_csv(validation_design), Path(validation_output_dir), "full"),
        (pd.read_csv(effect_design), Path(effect_output_dir), None),
    ]
    frames: dict[str, list[pd.DataFrame]] = {source: [] for source in SOURCE_ORDER[1:]}
    for design, source_dir, fixed_source in designs:
        for row in design.to_dict("records"):
            source = fixed_source or classify_ablation(row)
            country = str(row["country"]).strip().strip('"')
            condition = int(row["condition_no"])
            raw = read_netlogo_outputs(source_dir, f"{country}_c{condition}_*.txt")
            summary = summarize_census(raw.to_dict("records"))
            frames[source].append(summary)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for source, source_frames in frames.items():
        combined = pd.concat(source_frames, ignore_index=True)
        path = target / f"netlogo_{source}_50runs.csv"
        combined.to_csv(path, index=False)
        paths[source] = path
    return paths


def cluster_bootstrap_effects(
    country_cohort: pd.DataFrame, *, repetitions: int = 10_000, seed: int = 20260818
) -> pd.DataFrame:
    """Country-cluster bootstrap of each ablation minus the full model.

    Countries, rather than 48 country/cohort rows, are resampled so dependence
    among the four cohorts of one country is retained.  These intervals extend
    the paper's descriptive table; they do not alter its aggregation.
    """
    required = {
        "country", "ten_cohort", "source", "perc_hyper_sim",
        "perc_homo_sim", "perc_hypo_sim",
    }
    missing = required.difference(country_cohort.columns)
    if missing:
        raise ValueError(f"country/cohort file lacks columns: {sorted(missing)}")
    metrics = {
        "hypergamic": "perc_hyper_sim",
        "homogamic": "perc_homo_sim",
        "hypogamic": "perc_hypo_sim",
    }
    simulated = country_cohort.loc[country_cohort["source"].ne("ESS")].copy()
    countries = np.array(sorted(simulated["country"].unique()))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    full = simulated.loc[simulated["source"].eq("full")]
    for source in SOURCE_ORDER[2:]:
        ablated = simulated.loc[simulated["source"].eq(source)]
        for couple_type, metric in metrics.items():
            paired = full[["country", "ten_cohort", metric]].merge(
                ablated[["country", "ten_cohort", metric]],
                on=["country", "ten_cohort"], suffixes=("_full", "_ablated"),
                validate="one_to_one",
            )
            paired["difference"] = paired[f"{metric}_ablated"] - paired[f"{metric}_full"]
            by_country = paired.groupby("country", observed=True)["difference"].mean()
            draws = rng.choice(countries, size=(repetitions, len(countries)), replace=True)
            values = by_country.reindex(countries).to_numpy()
            indices = np.searchsorted(countries, draws)
            bootstrap = values[indices].mean(axis=1)
            rows.append({
                "source": source,
                "couple_type": couple_type,
                "difference_vs_full": paired["difference"].mean(),
                "difference_percentage_points": 100 * paired["difference"].mean(),
                "country_cluster_ci95_low": np.quantile(bootstrap, .025),
                "country_cluster_ci95_high": np.quantile(bootstrap, .975),
                "countries": len(countries),
                "country_cohort_cells": len(paired),
                "bootstrap_repetitions": repetitions,
            })
    return pd.DataFrame(rows)


def compare_with_paper(table_long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in table_long.itertuples(index=False):
        paper_mean, paper_sd = PAPER_TABLE3[str(row.source)][str(row.couple_type)]
        rows.append({
            "source": row.source,
            "couple_type": row.couple_type,
            "mesa_mean": row.mean,
            "paper_mean": paper_mean,
            "mean_difference": row.mean - paper_mean,
            "mesa_sd": row.sd,
            "paper_sd": paper_sd,
            "sd_difference": row.sd - paper_sd,
            "same_after_two_decimal_rounding": (
                round(float(row.mean) + 1e-12, 2) == paper_mean
                and round(float(row.sd) + 1e-12, 2) == paper_sd
            ),
        })
    return pd.DataFrame(rows)


def compare_table3_engines(mesa: pd.DataFrame, netlogo: pd.DataFrame) -> pd.DataFrame:
    keys = ["source", "couple_type"]
    merged = mesa.merge(
        netlogo, on=keys, suffixes=("_mesa", "_netlogo"), validate="one_to_one"
    )
    merged["mean_difference_mesa_minus_netlogo"] = merged["mean_mesa"] - merged["mean_netlogo"]
    merged["sd_difference_mesa_minus_netlogo"] = merged["sd_mesa"] - merged["sd_netlogo"]
    merged["same_after_two_decimal_rounding"] = (
        merged["mean_mesa"].round(2).eq(merged["mean_netlogo"].round(2))
        & merged["sd_mesa"].round(2).eq(merged["sd_netlogo"].round(2))
    )
    return merged


def write_ablation_dashboard(
    table_long: pd.DataFrame,
    effects: pd.DataFrame,
    paper_comparison: pd.DataFrame,
    output_html: str | Path,
) -> Path:
    table_records = json.dumps(table_long.to_dict("records"), ensure_ascii=False)
    effect_records = json.dumps(effects.to_dict("records"), ensure_ascii=False)
    comparison_records = json.dumps(paper_comparison.to_dict("records"), ensure_ascii=False)
    source_names = json.dumps(SOURCE_NAMES, ensure_ascii=False)
    type_names = json.dumps(TYPE_NAMES, ensure_ascii=False)
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Grow–Van Bavel Table 3 Mesa 消融实验</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#667085;--blue:#315efb;--red:#ca4050;--green:#16845b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1380px;margin:auto;padding:28px}}h1{{margin:0;font-size:28px}}.sub{{color:var(--muted);margin:5px 0 20px}}
.panel{{background:var(--card);padding:20px;border-radius:13px;margin:14px 0;box-shadow:0 3px 16px #1f293712}}table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px;border-bottom:1px solid #dfe4ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#f8fafc}}.source{{font-weight:650}}
.chart{{display:grid;gap:12px}}.chartrow{{display:grid;grid-template-columns:260px 1fr 72px;gap:12px;align-items:center}}.track{{height:22px;background:#edf1f6;border-radius:5px;position:relative;overflow:visible}}
.zero{{position:absolute;left:50%;top:-4px;bottom:-4px;border-left:1px solid #94a3b8}}.bar{{position:absolute;top:4px;height:14px;border-radius:3px;background:var(--blue)}}.ci{{position:absolute;top:10px;border-top:2px solid #172033;height:1px}}
.ok{{color:var(--green);font-weight:700}}.warn{{color:var(--red);font-weight:700}}.note{{color:var(--muted)}}
@media(max-width:760px){{main{{padding:12px}}.chartrow{{grid-template-columns:150px 1fr 56px}}.scroll{{overflow:auto}}}}
</style></head><body><main><h1>婚姻匹配机制消融实验</h1>
<div class="sub">严格使用原论文 Table 3 的五种模型版本；12 国 × 4 出生队列。均值和 SD 均跨 48 个国家—队列单元计算。</div>
<section class="panel"><h2>原论文式 Table 3</h2><div class="scroll"><table><thead><tr><th rowspan="2">来源</th><th colspan="2">教育上婚</th><th colspan="2">教育同质婚</th><th colspan="2">教育下婚</th></tr><tr><th>Mean</th><th>SD</th><th>Mean</th><th>SD</th><th>Mean</th><th>SD</th></tr></thead><tbody id="table"></tbody></table></div><p class="note">SD 是 12 国 × 4 队列之间的标准差，不是 50 次运行的标准差。ESS 行来自原 ESS_results.RData。</p></section>
<section class="panel"><h2>相对完整模型的变化</h2><p class="note">横轴范围 −15 至 +15 个百分点；线段为按国家聚类重抽样的 95% CI（10,000 次）。</p><div class="chart" id="chart"></div></section>
<section class="panel"><h2>与论文 Table 3 的两位小数核对</h2><p id="parity"></p><div class="scroll"><table><thead><tr><th>来源</th><th>婚配类型</th><th>Mesa Mean</th><th>论文 Mean</th><th>Mesa SD</th><th>论文 SD</th><th>两位小数</th></tr></thead><tbody id="compare"></tbody></table></div></section>
<section class="panel"><h2>解释边界</h2><p>单因素消融回答的是“在该模型及其他机制固定时，关闭某机制会怎样改变模型输出”，不是现实世界中的自然实验或独立因果识别。教育机会结构消融仅令 δ=0.5，教育人口构成本身仍保留。</p></section>
<script>const T={table_records},E={effect_records},C={comparison_records},SN={source_names},TN={type_names};
const fmt=x=>Number(x).toFixed(3),get=(s,t,k)=>T.find(x=>x.source===s&&x.couple_type===t)[k];
document.getElementById('table').innerHTML={json.dumps(SOURCE_ORDER)}.map(s=>`<tr><td class="source">${{SN[s]}}</td>${{{json.dumps(TYPE_ORDER)}.map(t=>`<td>${{fmt(get(s,t,'mean'))}}</td><td>${{fmt(get(s,t,'sd'))}}</td>`).join('')}}</tr>`).join('');
const scale=x=>50+(x/15)*50;document.getElementById('chart').innerHTML=E.map(x=>{{const v=x.difference_percentage_points,lo=100*x.country_cluster_ci95_low,hi=100*x.country_cluster_ci95_high;const left=Math.min(50,scale(v)),width=Math.abs(scale(v)-50);return `<div class="chartrow"><div>${{SN[x.source]}} · ${{TN[x.couple_type]}}</div><div class="track"><span class="zero"></span><span class="bar" style="left:${{left}}%;width:${{width}}%;background:${{v>=0?'var(--blue)':'var(--red)'}}"></span><span class="ci" style="left:${{scale(lo)}}%;width:${{scale(hi)-scale(lo)}}%"></span></div><div>${{v>=0?'+':''}}${{v.toFixed(1)}} pp</div></div>`}}).join('');
const matched=C.filter(x=>x.same_after_two_decimal_rounding).length;document.getElementById('parity').innerHTML=`两位小数同时匹配 Mean 和 SD：<b>${{matched}}/${{C.length}}</b> 个来源×婚配类型单元。`;
document.getElementById('compare').innerHTML=C.map(x=>`<tr><td>${{SN[x.source]}}</td><td>${{TN[x.couple_type]}}</td><td>${{fmt(x.mesa_mean)}}</td><td>${{x.paper_mean.toFixed(2)}}</td><td>${{fmt(x.mesa_sd)}}</td><td>${{x.paper_sd.toFixed(2)}}</td><td class="${{x.same_after_two_decimal_rounding?'ok':'warn'}}">${{x.same_after_two_decimal_rounding?'一致':'有差异'}}</td></tr>`).join('');</script></main></body></html>"""
    target = Path(output_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def build_ablation_report(
    *, table_long_csv: str | Path, country_cohort_csv: str | Path,
    output_dir: str | Path, netlogo_table_long_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    table_long = pd.read_csv(table_long_csv)
    country_cohort = pd.read_csv(country_cohort_csv)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    effects = cluster_bootstrap_effects(country_cohort)
    comparison = compare_with_paper(table_long)
    effects.to_csv(target / "mesa_ablation_effects_vs_full.csv", index=False)
    comparison.to_csv(target / "mesa_table3_vs_paper.csv", index=False)
    engine_comparison = None
    if netlogo_table_long_csv is not None:
        engine_comparison = compare_table3_engines(
            table_long, pd.read_csv(netlogo_table_long_csv)
        )
        engine_comparison.to_csv(target / "mesa_vs_netlogo_table3.csv", index=False)
    dashboard = write_ablation_dashboard(
        table_long, effects, comparison, target / "mesa_table3_ablation_dashboard.html"
    )
    summary = {
        "sources": SOURCE_ORDER,
        "countries": int(country_cohort["country"].nunique()),
        "cohorts": int(country_cohort["ten_cohort"].nunique()),
        "paper_cells_matching_after_two_decimal_rounding": int(
            comparison["same_after_two_decimal_rounding"].sum()
        ),
        "paper_cells_total": int(len(comparison)),
        "largest_absolute_mean_difference_from_paper": float(
            comparison["mean_difference"].abs().max()
        ),
    }
    if engine_comparison is not None:
        summary.update({
            "mesa_netlogo_cells_matching_after_two_decimal_rounding": int(
                engine_comparison["same_after_two_decimal_rounding"].sum()
            ),
            "mesa_netlogo_cells_total": int(len(engine_comparison)),
            "largest_absolute_mean_difference_mesa_minus_netlogo": float(
                engine_comparison["mean_difference_mesa_minus_netlogo"].abs().max()
            ),
        })
    (target / "mesa_table3_ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return effects, comparison, dashboard
