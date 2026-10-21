"""Static, dependency-free report for the calibrated China replication."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import ModelConfig
from .analysis import aggregate_conditions, calibration_loss


METRICS = {
    "perc_hyper_sim": ("perc_hyper_obs", "perc_hyper_obs_se_kish", "教育上婚", "share"),
    "perc_homo_sim": ("perc_homo_obs", "perc_homo_obs_se_kish", "教育同质婚", "share"),
    "perc_hypo_sim": ("perc_hypo_obs", "perc_hypo_obs_se_kish", "教育下婚", "share"),
    "a_diff_sim": ("a_diff_obs", "a_diff_obs_se_kish", "丈夫−妻子年龄差", "years"),
}
SOURCE_NAMES = {
    "CGSS": "CGSS 2021",
    "full": "完整模型",
    "no_s": "关闭教育偏好",
    "no_delta": "取消学校相遇结构（δ=0.5）",
    "no_a": "关闭年龄偏好",
    "no_y": "关闭收入前景偏好",
}
TYPE_NAMES = {
    "hypergamic": "教育上婚",
    "homogamic": "教育同质婚",
    "hypogamic": "教育下婚",
}


def validation_table(runs: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    selected = runs.loc[runs["condition_no"].eq(1)].copy()
    if selected["run_no"].nunique() != 50:
        raise ValueError("independent China validation requires 50 full-model runs")
    rows = []
    critical = stats.t.ppf(.975, 49)
    for cohort, group in selected.groupby("ten_cohort", observed=True):
        observed = targets.loc[targets["ten_cohort"].eq(cohort)]
        if len(observed) != 1:
            raise ValueError(f"target key is not unique for {cohort}")
        observed = observed.iloc[0]
        for sim, (obs, obs_se_column, label, unit) in METRICS.items():
            values = group[sim].to_numpy(dtype=float)
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            se = sd / np.sqrt(len(values))
            target = float(observed[obs])
            observed_se = float(observed[obs_se_column])
            combined_se = float(np.sqrt(se**2 + observed_se**2))
            z = (mean - target) / combined_se
            rows.append({
                "ten_cohort": cohort,
                "metric": label,
                "unit": unit,
                "observed": target,
                "model_mean": mean,
                "model_run_sd": sd,
                "model_mc_se": se,
                "model_mean_ci95_low": mean - critical * se,
                "model_mean_ci95_high": mean + critical * se,
                "observed_se_kish_approx": observed_se,
                "observed_ci95_kish_low": target - 1.96 * observed_se,
                "observed_ci95_kish_high": target + 1.96 * observed_se,
                "difference_model_minus_observed": mean - target,
                "combined_standardized_discrepancy_z": z,
                "two_sided_compatibility_p_unadjusted": float(2 * stats.norm.sf(abs(z))),
                "observed_within_model_mean_plus_minus_one_run_sd": (
                    mean - sd <= target <= mean + sd
                ),
                "runs": len(values),
            })
    result = pd.DataFrame(rows)
    order = np.argsort(result["two_sided_compatibility_p_unadjusted"].to_numpy())
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(result) - rank) * result.loc[index, "two_sided_compatibility_p_unadjusted"])
        running = max(running, value)
        adjusted[index] = running
    result["two_sided_compatibility_p_holm_24"] = adjusted
    result["incompatibility_detected_holm_5pct"] = adjusted < .05
    return result


def parameter_comparison(parameters: pd.DataFrame) -> pd.DataFrame:
    if len(parameters) != 1:
        raise ValueError("parameter file must contain one calibrated row")
    source = ModelConfig()
    labels = {
        "ws_f": "女性教育偏好权重", "ws_m": "男性教育偏好权重",
        "wy_f": "女性收入前景偏好权重", "wy_m": "男性收入前景偏好权重",
        "wa_f": "女性年龄偏好权重", "wa_m": "男性年龄偏好权重",
        "delta": "学校结构相遇概率",
    }
    row = parameters.iloc[0]
    return pd.DataFrame([{
        "parameter": name,
        "meaning": meaning,
        "paper_transferred_value": getattr(source, name),
        "china_calibrated_value": float(row[name]),
    } for name, meaning in labels.items()])


def _records(frame: pd.DataFrame) -> str:
    clean = frame.replace({np.nan: None})
    return json.dumps(clean.to_dict("records"), ensure_ascii=False)


def write_china_dashboard(
    *,
    validation: pd.DataFrame,
    parameters: pd.DataFrame,
    table_long: pd.DataFrame,
    effects: pd.DataFrame,
    fit_summary: dict,
    input_audit: pd.DataFrame,
    exclusion_audit: pd.DataFrame,
    output_html: str | Path,
) -> Path:
    source_names = json.dumps(SOURCE_NAMES, ensure_ascii=False)
    type_names = json.dumps(TYPE_NAMES, ensure_ascii=False)
    html = fr"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>中国婚姻匹配 Mesa ABM</title>
<style>
:root{{--bg:#f3f5f8;--card:#fff;--ink:#172033;--muted:#667085;--blue:#315efb;--red:#c73e4d;--green:#13845b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 system-ui,"Microsoft YaHei",sans-serif}}
main{{max-width:1400px;margin:auto;padding:28px}}h1{{margin:0;font-size:28px}}h2{{margin-top:0}}.sub,.note{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}.panel{{background:var(--card);padding:20px;border-radius:13px;margin:14px 0;box-shadow:0 3px 16px #1f293712}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e1e6ee;text-align:right}}th:first-child,td:first-child{{text-align:left}}thead th{{background:#f8fafc}}.scroll{{overflow:auto}}
.chart{{display:grid;gap:10px}}.row{{display:grid;grid-template-columns:280px 1fr 70px;gap:10px;align-items:center}}.track{{height:22px;background:#edf1f6;border-radius:5px;position:relative}}.zero{{position:absolute;left:50%;top:-3px;bottom:-3px;border-left:1px solid #94a3b8}}.bar{{position:absolute;top:4px;height:14px;border-radius:3px}}.ok{{color:var(--green);font-weight:700}}.warn{{color:var(--red);font-weight:700}}
svg{{width:100%;height:360px}}@media(max-width:850px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}.row{{grid-template-columns:170px 1fr 58px}}}}
</style></head><body><main><h1>中国语境婚姻匹配 ABM：校准、验证与消融</h1>
<p class="sub">Grow–Van Bavel (2015) Mesa 科学内核不变；中国教育/收入前景输入，δ=0.9；每条件 50 次独立模拟。</p>
<section class="panel"><h2>CGSS 与独立 50 次完整模型</h2><div id="metric"></div><p class="note">蓝色误差线是模型均值的 Monte Carlo 95% t 区间。CGSS 不确定性另以 Kish 有效样本量作近似；这不是完整复杂抽样方差。24 个“队列×指标”的兼容性检验统一采用 Holm 校正，但“不拒绝差异”不构成等价性证明。主规格行为参数沿用论文校准值，未使用 CGSS 再估，因此这里是对“参数迁移+中国机会结构”规格的外部经验对照；失败的中国行为重估另作透明诊断。</p></section>
<div class="grid"><section class="panel"><h2>行为参数与规格选择</h2><div class="scroll"><table><thead><tr><th>参数</th><th>含义</th><th>论文值</th><th>主规格值</th></tr></thead><tbody id="params"></tbody></table></div><p class="note">主规格（仅替换中国数据）独立 50 次实际 fit：{fit_summary['primary_actual_fit']:.4f}；另一独立迁移种子块 fit：{fit_summary['paper_parameter_transfer_fit']:.4f}。中国行为重估的最佳实际 CCD 条件在校准批次 fit={fit_summary['behavior_candidate_calibration_fit']:.4f}，独立验证降至 {fit_summary['behavior_candidate_validation_fit']:.4f}；被否决的 WLS 边界候选曾预测 fit={fit_summary['rejected_rsm_predicted_fit']:.4f}，独立验证仅 {fit_summary['rejected_rsm_validation_fit']:.4f}。因此主规格遵循“唯一区别在数据”并保留论文行为参数。fit 越大越好。</p></section>
<section class="panel"><h2>数据支持范围</h2><div id="coverage"></div><p class="note" id="excluded"></p></section></div>
<section class="panel"><h2>中国版 Table 3（跨 6 个出生队列）</h2><div class="scroll"><table><thead><tr><th>来源</th><th>婚配类型</th><th>Mean</th><th>SD</th></tr></thead><tbody id="table3"></tbody></table></div><p class="note">先对每个“条件×运行×队列”计算比例，再在 50 次运行上取均值；此处 Mean/SD 跨六个中国队列计算。它与论文跨 12 国×4 队列的统计层级相同，但样本单元数不同，不能把 SD 数值直接横向比较。</p></section>
<section class="panel"><h2>单机制消融相对完整模型</h2><div class="chart" id="effects"></div><p class="note">每次只关闭一个机制；教育机会结构消融仅令 δ=0.5，中国教育人口构成本身仍保留。范围 −20 至 +20 个百分点。</p></section>
<section class="panel"><h2>解释边界</h2><p>这些消融识别的是模型内部机制贡献，不是现实世界自然实验。教育数据仅有 1965–1990 三个队列锚点，收入前景数据到 1985；区间内线性插值，区间外边界保持。因而早期和晚期队列结果依赖显式外推假设，校准参数应理解为“给定这些输入处理与模型规则下”的条件性参数。</p></section>
<script>
const V={_records(validation)},P={_records(parameters)},T={_records(table_long)},E={_records(effects)},A={_records(input_audit)},X={_records(exclusion_audit)},SN={source_names},TN={type_names};
const f=(x,n=3)=>Number(x).toFixed(n);const metricNames=[...new Set(V.map(x=>x.metric))];let active=metricNames[0];
function drawMetric(){{const d=V.filter(x=>x.metric===active),share=d[0].unit==='share';const w=1100,h=330,p=50,max=share?1:Math.max(5,...d.flatMap(x=>[x.observed,x.model_mean_ci95_high]));const xs=i=>p+i*(w-2*p)/(d.length-1),ys=v=>h-p-v*(h-2*p)/max;let svg=`<div><select id="pick">${{metricNames.map(x=>`<option ${{x===active?'selected':''}}>${{x}}</option>`).join('')}}</select></div><svg viewBox="0 0 ${{w}} ${{h}}">`;
for(let k=0;k<=5;k++){{const y=p+k*(h-2*p)/5,val=max*(1-k/5);svg+=`<line x1="${{p}}" x2="${{w-p}}" y1="${{y}}" y2="${{y}}" stroke="#e3e8ef"/><text x="5" y="${{y+4}}">${{f(val,share?1:1)}}</text>`}}
svg+=`<polyline fill="none" stroke="#c73e4d" stroke-width="3" points="${{d.map((x,i)=>`${{xs(i)}},${{ys(x.observed)}}`).join(' ')}}"/><polyline fill="none" stroke="#315efb" stroke-width="3" points="${{d.map((x,i)=>`${{xs(i)}},${{ys(x.model_mean)}}`).join(' ')}}"/>`;
d.forEach((x,i)=>{{const xx=xs(i);svg+=`<line x1="${{xx}}" x2="${{xx}}" y1="${{ys(x.model_mean_ci95_low)}}" y2="${{ys(x.model_mean_ci95_high)}}" stroke="#315efb"/><circle cx="${{xx}}" cy="${{ys(x.observed)}}" r="4" fill="#c73e4d"/><circle cx="${{xx}}" cy="${{ys(x.model_mean)}}" r="4" fill="#315efb"/><text x="${{xx}}" y="${{h-12}}" text-anchor="middle">${{x.ten_cohort.replace(/[()\]]/g,'')}}</text>`}});svg+=`<text x="${{w-250}}" y="20" fill="#c73e4d">● CGSS</text><text x="${{w-150}}" y="20" fill="#315efb">● 模型</text></svg>`;document.getElementById('metric').innerHTML=svg;document.getElementById('pick').onchange=e=>{{active=e.target.value;drawMetric()}}}}
drawMetric();document.getElementById('params').innerHTML=P.map(x=>`<tr><td>${{x.parameter}}</td><td>${{x.meaning}}</td><td>${{f(x.paper_transferred_value)}}</td><td>${{f(x.china_calibrated_value)}}</td></tr>`).join('');
const cov={{}};A.forEach(x=>{{cov[x.input]??={{observed:0,interpolated:0,extrapolated_boundary_hold:0}};cov[x.input][x.status]++}});document.getElementById('coverage').innerHTML=`<table><thead><tr><th>输入</th><th>观测</th><th>插值</th><th>边界保持</th></tr></thead><tbody>${{Object.entries(cov).map(([k,v])=>`<tr><td>${{k}}</td><td>${{v.observed}}</td><td>${{v.interpolated}}</td><td>${{v.extrapolated_boundary_hold}}</td></tr>`).join('')}}</tbody></table>`;document.getElementById('excluded').textContent=`收入教育组缺失而排除：${{X[0].rows_excluded}} 行，n=${{X[0].n_excluded}}；保留 n=${{X[0].n_retained}}。`;
document.getElementById('table3').innerHTML=T.map(x=>`<tr><td>${{SN[x.source]||x.source}}</td><td>${{TN[x.couple_type]||x.couple_type}}</td><td>${{f(x.mean)}}</td><td>${{f(x.sd)}}</td></tr>`).join('');
const scale=x=>50+x/20*50;document.getElementById('effects').innerHTML=E.map(x=>{{const v=x.mechanism_effect_percentage_points,left=Math.min(50,scale(v)),width=Math.abs(scale(v)-50);return `<div class="row"><div>${{SN[x.source]}} · ${{TN[x.couple_type]}}</div><div class="track"><span class="zero"></span><span class="bar" style="left:${{left}}%;width:${{width}}%;background:${{v>=0?'#315efb':'#c73e4d'}}"></span></div><div>${{v>=0?'+':''}}${{f(v,1)}} pp</div></div>`}}).join('');
</script></main></body></html>"""
    target = Path(output_html)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding="utf-8")
    return target


def build_china_report(
    *,
    targets_csv: str | Path,
    input_audit_csv: str | Path,
    exclusion_audit_csv: str | Path,
    parameters_csv: str | Path,
    validation_runs_csv: str | Path,
    table_long_csv: str | Path,
    effects_csv: str | Path,
    calibration_loss_csv: str | Path,
    transferred_runs_csv: str | Path,
    output_dir: str | Path,
) -> tuple[pd.DataFrame, Path]:
    targets = pd.read_csv(targets_csv)
    runs = pd.read_csv(validation_runs_csv)
    parameters_raw = pd.read_csv(parameters_csv)
    validation = validation_table(runs, targets)
    parameters = parameter_comparison(parameters_raw)
    candidate_condition = aggregate_conditions(runs.loc[runs["condition_no"].eq(1)])
    candidate_loss = calibration_loss(candidate_condition, targets)
    if len(candidate_loss) != 1:
        raise ValueError("candidate validation did not produce one condition-level loss")
    design_loss = pd.read_csv(calibration_loss_csv)
    transferred_runs = pd.read_csv(transferred_runs_csv)
    transferred_condition = aggregate_conditions(
        transferred_runs.loc[transferred_runs["condition_no"].eq(1)]
    )
    transferred_loss = calibration_loss(transferred_condition, targets)
    if len(transferred_loss) != 1:
        raise ValueError("transferred baseline did not produce one condition-level loss")
    fit_summary = {
        "behavior_candidate_calibration_fit": float(parameters_raw.iloc[0]["behavior_candidate_calibration_fit"]),
        "behavior_candidate_validation_fit": float(parameters_raw.iloc[0]["behavior_candidate_validation_fit"]),
        "rejected_rsm_predicted_fit": float(parameters_raw.iloc[0]["rejected_rsm_predicted_fit"]),
        "rejected_rsm_validation_fit": float(parameters_raw.iloc[0]["rejected_rsm_validation_fit"]),
        "primary_actual_fit": float(candidate_loss.iloc[0]["fit"]),
        "primary_actual_deviation": float(candidate_loss.iloc[0]["deviation"]),
        "best_design_fit": float(design_loss["fit"].max()),
        "best_design_condition": int(design_loss.loc[design_loss["fit"].idxmax(), "condition_no"]),
        "paper_parameter_transfer_fit": float(transferred_loss.iloc[0]["fit"]),
        "paper_parameter_transfer_deviation": float(transferred_loss.iloc[0]["deviation"]),
    }
    table_long = pd.read_csv(table_long_csv)
    effects = pd.read_csv(effects_csv)
    input_audit = pd.read_csv(input_audit_csv)
    exclusion = pd.read_csv(exclusion_audit_csv)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    validation.to_csv(target / "china_validation_by_cohort_metric.csv", index=False)
    parameters.to_csv(target / "china_parameter_comparison.csv", index=False)
    candidate_loss.to_csv(target / "china_primary_independent_loss.csv", index=False)
    transferred_loss.to_csv(target / "china_paper_parameter_transfer_loss.csv", index=False)
    dashboard = write_china_dashboard(
        validation=validation,
        parameters=parameters,
        table_long=table_long,
        effects=effects,
        input_audit=input_audit,
        exclusion_audit=exclusion,
        fit_summary=fit_summary,
        output_html=target / "china_abm_calibration_ablation_dashboard.html",
    )
    share_rows = validation[validation["unit"].eq("share")]
    age_rows = validation[validation["unit"].eq("years")]
    summary = {
        "cohorts": int(validation["ten_cohort"].nunique()),
        "validation_runs": 50,
        "delta_full": float(parameters_raw.iloc[0]["delta"]),
        **fit_summary,
        "share_mae_all_cells": float(share_rows["difference_model_minus_observed"].abs().mean()),
        "share_max_abs_difference": float(share_rows["difference_model_minus_observed"].abs().max()),
        "age_mae_years": float(age_rows["difference_model_minus_observed"].abs().mean()),
        "cells_observed_within_model_mean_plus_minus_one_run_sd": int(
            validation["observed_within_model_mean_plus_minus_one_run_sd"].sum()
        ),
        "cells_total": int(len(validation)),
        "incompatibility_cells_after_holm_5pct": int(
            validation["incompatibility_detected_holm_5pct"].sum()
        ),
        "dashboard": str(dashboard.resolve()),
        "interpretation": "primary model is a data-only China substitution with paper behavior parameters; CGSS is an external empirical check for that transfer specification; behavior recalibration attempts are retained as rejected diagnostics",
    }
    (target / "china_report_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return validation, dashboard
