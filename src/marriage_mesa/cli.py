from __future__ import annotations

import argparse

from .experiment import run_design
from .parity import compare_netlogo_mesa
from .workflow import WorkflowSpec, run_workflow


def _conditions(text: str | None):
    if not text:
        return None
    values: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def _integer_sequence(text: str | None):
    return None if not text else tuple(int(value) for value in text.split(","))


def main() -> None:
    parser = argparse.ArgumentParser(prog="marriage-mesa")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run-design", help="run an original experimental-design CSV")
    run.add_argument("--design", required=True)
    run.add_argument("--input-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--conditions", help="for example 1-3,8")
    run.add_argument("--repetitions", type=int, default=50)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--base-seed", type=int, default=20260817)
    run.add_argument("--census-dir")
    run.add_argument("--cohort-breaks", help="comma-separated edges, e.g. 1940,1950,1960")

    parity = commands.add_parser("parity", help="compare Mesa repeats with original NetLogo outputs")
    parity.add_argument("--netlogo-dir", required=True)
    parity.add_argument("--pattern", default="BE_c1_*.txt")
    parity.add_argument("--mesa-results", required=True)
    parity.add_argument("--output", required=True)
    parity.add_argument("--report")
    parity.add_argument("--standardized-tolerance", type=float, default=0.35)

    parity_all = commands.add_parser(
        "parity-all", help="compare every country in the original validation design"
    )
    parity_all.add_argument("--design", required=True)
    parity_all.add_argument("--netlogo-dir", required=True)
    parity_all.add_argument("--mesa-results", required=True)
    parity_all.add_argument("--output-dir", required=True)
    parity_all.add_argument("--standardized-tolerance", type=float, default=0.35)

    workflow = commands.add_parser(
        "run-workflow", help="run, validate, and write a provenance manifest"
    )
    workflow.add_argument("--design", required=True)
    workflow.add_argument("--input-dir", required=True)
    workflow.add_argument("--output", required=True)
    workflow.add_argument("--conditions", help="for example 1-3,8")
    workflow.add_argument("--repetitions", type=int, default=50)
    workflow.add_argument("--workers", type=int, default=1)
    workflow.add_argument("--base-seed", type=int, default=20260817)
    workflow.add_argument("--census-dir")
    workflow.add_argument("--manifest")
    workflow.add_argument("--source-model")
    workflow.add_argument("--cohort-breaks", help="comma-separated cohort edges")

    china_prepare = commands.add_parser(
        "prepare-china", help="prepare China inputs, CGSS targets, and experiment designs"
    )
    china_prepare.add_argument("--education", required=True)
    china_prepare.add_argument("--income", required=True)
    china_prepare.add_argument("--cgss", required=True)
    china_prepare.add_argument("--output-dir", required=True)

    china_design = commands.add_parser(
        "china-final-design", help="build full/ablation design from one calibrated parameter row"
    )
    china_design.add_argument("--parameters", required=True)
    china_design.add_argument("--output", required=True)

    china_report = commands.add_parser(
        "china-report", help="build the China calibration/validation/ablation dashboard"
    )
    china_report.add_argument("--targets", required=True)
    china_report.add_argument("--input-audit", required=True)
    china_report.add_argument("--exclusion-audit", required=True)
    china_report.add_argument("--parameters", required=True)
    china_report.add_argument("--validation-runs", required=True)
    china_report.add_argument("--table-long", required=True)
    china_report.add_argument("--effects", required=True)
    china_report.add_argument("--calibration-loss", required=True)
    china_report.add_argument("--transferred-runs", required=True)
    china_report.add_argument("--output-dir", required=True)

    visualize = commands.add_parser("visualize", help="launch the optional Mesa Solara dashboard")
    visualize.add_argument("--design", required=True)
    visualize.add_argument("--input-dir", required=True)
    visualize.add_argument("--condition", required=True, type=int)
    visualize.add_argument("--seed", type=int, default=20260817)

    ablation = commands.add_parser(
        "ablation-report", help="build the paper-style Table 3 ablation report"
    )
    ablation.add_argument("--table-long", required=True)
    ablation.add_argument("--country-cohort", required=True)
    ablation.add_argument("--output-dir", required=True)
    ablation.add_argument("--netlogo-table-long")

    extract_ablation = commands.add_parser(
        "extract-netlogo-ablation",
        help="convert archived NetLogo validation/effect runs to run-level CSVs",
    )
    extract_ablation.add_argument("--validation-design", required=True)
    extract_ablation.add_argument("--effect-design", required=True)
    extract_ablation.add_argument("--validation-output-dir", required=True)
    extract_ablation.add_argument("--effect-output-dir", required=True)
    extract_ablation.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "run-design":
        run_design(
            args.design,
            args.input_dir,
            args.output,
            conditions=_conditions(args.conditions),
            repetitions=args.repetitions,
            workers=args.workers,
            base_seed=args.base_seed,
            census_dir=args.census_dir,
            cohort_breaks=_integer_sequence(args.cohort_breaks),
        )
    elif args.command == "parity":
        result = compare_netlogo_mesa(
            args.netlogo_dir,
            args.pattern,
            args.mesa_results,
            args.output,
            args.report,
            args.standardized_tolerance,
        )
        print(result.to_string(index=False))
    elif args.command == "parity-all":
        from .reporting import run_all_country_parity
        summary, detail, dashboard = run_all_country_parity(
            design_path=args.design,
            netlogo_dir=args.netlogo_dir,
            mesa_results=args.mesa_results,
            output_dir=args.output_dir,
            standardized_tolerance=args.standardized_tolerance,
        )
        print(summary.to_string(index=False))
        print(f"detail checks: {len(detail)}; dashboard: {dashboard}")
    elif args.command == "run-workflow":
        selected_conditions = _conditions(args.conditions)
        _, manifest = run_workflow(WorkflowSpec(
            design_path=args.design,
            input_dir=args.input_dir,
            output_csv=args.output,
            conditions=None if selected_conditions is None else tuple(selected_conditions),
            repetitions=args.repetitions,
            workers=args.workers,
            base_seed=args.base_seed,
            census_dir=args.census_dir,
            manifest_path=args.manifest,
            source_model_path=args.source_model,
            cohort_breaks=_integer_sequence(args.cohort_breaks),
        ))
        print(f"workflow manifest: {manifest}")
    elif args.command == "prepare-china":
        from .china import prepare_china_assets
        paths = prepare_china_assets(
            education_xlsx=args.education,
            income_xlsx=args.income,
            cgss_dta=args.cgss,
            output_dir=args.output_dir,
        )
        for name, path in paths.items():
            print(f"{name}: {path}")
    elif args.command == "china-final-design":
        from .china import ablation_design, config_from_parameter_file
        design = ablation_design(config_from_parameter_file(args.parameters))
        design.to_csv(args.output, index=False)
        print(f"China full/ablation design: {args.output}")
    elif args.command == "china-report":
        from .china_reporting import build_china_report
        validation, dashboard = build_china_report(
            targets_csv=args.targets,
            input_audit_csv=args.input_audit,
            exclusion_audit_csv=args.exclusion_audit,
            parameters_csv=args.parameters,
            validation_runs_csv=args.validation_runs,
            table_long_csv=args.table_long,
            effects_csv=args.effects,
            calibration_loss_csv=args.calibration_loss,
            transferred_runs_csv=args.transferred_runs,
            output_dir=args.output_dir,
        )
        print(f"validation cells: {len(validation)}; dashboard: {dashboard}")
    elif args.command == "visualize":
        from .visualization import launch_dashboard
        launch_dashboard(
            design_path=args.design,
            input_dir=args.input_dir,
            condition_no=args.condition,
            seed=args.seed,
        )
    elif args.command == "ablation-report":
        from .ablation import build_ablation_report
        effects, comparison, dashboard = build_ablation_report(
            table_long_csv=args.table_long,
            country_cohort_csv=args.country_cohort,
            output_dir=args.output_dir,
            netlogo_table_long_csv=args.netlogo_table_long,
        )
        print(effects.to_string(index=False))
        print(
            f"paper-rounded matches: "
            f"{int(comparison['same_after_two_decimal_rounding'].sum())}/{len(comparison)}"
        )
        print(f"dashboard: {dashboard}")
    else:
        from .ablation import extract_netlogo_ablation_runs
        paths = extract_netlogo_ablation_runs(
            validation_design=args.validation_design,
            effect_design=args.effect_design,
            validation_output_dir=args.validation_output_dir,
            effect_output_dir=args.effect_output_dir,
            output_dir=args.output_dir,
        )
        for source, path in paths.items():
            print(f"{source}: {path}")


if __name__ == "__main__":
    main()
