"""Optional Mesa Solara visualization; no spatial state enters the science."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .config import ModelConfig
from .experiment import read_design
from .inputs import CohortInputs
from .model import MarriageModel


def model_from_design(
    *, design_path: str | Path, input_dir: str | Path, condition_no: int, seed: int
) -> MarriageModel:
    design = read_design(design_path)
    selected = design.loc[design["condition_no"].eq(condition_no)]
    if len(selected) != 1:
        raise ValueError(f"condition {condition_no} must identify exactly one design row")
    config = ModelConfig.from_mapping(
        selected.iloc[0].to_dict(),
        run_no=1,
        seed_to_use="own",
        own_seed=seed,
    )
    inputs = CohortInputs.from_netlogo(input_dir, config.country)
    return MarriageModel(config=config, inputs=inputs, collect_census=True)


def build_dashboard(
    *, design_path: str | Path, input_dir: str | Path, condition_no: int, seed: int = 20260817
):
    # Matplotlib's default Windows cache can be unwritable in managed
    # environments.  A visualization-only cache must not touch model state.
    cache_dir = Path(tempfile.gettempdir()) / "marriage_mesa_matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        from mesa.visualization import SolaraViz, make_plot_component
    except ImportError as error:
        raise RuntimeError(
            "Visualization dependencies are missing; install with pip install -e '.[visualization]'"
        ) from error

    model = model_from_design(
        design_path=design_path,
        input_dir=input_dir,
        condition_no=condition_no,
        seed=seed,
    )
    components = [
        make_plot_component({"single": "#6b7280", "dating": "#2563eb", "married": "#dc2626"}),
        make_plot_component({"population": "#111827"}),
    ]
    params = {
        "config": model.config,
        "inputs": model.inputs,
        "collect_census": True,
    }
    return SolaraViz(
        model,
        components=components,
        model_params=params,
        name=f"Grow–Van Bavel marriage market: {model.config.country}, condition {condition_no}",
        play_interval=75,
        render_interval=1,
    )


def dashboard_from_environment():
    required = ["MARRIAGE_MESA_DESIGN", "MARRIAGE_MESA_INPUT_DIR", "MARRIAGE_MESA_CONDITION"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"missing visualization environment variables: {', '.join(missing)}")
    return build_dashboard(
        design_path=os.environ["MARRIAGE_MESA_DESIGN"],
        input_dir=os.environ["MARRIAGE_MESA_INPUT_DIR"],
        condition_no=int(os.environ["MARRIAGE_MESA_CONDITION"]),
        seed=int(os.environ.get("MARRIAGE_MESA_SEED", "20260817")),
    )


def launch_dashboard(
    *, design_path: str | Path, input_dir: str | Path, condition_no: int, seed: int
) -> None:
    environment = os.environ.copy()
    environment.update({
        "MARRIAGE_MESA_DESIGN": str(Path(design_path).resolve()),
        "MARRIAGE_MESA_INPUT_DIR": str(Path(input_dir).resolve()),
        "MARRIAGE_MESA_CONDITION": str(condition_no),
        "MARRIAGE_MESA_SEED": str(seed),
    })
    app = Path(__file__).with_name("solara_app.py")
    subprocess.run([sys.executable, "-m", "solara", "run", str(app)],
                   check=True, env=environment)
