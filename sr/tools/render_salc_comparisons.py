"""Render selected SalC comparison figures without recomputing a full audit."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_salc import (
    load_config,
    refuse_nonempty_output,
    render_pair_figure,
    validate_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_render_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Render config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Render config must contain a YAML mapping")
    required = ("audit_config", "output_dir", "pair_ids", "methods")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing render config fields: {missing}")
    return config


def resolve_salc_parameters(render: dict, audit: dict) -> tuple[float, float]:
    """Resolve optional visualization parameters without changing old configs."""
    smoothing_radius = float(
        render.get("smoothing_radius", audit.get("smoothing_radius", 25.0))
    )
    salient_percent = float(
        render.get("salient_percent", audit.get("salient_percent", 10.0))
    )
    if smoothing_radius < 0:
        raise ValueError("smoothing_radius must be non-negative")
    if not 0 < salient_percent <= 100:
        raise ValueError("salient_percent must be in (0, 100]")
    return smoothing_radius, salient_percent


def run(render_config_path: Path) -> Path:
    render = load_render_config(render_config_path)
    audit_config_path = Path(str(render["audit_config"])).expanduser().resolve()
    audit = load_config(audit_config_path)
    output_dir = Path(str(render["output_dir"])).expanduser().resolve()
    refuse_nonempty_output(output_dir)
    shutil.copy2(render_config_path, output_dir / "render.yaml")
    shutil.copy2(audit_config_path, output_dir / "audit_source.yaml")

    pair_ids = [str(value) for value in render["pair_ids"]]
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_ids must be non-empty and unique")
    methods = [str(value) for value in render["methods"]]
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")

    reference_source = validate_source("reference", audit["reference"])
    method_sources = {
        str(name): validate_source(str(name), source)
        for name, source in audit["methods"].items()
    }
    if "Source" not in method_sources:
        raise ValueError("Audit config must include Source")
    unknown = [name for name in methods if name not in method_sources]
    if unknown:
        raise ValueError(f"Unknown render methods: {unknown}")

    panel_size = int(render.get("panel_size", 420))
    columns = int(render.get("columns", 5))
    overlay_opacity = float(render.get("overlay_opacity", 0.55))
    overlay_gamma = float(render.get("overlay_gamma", 0.65))
    smoothing_radius, salient_percent = resolve_salc_parameters(render, audit)
    crop_to_reference_fov = bool(render.get("crop_to_reference_fov", False))
    crop_margin_fraction = float(render.get("crop_margin_fraction", 0.02))
    font_size = int(render.get("font_size", 10))
    filename_pattern = str(
        render.get("filename_pattern", "{pair}_salc_comparison_2x5.png")
    )
    if "{pair}" not in filename_pattern:
        raise ValueError("filename_pattern must contain {pair}")
    for pair_id in tqdm(pair_ids, desc="Render SalC comparisons", unit="pair"):
        render_pair_figure(
            pair_id,
            reference_source,
            method_sources,
            methods,
            output_dir / filename_pattern.format(pair=pair_id),
            channel=str(audit.get("channel", "green")),
            smoothing_radius=smoothing_radius,
            salient_percent=salient_percent,
            legacy_zero_interpolation=bool(
                audit.get("legacy_zero_interpolation", True)
            ),
            panel_size=panel_size,
            columns=columns,
            overlay_opacity=overlay_opacity,
            font_size=font_size,
            overlay_gamma=overlay_gamma,
            crop_to_reference_fov=crop_to_reference_fov,
            crop_margin_fraction=crop_margin_fraction,
        )
    print(f"Wrote selected SalC comparisons: {output_dir}")
    return output_dir


def self_test() -> None:
    try:
        load_render_config(Path("missing.yaml"))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing render config was not rejected")
    assert "{pair}_salc_comparison_2x5.png".format(pair="0045") == (
        "0045_salc_comparison_2x5.png"
    )
    assert resolve_salc_parameters(
        {}, {"smoothing_radius": 1, "salient_percent": 4}
    ) == (1.0, 4.0)
    assert resolve_salc_parameters(
        {"smoothing_radius": 3, "salient_percent": 4},
        {"smoothing_radius": 1, "salient_percent": 4},
    ) == (3.0, 4.0)
    print("SalC comparison renderer self-test passed")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.render_config is not None:
        run(arguments.render_config)
    else:
        raise SystemExit("Provide --render-config or --self-test")
