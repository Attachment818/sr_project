"""Audit SalC table rows when methods have different registration success counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from salc_reference import as_unit_gray, extract_salient_intensity, score_salience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Audit config not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Audit config must contain a YAML mapping")
    required = ("dataset_name", "output_dir", "pair_ids", "reference", "source", "methods")
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    return config


def refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_image(source: Mapping[str, object], pair_id: str) -> Path:
    if "directory" not in source or "pattern" not in source:
        raise ValueError("Every image source requires directory and pattern")
    path = Path(str(source["directory"])) / str(source["pattern"]).format(pair=pair_id)
    return path.expanduser().resolve()


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read image: {path}")
    return image


def read_seconds(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Time file not found: {path}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"No numeric time found in: {path}")
    seconds = float(match.group(0))
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"Invalid registration time in {path}: {seconds}")
    return seconds


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    methods: Mapping[str, Mapping[str, object]],
    rows: Sequence[Mapping[str, object]],
    pair_count: int,
) -> list[dict]:
    summary = []
    for method_name, method in methods.items():
        method_rows = [row for row in rows if row["method"] == method_name]
        aligned = [row for row in method_rows if row["status"] == "aligned"]
        missing = [str(row["pair_id"]) for row in method_rows if row["status"] != "aligned"]
        aligned_scores = [float(row["salc"]) for row in aligned]
        all_scores = [float(row["salc"]) for row in method_rows]
        total_seconds = read_seconds(Path(str(method["time_file"])).expanduser().resolve())
        attempted_count = int(method.get("attempted_count", pair_count))
        if attempted_count <= 0:
            raise ValueError(f"{method_name} attempted_count must be positive")
        summary.append(
            {
                "method": method_name,
                "aligned_count": len(aligned),
                "missing_count": len(missing),
                "missing_pair_ids": ",".join(missing),
                "salc_aligned_only": float(np.mean(aligned_scores)) if aligned_scores else float("nan"),
                "salc_source_fallback": float(np.mean(all_scores)),
                "total_time_seconds": total_seconds,
                "time_seconds_per_attempt": total_seconds / attempted_count,
                "attempted_count": attempted_count,
            }
        )
    return summary


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    refuse_nonempty_output(output_dir)
    shutil.copy2(config_path, output_dir / "audit.yaml")

    pair_ids = [str(pair_id) for pair_id in config["pair_ids"]]
    if not pair_ids or len(set(pair_ids)) != len(pair_ids):
        raise ValueError("pair_ids must be non-empty and unique")
    methods = config["methods"]
    if not isinstance(methods, Mapping) or not methods:
        raise ValueError("methods must be a non-empty mapping")
    channel = str(config.get("channel", "green"))
    smoothing_radius = float(config.get("smoothing_radius", 25.0))
    salient_percent = float(config.get("salient_percent", 10.0))
    legacy = bool(config.get("legacy_zero_interpolation", True))

    rows: list[dict] = []
    for pair_id in tqdm(pair_ids, desc=f"Audit {config['dataset_name']}", unit="pair"):
        reference_path = resolve_image(config["reference"], pair_id)
        source_path = resolve_image(config["source"], pair_id)
        reference = as_unit_gray(read_image(reference_path), channel)
        source = as_unit_gray(read_image(source_path), channel)
        if reference.shape != source.shape:
            raise ValueError(
                f"Reference/source shape mismatch for pair {pair_id}: "
                f"{reference.shape} vs {source.shape}"
            )
        reference_salience = extract_salient_intensity(
            reference,
            smoothing_radius=smoothing_radius,
            salient_percent=salient_percent,
            legacy_zero_interpolation=False,
        )
        source_salience = extract_salient_intensity(
            source,
            smoothing_radius=smoothing_radius,
            salient_percent=salient_percent,
            legacy_zero_interpolation=legacy,
        )
        source_score = score_salience(reference_salience, source_salience)

        for method_name, method in methods.items():
            aligned_path = resolve_image(method, pair_id)
            if aligned_path.is_file():
                aligned = as_unit_gray(read_image(aligned_path), channel)
                if aligned.shape != reference.shape:
                    raise ValueError(
                        f"Shape mismatch for {method_name} pair {pair_id}: "
                        f"{aligned.shape} vs {reference.shape}"
                    )
                aligned_salience = extract_salient_intensity(
                    aligned,
                    smoothing_radius=smoothing_radius,
                    salient_percent=salient_percent,
                    legacy_zero_interpolation=legacy,
                )
                score = score_salience(reference_salience, aligned_salience)
                status = "aligned"
                scored_path = aligned_path
            else:
                score = source_score
                status = "source_fallback"
                scored_path = source_path
            rows.append(
                {
                    "pair_id": pair_id,
                    "method": method_name,
                    "status": status,
                    "salc": score,
                    "reference_path": str(reference_path),
                    "scored_path": str(scored_path),
                }
            )

    summary = summarize(methods, rows, len(pair_ids))
    write_csv(
        output_dir / "salc_scores.csv",
        rows,
        ("pair_id", "method", "status", "salc", "reference_path", "scored_path"),
    )
    write_csv(
        output_dir / "table_extension_summary.csv",
        summary,
        (
            "method",
            "aligned_count",
            "missing_count",
            "missing_pair_ids",
            "salc_aligned_only",
            "salc_source_fallback",
            "total_time_seconds",
            "time_seconds_per_attempt",
            "attempted_count",
        ),
    )
    payload = {
        "dataset_name": config["dataset_name"],
        "pair_count": len(pair_ids),
        "missing_policy": "Use the unregistered source image when an aligned output is absent.",
        "parameters": {
            "channel": channel,
            "smoothing_radius": smoothing_radius,
            "salient_percent": salient_percent,
            "legacy_zero_interpolation": legacy,
        },
        "methods": summary,
    }
    (output_dir / "table_extension_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote SalC table extension audit: {output_dir}")
    return output_dir


def self_test() -> None:
    rows = [
        {"pair_id": "1", "method": "A", "status": "aligned", "salc": 0.8},
        {"pair_id": "2", "method": "A", "status": "source_fallback", "salc": 0.4},
    ]
    assert np.isclose(np.mean([float(row["salc"]) for row in rows]), 0.6)
    try:
        resolve_image({}, "1")
    except ValueError:
        pass
    else:
        raise AssertionError("Missing image-source fields were not rejected")
    print("SalC table-extension self-test passed")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        self_test()
    elif arguments.config is not None:
        run(arguments.config)
    else:
        raise SystemExit("Provide --config or --self-test")
