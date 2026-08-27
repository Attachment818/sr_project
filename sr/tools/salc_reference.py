"""Reference-compatible Salience Correlation (SalC) utilities.

This module ports the MATLAB implementation supplied for the supplementary
experiments.  It is intentionally limited to already registered image pairs,
which is the ``u = 0`` mode used by the supplied HRF driver.  The original
MATLAB files remain untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class SalcResult:
    score: float
    reference_salience: np.ndarray
    registered_salience: np.ndarray


def as_unit_gray(image: np.ndarray, channel: str) -> np.ndarray:
    """Convert an OpenCV image to a float64 single-channel image in [0, 1]."""
    if image is None:
        raise ValueError("Input image is None")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] >= 3:
        if channel == "green":
            gray = image[:, :, 1]
        elif channel == "gray":
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported channel mode: {channel}")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    gray = gray.astype(np.float64, copy=False)
    if np.issubdtype(image.dtype, np.integer):
        gray /= float(np.iinfo(image.dtype).max)
    elif gray.size and (gray.min() < 0.0 or gray.max() > 1.0):
        raise ValueError("Floating-point images must already be in [0, 1]")
    return gray


def _legacy_zero_shifted_linear(image: np.ndarray) -> np.ndarray:
    """Port ``ShiftedLinear_Interp_2D(0, 0, image)`` from the MATLAB code."""
    if image.ndim != 2:
        raise ValueError("Shifted-linear interpolation expects a 2-D image")
    tau = 0.5 * (1.0 - np.sqrt(3.0) / 3.0)
    z0 = tau / (1.0 - tau)

    padded = np.pad(image, ((1, 1), (1, 1)), mode="symmetric")

    def causal_prefilter(values: np.ndarray, axis: int) -> np.ndarray:
        first = np.take(values, [0], axis=axis)
        centered = values - first
        filtered = lfilter([1.0], [1.0, z0], centered, axis=axis)
        corrected = filtered + first / (1.0 + z0)
        return corrected / (1.0 - tau)

    coefficients = causal_prefilter(padded, axis=0)
    coefficients = causal_prefilter(coefficients, axis=1)

    rows, cols = image.shape
    row_query = np.arange(1, rows + 1, dtype=np.float64) - tau
    col_query = np.arange(1, cols + 1, dtype=np.float64) - tau
    row0 = np.floor(row_query).astype(np.intp)
    col0 = np.floor(col_query).astype(np.intp)
    row_weight = (row_query - row0)[:, None]
    col_weight = (col_query - col0)[None, :]

    top = (
        coefficients[row0[:, None], col0[None, :]] * (1.0 - col_weight)
        + coefficients[row0[:, None], col0[None, :] + 1] * col_weight
    )
    bottom = (
        coefficients[row0[:, None] + 1, col0[None, :]] * (1.0 - col_weight)
        + coefficients[row0[:, None] + 1, col0[None, :] + 1] * col_weight
    )
    return top * (1.0 - row_weight) + bottom * row_weight


def _legacy_wirtinger(
    image: np.ndarray,
    smoothing_radius: float,
    *,
    mirror_width: int = 19,
) -> np.ndarray:
    """Port ``abs(wirtinger(image, radius))`` for exponent 1."""
    if min(image.shape) <= mirror_width:
        raise ValueError(
            f"Image shape {image.shape} is too small for mirror width {mirror_width}"
        )
    extended = np.pad(
        image,
        ((mirror_width, mirror_width), (mirror_width, mirror_width)),
        mode="reflect",
    )
    rows, cols = extended.shape
    row_frequency = np.fft.fftfreq(rows) * (2.0 * np.pi)
    col_frequency = np.fft.fftfreq(cols) * (2.0 * np.pi)
    wx = row_frequency[:, None]
    wy = col_frequency[None, :]
    scale = (float(smoothing_radius) + 2.0) / 4.0
    response = (1j * (wx + 1j * wy)) * np.exp(
        -0.5 * scale * scale * (wx * wx + wy * wy)
    )

    # Preserve the historical non-square Nyquist masking behavior in
    # imagefilter.m: its final N value is the second image dimension.
    if cols % 2 == 0:
        response[:, cols // 2] = 0.0
        row_bins = np.arange(rows)
        row_bins = np.where(row_bins < rows / 2.0, row_bins, row_bins - rows)
        legacy_row = np.flatnonzero(row_bins == -(cols / 2.0))
        if legacy_row.size:
            response[legacy_row, :] = 0.0

    spectrum = np.fft.fft2(extended)
    spectrum *= response
    filtered = np.fft.ifft2(spectrum).real
    if mirror_width:
        filtered = filtered[
            mirror_width:-mirror_width,
            mirror_width:-mirror_width,
        ]
    return np.abs(filtered)


def _largest_fraction_mask(values: np.ndarray, percent: float) -> np.ndarray:
    if not 0.0 < percent <= 100.0:
        raise ValueError("salient_percent must be in (0, 100]")
    flat = np.abs(values).reshape(-1)
    total = flat.size
    discard = int(np.floor(total * (1.0 - percent / 100.0) + 0.5))
    if discard <= 0:
        return np.ones(values.shape, dtype=bool)
    if discard >= total:
        return np.zeros(values.shape, dtype=bool)
    selected = np.argpartition(flat, discard)[discard:]
    mask = np.zeros(total, dtype=bool)
    mask[selected] = True
    return mask.reshape(values.shape)


def extract_salient_intensity(
    image: np.ndarray,
    *,
    smoothing_radius: float = 25.0,
    salient_percent: float = 10.0,
    legacy_zero_interpolation: bool = True,
) -> np.ndarray:
    """Return the sparse intensity image used by the supplied SalC code."""
    working = np.asarray(image, dtype=np.float64)
    if working.ndim != 2:
        raise ValueError("SalC expects a two-dimensional intensity image")
    if legacy_zero_interpolation:
        working = _legacy_zero_shifted_linear(working)
    gradient = _legacy_wirtinger(working, smoothing_radius)
    mask = _largest_fraction_mask(gradient, salient_percent)
    return working * mask


def score_salience(reference: np.ndarray, registered: np.ndarray) -> float:
    if reference.shape != registered.shape:
        raise ValueError(
            f"Salience shapes differ: {reference.shape} vs {registered.shape}"
        )
    a = reference.reshape(-1)
    b = registered.reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return float(abs(np.vdot(a, b)) / denominator)


def compute_salc(
    reference: np.ndarray,
    registered: np.ndarray,
    *,
    smoothing_radius: float = 25.0,
    salient_percent: float = 10.0,
    legacy_zero_interpolation: bool = True,
) -> SalcResult:
    if reference.shape != registered.shape:
        raise ValueError(
            f"Input shapes differ: {reference.shape} vs {registered.shape}"
        )
    reference_salience = extract_salient_intensity(
        reference,
        smoothing_radius=smoothing_radius,
        salient_percent=salient_percent,
        legacy_zero_interpolation=False,
    )
    registered_salience = extract_salient_intensity(
        registered,
        smoothing_radius=smoothing_radius,
        salient_percent=salient_percent,
        legacy_zero_interpolation=legacy_zero_interpolation,
    )
    return SalcResult(
        score=score_salience(reference_salience, registered_salience),
        reference_salience=reference_salience,
        registered_salience=registered_salience,
    )


def salience_overlay(
    reference_salience: np.ndarray,
    registered_salience: np.ndarray,
    *,
    gamma: float = 0.65,
    opacity: float = 0.55,
) -> np.ndarray:
    """Render target-blue/source-orange salience with neutral overlap."""
    if reference_salience.shape != registered_salience.shape:
        raise ValueError("Salience images must have identical shapes")
    if not 0.0 < opacity <= 1.0:
        raise ValueError("opacity must be in (0, 1]")

    def strength(values: np.ndarray) -> np.ndarray:
        nonzero = values[values > 0]
        if nonzero.size == 0:
            return np.zeros(values.shape, dtype=np.float64)
        scale = float(np.percentile(nonzero, 99.0))
        if scale <= 0.0:
            return np.zeros(values.shape, dtype=np.float64)
        return np.power(np.clip(values / scale, 0.0, 1.0), gamma)

    target = strength(reference_salience) * opacity
    source = strength(registered_salience) * opacity
    overlap = np.minimum(target, source)
    target_only = target - overlap
    source_only = source - overlap
    background = 1.0 - np.maximum(target, source)

    blue = np.array([0.25, 0.55, 0.95], dtype=np.float64)
    orange = np.array([0.95, 0.55, 0.18], dtype=np.float64)
    neutral = np.array([0.42, 0.42, 0.42], dtype=np.float64)
    rgb = (
        background[:, :, None]
        + target_only[:, :, None] * blue
        + source_only[:, :, None] * orange
        + overlap[:, :, None] * neutral
    )
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def self_test() -> None:
    grid_y, grid_x = np.mgrid[:96, :112]
    image = (
        0.25
        + 0.35 * np.sin(grid_x / 7.0)
        + 0.25 * np.cos(grid_y / 9.0)
    )
    image = np.clip(image, 0.0, 1.0)
    shifted = np.roll(image, shift=7, axis=1)
    identity = compute_salc(image, image)
    mismatch = compute_salc(image, shifted)
    assert identity.score > 0.99
    assert np.isfinite(mismatch.score)
    assert mismatch.score < identity.score
    overlay = salience_overlay(
        identity.reference_salience,
        mismatch.registered_salience,
    )
    assert overlay.shape == (*image.shape, 3)
    assert overlay.dtype == np.uint8
    print("SalC reference self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.self_test:
        raise SystemExit("Use --self-test; dataset execution is handled by audit_salc.py")
    self_test()
