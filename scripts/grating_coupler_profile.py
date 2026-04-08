"""Profiling harness for the finite grating coupler script.

This runs the same finite-supercell model as ``grating_coupler.py`` and prints
the dominant function calls from ``cProfile`` so larger machines can be used to
benchmark realistic dense RCWA settings.
"""

from __future__ import annotations

import cProfile
import io
import pathlib
import pstats
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts import grating_coupler as gc


def _profile_call(label: str, fn, top_n: int, sort_key: str) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    result = fn()
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(sort_key)
    stats.print_stats(top_n)

    print(f"=== {label} ===")
    print(stream.getvalue())

    if isinstance(result, dict):
        for pol, (reflected, transmitted, residual) in result.items():
            peak_index = int(np.argmax(residual))
            print(
                f"{pol}: peak non-zero-order power = {float(residual[peak_index]):.6f} at "
                f"{peak_index}"
            )


def run_design_profile(N: int, num_points: int, top_n: int, sort_key: str) -> None:
    _profile_call(
        label=f"design point N={N} num_points={num_points}",
        fn=lambda: gc.run_design_point_demo(N=N, num_points=num_points),
        top_n=top_n,
        sort_key=sort_key,
    )


def run_sweep_profile(
    N: int,
    num_points: int,
    center_wavelength_nm: float,
    half_span_nm: float,
    num_samples: int,
    top_n: int,
    sort_key: str,
) -> None:
    wavelengths_nm = np.linspace(
        center_wavelength_nm - half_span_nm,
        center_wavelength_nm + half_span_nm,
        num_samples,
    )
    _profile_call(
        label=(
            f"wavelength sweep N={N} num_points={num_points} "
            f"samples={num_samples} center={center_wavelength_nm:.1f} nm"
        ),
        fn=lambda: gc.sweep_wavelength_response_all_pols(
            wavelengths_nm,
            N=N,
            num_points=num_points,
        ),
        top_n=top_n,
        sort_key=sort_key,
    )


def main() -> None:
    mode = "both"
    design_N = 32
    sweep_N = 24
    num_points = 1024
    center_wavelength_nm = gc.design_wl_nm
    half_span_nm = 8.0
    num_samples = 5
    top_n = 30
    sort_key = "cumtime"

    if mode in {"design", "both"}:
        run_design_profile(
            N=design_N,
            num_points=num_points,
            top_n=top_n,
            sort_key=sort_key,
        )

    if mode in {"sweep", "both"}:
        run_sweep_profile(
            N=sweep_N,
            num_points=num_points,
            center_wavelength_nm=center_wavelength_nm,
            half_span_nm=half_span_nm,
            num_samples=num_samples,
            top_n=top_n,
            sort_key=sort_key,
        )


if __name__ == "__main__":
    main()
