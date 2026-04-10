"""NbOCl2 twist-angle pole search using RCWA with zero harmonics.

This mirrors the workflow in the user's working 4x4 script:

1. Build a complex-kx map of the selected reflection channel.
2. Detect local maxima of |r|.
3. Refine those candidates by solving 1 / r = 0 in the complex plane.
4. Sweep twist angle and plot Re(kx), with Im(kx) on the color axis.
5. Overlay Re(kx) at 405 nm with 2 x Re(kx) at 810 nm.

The solve still goes through RCWAFFF, but with N = 0 the structure is a
homogeneous single-layer stack.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
import pathlib
import sys
import time

import numpy as np
from scipy.ndimage import maximum_filter, label
from scipy.optimize import root

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(pathlib.Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcwa import Layer, Solver, Stack


# --- Parameters ---
THICKNESS_M = 122e-9
THICKNESS_NM = THICKNESS_M * 1e9
EPS_SUBSTRATE = 1.4696**2
EPS_SUPERSTRATE = 1.0
X_DOMAIN_NM = (0.0, 500.0)

EPS_DIAG_810 = np.diag([4.0, 6.14, 1.6**2]).astype(np.complex128)
EPS_DIAG_405 = np.diag([5.406, 10.465 + 1.923j, 1.6**2]).astype(np.complex128)

MODE_IDX = 1  # 0=TM -> r_pp, 1=TE -> r_ss
RCWA_ORDER = 0
NUM_POINTS_RCWA = 16

KX_RE_RANGE_M_INV = (0.5e7, 5.0e7)
KX_IM_RANGE_M_INV = (10.0e6, 1.0e6)

SINGLE_ANGLE_DEG = 0.0
FRAME_GRID_POINTS = 151
ANGLE_DEG_SAMPLES = np.linspace(0.0, 90.0, 45)

POLE_GRID_RE_POINTS = 301
POLE_GRID_IM_POINTS = 101
PEAK_NEIGHBORHOOD = 5
PEAK_THRESHOLD = 2.5
MAX_REFINE = 10
POLE_MAGNITUDE_THRESHOLD = 1e5
IM_KX_PLOT_MIN_UM_INV = -4.0
ROOT_TOL = 1e-10


@dataclass(frozen=True)
class SweepSpec:
    wavelength_nm: float
    eps_diag: np.ndarray
    label: str


@dataclass(frozen=True)
class PolePoint:
    angle_deg: float
    wavelength_nm: float
    kx_m_inv: complex
    residual: float
    magnitude: float


def rotate_eps_xy(eps_diag: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate a diagonal dielectric tensor in the xy plane."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return rotation @ eps_diag @ rotation.T


def _forward_q(eps: complex, kappa_normalized: complex) -> complex:
    q = np.sqrt(complex(eps) - complex(kappa_normalized) ** 2)
    if np.imag(q) < 0.0 or (abs(np.imag(q)) < 1e-14 and np.real(q) < 0.0):
        q = -q
    return q


def _physical_port_fields(eps: complex, kappa_normalized: complex) -> np.ndarray:
    q = _forward_q(eps, kappa_normalized)
    return np.stack(
        [
            np.array([0.0, -q, 1.0, 0.0], dtype=np.complex128),
            np.array([eps / q, 0.0, 0.0, -1.0], dtype=np.complex128),
            np.array([0.0, q, 1.0, 0.0], dtype=np.complex128),
            np.array([eps / q, 0.0, 0.0, 1.0], dtype=np.complex128),
        ],
        axis=1,
    )


def _build_stack_from_rotated_eps(
    wavelength_nm: float,
    eps_tensor: np.ndarray,
    kx_m_inv: complex,
) -> Stack:
    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=complex(kx_m_inv) * 1e-9,
        eps_substrate=EPS_SUBSTRATE,
        eps_superstrate=EPS_SUPERSTRATE,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=THICKNESS_NM,
            eps_tensor=eps_tensor,
            x_domain_nm=X_DOMAIN_NM,
        )
    )
    return stack


def _reflection_matrix_in_physical_ps_basis(stack: Stack) -> np.ndarray:
    solve_data = Solver.build_stack_solve_data(
        stack,
        RCWA_ORDER,
        num_points=NUM_POINTS_RCWA,
        verbose=False,
    )
    scattering_physical = Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(
                _physical_port_fields(stack.eps_substrate, stack.kappa_normalized),
                solve_data.substrate_continuity_fields,
            ),
            solve_data.total_scattering,
        ]
    )
    return np.asarray(scattering_physical[0], dtype=np.complex128)


def _mode_reflection_amplitude_rotated(
    wavelength_nm: float,
    eps_tensor: np.ndarray,
    kx_m_inv: complex,
) -> complex:
    try:
        stack = _build_stack_from_rotated_eps(
            wavelength_nm=wavelength_nm,
            eps_tensor=eps_tensor,
            kx_m_inv=kx_m_inv,
        )
        reflection_matrix = _reflection_matrix_in_physical_ps_basis(stack)
    except Exception:
        return np.nan + 1j * np.nan

    if MODE_IDX == 1:
        return complex(reflection_matrix[0, 0])
    if MODE_IDX == 0:
        return complex(reflection_matrix[1, 1])
    raise ValueError(f"Unsupported MODE_IDX={MODE_IDX}. Expected 0 or 1.")


def _batch_mode_reflection(
    wavelength_nm: float,
    eps_tensor: np.ndarray,
    kx_values_m_inv: np.ndarray,
) -> np.ndarray:
    values = np.empty(kx_values_m_inv.shape[0], dtype=np.complex128)
    for idx, kx_m_inv in enumerate(kx_values_m_inv):
        values[idx] = _mode_reflection_amplitude_rotated(
            wavelength_nm=wavelength_nm,
            eps_tensor=eps_tensor,
            kx_m_inv=complex(kx_m_inv),
        )
    return values


def _inv_r_residual(
    x: np.ndarray,
    wavelength_nm: float,
    eps_tensor: np.ndarray,
) -> np.ndarray:
    kx_m_inv = complex(x[0], x[1])
    if not (
        KX_RE_RANGE_M_INV[0] <= x[0] <= KX_RE_RANGE_M_INV[1]
        and KX_IM_RANGE_M_INV[0] <= x[1] <= KX_IM_RANGE_M_INV[1]
    ):
        return np.array([1e10, 1e10], dtype=np.float64)

    r_value = _mode_reflection_amplitude_rotated(
        wavelength_nm=wavelength_nm,
        eps_tensor=eps_tensor,
        kx_m_inv=kx_m_inv,
    )
    if not np.isfinite(r_value.real) or not np.isfinite(r_value.imag) or abs(r_value) == 0.0:
        return np.array([1e10, 1e10], dtype=np.float64)

    inv_r = 1.0 / r_value
    return np.array([inv_r.real, inv_r.imag], dtype=np.float64)


def find_poles(
    wavelength_nm: float,
    eps_tensor: np.ndarray,
    n_coarse_re: int = POLE_GRID_RE_POINTS,
    n_coarse_im: int = POLE_GRID_IM_POINTS,
    peak_neighborhood: int = PEAK_NEIGHBORHOOD,
    peak_threshold: float = PEAK_THRESHOLD,
    max_refine: int = MAX_REFINE,
) -> list[tuple[complex, float, float]]:
    """Find poles of the selected reflection channel in the complex kx plane."""
    kx_re_grid = np.linspace(*KX_RE_RANGE_M_INV, n_coarse_re)
    kx_im_grid = np.linspace(*KX_IM_RANGE_M_INV, n_coarse_im)
    kx_re_2d, kx_im_2d = np.meshgrid(kx_re_grid, kx_im_grid, indexing="ij")
    kx_flat = (kx_re_2d + 1j * kx_im_2d).ravel()

    r_batch = _batch_mode_reflection(
        wavelength_nm=wavelength_nm,
        eps_tensor=eps_tensor,
        kx_values_m_inv=kx_flat,
    )
    abs_r = np.abs(r_batch.reshape(n_coarse_re, n_coarse_im))
    finite_abs_r = np.where(np.isfinite(abs_r), abs_r, 0.0)

    local_max = maximum_filter(finite_abs_r, size=peak_neighborhood, mode="nearest")
    is_peak = np.isclose(finite_abs_r, local_max) & (
        finite_abs_r > np.median(finite_abs_r) * peak_threshold
    )
    labeled, n_peaks = label(is_peak)

    if n_peaks == 0:
        return []

    candidates: list[tuple[float, float]] = []
    for peak_idx in range(1, n_peaks + 1):
        region = np.argwhere(labeled == peak_idx)
        region_values = finite_abs_r[region[:, 0], region[:, 1]]
        best_ij = region[int(np.argmax(region_values))]
        candidates.append((kx_re_grid[int(best_ij[0])], kx_im_grid[int(best_ij[1])]))

    candidates.sort(key=lambda xy: xy[0])
    candidates = candidates[:max_refine]

    poles: list[tuple[complex, float, float]] = []
    for kx_re0, kx_im0 in candidates:
        try:
            solution = root(
                _inv_r_residual,
                np.array([kx_re0, kx_im0], dtype=np.float64),
                args=(wavelength_nm, eps_tensor),
                method="hybr",
                tol=ROOT_TOL,
            )
        except Exception:
            solution = None

        if solution is not None and solution.success:
            kx_candidate = complex(float(solution.x[0]), float(solution.x[1]))
        else:
            kx_candidate = complex(kx_re0, kx_im0)

        if not (
            KX_RE_RANGE_M_INV[0] <= kx_candidate.real <= KX_RE_RANGE_M_INV[1]
            and KX_IM_RANGE_M_INV[0] <= kx_candidate.imag <= KX_IM_RANGE_M_INV[1]
        ):
            continue

        r_value = _mode_reflection_amplitude_rotated(
            wavelength_nm=wavelength_nm,
            eps_tensor=eps_tensor,
            kx_m_inv=kx_candidate,
        )
        if not np.isfinite(r_value.real) or not np.isfinite(r_value.imag):
            continue

        magnitude = abs(r_value)
        residual = abs(1.0 / r_value) if magnitude != 0.0 else np.inf
        if magnitude < POLE_MAGNITUDE_THRESHOLD:
            continue

        poles.append((kx_candidate, magnitude, residual))

    unique: list[tuple[complex, float, float]] = []
    for candidate in poles:
        kx_candidate = candidate[0]
        if all(abs(kx_candidate - existing[0]) > 1.0 for existing in unique):
            unique.append(candidate)

    unique.sort(key=lambda item: item[0].real)
    return unique


def _single_frame_plot(
    specs: list[SweepSpec],
    angle_deg: float,
    out_path: pathlib.Path,
) -> None:
    theta_rad = np.deg2rad(angle_deg)
    kx_re_grid = np.linspace(*KX_RE_RANGE_M_INV, FRAME_GRID_POINTS)
    kx_im_grid = np.linspace(*KX_IM_RANGE_M_INV, FRAME_GRID_POINTS)
    kx_re_2d, kx_im_2d = np.meshgrid(kx_re_grid, kx_im_grid, indexing="ij")
    kx_flat = (kx_re_2d + 1j * kx_im_2d).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, spec in zip(axes, specs):
        eps_tensor = rotate_eps_xy(spec.eps_diag, theta_rad)
        start_time = time.time()
        r_batch = _batch_mode_reflection(
            wavelength_nm=spec.wavelength_nm,
            eps_tensor=eps_tensor,
            kx_values_m_inv=kx_flat,
        )
        print(
            f"[NbOCl2Waveguiding] Single frame {spec.label} computed in "
            f"{time.time() - start_time:.1f}s"
        )
        abs_r = np.abs(r_batch).reshape(FRAME_GRID_POINTS, FRAME_GRID_POINTS)
        values = abs_r
        image = ax.pcolormesh(
            kx_re_grid * 1e-6,
            kx_im_grid * 1e-6,
            values.T,
            cmap="hot",
            shading="auto",
        )
        fig.colorbar(image, ax=ax, label=rf"$|r_{{{'ss' if MODE_IDX == 1 else 'pp'}}}|$")
        ax.set_xlabel(r"Re($k_x$) ($\mu m^{-1}$)")
        ax.set_ylabel(r"Im($k_x$) ($\mu m^{-1}$)")
        ax.set_title(rf"{spec.label}, $\theta={angle_deg:.0f}^\circ$")

    fig.suptitle(
        rf"$|r_{{{'ss' if MODE_IDX == 1 else 'pp'}}}|$ in complex $k_x$ plane, "
        rf"$d={THICKNESS_NM:.0f}$ nm"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)


def _sweep_angle_poles(spec: SweepSpec) -> list[PolePoint]:
    points: list[PolePoint] = []
    start_time = time.time()
    for angle_deg in ANGLE_DEG_SAMPLES:
        eps_tensor = rotate_eps_xy(spec.eps_diag, np.deg2rad(angle_deg))
        poles = find_poles(
            wavelength_nm=spec.wavelength_nm,
            eps_tensor=eps_tensor,
        )
        for kx_candidate, magnitude, residual in poles:
            im_um_inv = kx_candidate.imag * 1e-6
            if im_um_inv < IM_KX_PLOT_MIN_UM_INV:
                continue
            points.append(
                PolePoint(
                    angle_deg=float(angle_deg),
                    wavelength_nm=spec.wavelength_nm,
                    kx_m_inv=kx_candidate,
                    residual=float(residual),
                    magnitude=float(magnitude),
                )
            )
        print(
            f"[NbOCl2Waveguiding] {spec.label}, angle={angle_deg:5.1f} deg: "
            f"{len(poles)} poles"
        )

    print(f"[NbOCl2Waveguiding] Completed {spec.label} in {time.time() - start_time:.1f}s")
    return points


def _write_pole_csv(out_path: pathlib.Path, pole_points: list[PolePoint]) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "wavelength_nm",
                "angle_deg",
                "kx_re_um_inv",
                "kx_im_um_inv",
                "residual_inv_r",
                "abs_r",
            ]
        )
        for point in pole_points:
            writer.writerow(
                [
                    point.wavelength_nm,
                    point.angle_deg,
                    point.kx_m_inv.real * 1e-6,
                    point.kx_m_inv.imag * 1e-6,
                    point.residual,
                    point.magnitude,
                ]
            )


def _scatter_arrays(points: list[PolePoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([point.angle_deg for point in points], dtype=np.float64),
        np.array([point.kx_m_inv.real * 1e-6 for point in points], dtype=np.float64),
        np.array([point.kx_m_inv.imag * 1e-6 for point in points], dtype=np.float64),
    )


def _plot_angle_sweep(
    specs: list[SweepSpec],
    points_by_wavelength: dict[float, list[PolePoint]],
    out_path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, spec in zip(axes, specs):
        angle_deg, re_kx_um_inv, im_kx_um_inv = _scatter_arrays(
            points_by_wavelength[spec.wavelength_nm]
        )
        scatter = ax.scatter(
            angle_deg,
            re_kx_um_inv,
            c=im_kx_um_inv,
            cmap="coolwarm",
            s=15,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax, label=r"Im($k_x$) ($\mu m^{-1}$)")
        ax.set_xlabel(r"Rotation angle $\theta$ (deg)")
        ax.set_ylabel(r"Re($k_x$) of mode ($\mu m^{-1}$)")
        ax.set_title(
            rf"NbOCl$_2$ modes vs rotation, {spec.label}, $d={THICKNESS_NM:.0f}$ nm"
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def extract_branches(
    angles_deg: np.ndarray,
    re_kx_um_inv: np.ndarray,
    continuity_threshold: float = 3.0,
) -> tuple[list[list[float]], list[list[float]]]:
    """Group scatter points into approximately continuous branches."""
    unique_angles = np.sort(np.unique(angles_deg))
    per_angle = {angle: sorted(re_kx_um_inv[angles_deg == angle]) for angle in unique_angles}

    branches: list[tuple[list[float], list[float]]] = []
    for angle_deg in unique_angles:
        values = list(per_angle[angle_deg])
        used = [False] * len(values)

        for branch_angles, branch_values in branches:
            if not values:
                break
            last_value = branch_values[-1]
            best_idx = None
            best_dist = np.inf
            for idx, value in enumerate(values):
                if not used[idx] and abs(value - last_value) < best_dist:
                    best_idx = idx
                    best_dist = abs(value - last_value)
            if best_idx is not None and best_dist < continuity_threshold:
                branch_angles.append(float(angle_deg))
                branch_values.append(float(values[best_idx]))
                used[best_idx] = True

        for idx, value in enumerate(values):
            if not used[idx]:
                branches.append(([float(angle_deg)], [float(value)]))

    branch_angles = [branch[0] for branch in branches]
    branch_values = [branch[1] for branch in branches]
    return branch_angles, branch_values


def _find_branch_intersections(
    angles_810_deg: np.ndarray,
    re_810_um_inv: np.ndarray,
    angles_405_deg: np.ndarray,
    re_405_um_inv: np.ndarray,
) -> list[tuple[float, float]]:
    from scipy.interpolate import interp1d

    branch_ang_810, branch_re_810 = extract_branches(angles_810_deg, 2.0 * re_810_um_inv)
    branch_ang_405, branch_re_405 = extract_branches(angles_405_deg, re_405_um_inv)

    intersections: list[tuple[float, float]] = []
    for ang_810, re_810 in zip(branch_ang_810, branch_re_810):
        if len(ang_810) < 2:
            continue
        interp_810 = interp1d(ang_810, re_810, bounds_error=False, fill_value=np.nan)
        for ang_405, re_405 in zip(branch_ang_405, branch_re_405):
            if len(ang_405) < 2:
                continue
            interp_405 = interp1d(ang_405, re_405, bounds_error=False, fill_value=np.nan)
            angle_min = max(min(ang_810), min(ang_405))
            angle_max = min(max(ang_810), max(ang_405))
            if angle_min >= angle_max:
                continue

            angle_fine = np.linspace(angle_min, angle_max, 1000)
            diff = interp_810(angle_fine) - interp_405(angle_fine)
            valid = ~np.isnan(diff)
            diff_valid = diff[valid]
            angle_valid = angle_fine[valid]
            sign_changes = np.where(np.diff(np.sign(diff_valid)))[0]
            for idx in sign_changes:
                angle_cross = angle_valid[idx] - diff_valid[idx] * (
                    angle_valid[idx + 1] - angle_valid[idx]
                ) / (diff_valid[idx + 1] - diff_valid[idx])
                re_cross = float(interp_810(angle_cross))
                intersections.append((float(angle_cross), re_cross))

    return intersections


def _plot_phase_matching_overlay(
    points_810: list[PolePoint],
    points_405: list[PolePoint],
    out_path: pathlib.Path,
    intersections_csv_path: pathlib.Path,
) -> None:
    angles_810_deg, re_810_um_inv, _ = _scatter_arrays(points_810)
    angles_405_deg, re_405_um_inv, _ = _scatter_arrays(points_405)

    intersections = _find_branch_intersections(
        angles_810_deg=angles_810_deg,
        re_810_um_inv=re_810_um_inv,
        angles_405_deg=angles_405_deg,
        re_405_um_inv=re_405_um_inv,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        angles_810_deg,
        2.0 * re_810_um_inv,
        c="crimson",
        s=20,
        edgecolors="none",
        label=r"$2 \times \mathrm{Re}(k_x)$ at 810 nm",
    )
    ax.scatter(
        angles_405_deg,
        re_405_um_inv,
        c="dodgerblue",
        s=20,
        edgecolors="none",
        label=r"$\mathrm{Re}(k_x)$ at 405 nm",
    )
    for angle_cross, re_cross in intersections:
        ax.plot(angle_cross, re_cross, "k*", markersize=13, zorder=5)
        ax.annotate(
            f"{angle_cross:.1f}°",
            (angle_cross, re_cross),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel(r"Rotation angle $\theta$ (deg)")
    ax.set_ylabel(r"Re($k_x$) ($\mu m^{-1}$)")
    ax.set_title(rf"Phase matching: NbOCl$_2$, $d={THICKNESS_NM:.0f}$ nm")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    with intersections_csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_deg", "re_kx_um_inv"])
        for angle_cross, re_cross in intersections:
            writer.writerow([angle_cross, re_cross])


def main() -> None:
    specs = [
        SweepSpec(wavelength_nm=810.0, eps_diag=EPS_DIAG_810, label=r"$\lambda = 810$ nm"),
        SweepSpec(wavelength_nm=405.0, eps_diag=EPS_DIAG_405, label=r"$\lambda = 405$ nm"),
    ]
    out_dir = pathlib.Path(__file__).resolve().parent

    theta_39_rad = np.deg2rad(39.0)
    print(f"Thickness: {THICKNESS_NM:.0f} nm")
    print(f"eps_810 rotated 39°:\n{rotate_eps_xy(EPS_DIAG_810, theta_39_rad)}\n")
    print(f"eps_405 rotated 39°:\n{rotate_eps_xy(EPS_DIAG_405, theta_39_rad)}\n")

    _single_frame_plot(
        specs=specs,
        angle_deg=SINGLE_ANGLE_DEG,
        out_path=out_dir / "RotatedModes_single_frame.png",
    )
    plt.show()
    points_by_wavelength: dict[float, list[PolePoint]] = {}
    all_points: list[PolePoint] = []
    for spec in specs:
        points = _sweep_angle_poles(spec)
        points_by_wavelength[spec.wavelength_nm] = points
        all_points.extend(points)

    _write_pole_csv(out_dir / "nbocl2_twist_poles.csv", all_points)
    np.savez_compressed(
        out_dir / "nbocl2_twist_poles.npz",
        wavelength_nm=np.array([point.wavelength_nm for point in all_points], dtype=np.float64),
        angle_deg=np.array([point.angle_deg for point in all_points], dtype=np.float64),
        kx_m_inv=np.array([point.kx_m_inv for point in all_points], dtype=np.complex128),
        residual=np.array([point.residual for point in all_points], dtype=np.float64),
        magnitude=np.array([point.magnitude for point in all_points], dtype=np.float64),
    )

    _plot_angle_sweep(
        specs=specs,
        points_by_wavelength=points_by_wavelength,
        out_path=out_dir / "RotatedModes_vs_angle.png",
    )
    _plot_phase_matching_overlay(
        points_810=points_by_wavelength[810.0],
        points_405=points_by_wavelength[405.0],
        out_path=out_dir / "RotatedModes_phase_matching.png",
        intersections_csv_path=out_dir / "nbocl2_phase_matching_intersections.csv",
    )

    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'nbocl2_twist_poles.csv'}")
    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'nbocl2_twist_poles.npz'}")
    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'RotatedModes_single_frame.png'}")
    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'RotatedModes_vs_angle.png'}")
    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'RotatedModes_phase_matching.png'}")
    print(f"[NbOCl2Waveguiding] Wrote {out_dir / 'nbocl2_phase_matching_intersections.csv'}")


if __name__ == "__main__":
    main()
