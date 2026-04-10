"""Compare the theta=0 complex-kx NbOCl2 map against pyGTM.

This script focuses on the single-layer homogeneous stack at 405 nm and
produces three maps on the same complex-kx grid:

1. The current RCWA path used by ``sweep_twist_roots.py``.
2. A corrected RCWA modal-TE reflection map with air as the incident side.
3. The pyGTM reference ``r_ss`` map.

The goal is to make the failure mode explicit before returning to the full
twist-angle sweep.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(pathlib.Path(__file__).resolve().parent / ".matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "pyGTM"))

from rcwa import Layer, Solver, Stack
import GTM.GTMcore as GTM


C_M_PER_S = 299792458.0
THICKNESS_M = 122e-9
THICKNESS_NM = THICKNESS_M * 1e9
WAVELENGTH_NM = 405.0
WAVELENGTH_M = WAVELENGTH_NM * 1e-9
K0_M_INV = 2.0 * np.pi / WAVELENGTH_M
EPS_SUBSTRATE = 1.4696**2
EPS_SUPERSTRATE = 1.0
EPS_DIAG_405 = np.diag([5.406, 10.465 + 1.923j, 1.6**2]).astype(np.complex128)
X_DOMAIN_NM = (0.0, 500.0)

KX_RE_RANGE_M_INV = (0.5e7, 5.0e7)
KX_IM_RANGE_M_INV = (-10.0e6, 10.0e6)
NUM_KX_RE = 161
NUM_KX_IM = 121
NUM_POINTS_RCWA = 16


def rotate_eps_xy(eps_diag: np.ndarray, theta_rad: float) -> np.ndarray:
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


def _const_eps(value: complex):
    return lambda frequency_hz, _value=complex(value): _value


def _forward_q(eps: complex, kappa_normalized: complex) -> complex:
    q = np.sqrt(complex(eps) - complex(kappa_normalized) ** 2)
    if np.imag(q) < 0.0 or (abs(np.imag(q)) < 1e-14 and np.real(q) < 0.0):
        q = -q
    return q


def _legacy_port_fields(eps: complex, kappa_normalized: complex) -> np.ndarray:
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


def _build_stack(
    eps_incident: complex,
    eps_exit: complex,
    eps_tensor: np.ndarray,
    kx_m_inv: complex,
) -> Stack:
    stack = Stack(
        wavelength_nm=WAVELENGTH_NM,
        kappa_inv_nm=complex(kx_m_inv) * 1e-9,
        eps_substrate=eps_incident,
        eps_superstrate=eps_exit,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=THICKNESS_NM,
            eps_tensor=eps_tensor,
            x_domain_nm=X_DOMAIN_NM,
        )
    )
    return stack


def _legacy_rcwa_rss(eps_tensor: np.ndarray, kx_m_inv: complex) -> complex:
    """Reproduce the current sweep script's RCWA reflection path."""
    stack = _build_stack(
        eps_incident=EPS_SUBSTRATE,
        eps_exit=EPS_SUPERSTRATE,
        eps_tensor=eps_tensor,
        kx_m_inv=kx_m_inv,
    )
    solve_data = Solver.build_stack_solve_data(
        stack,
        0,
        num_points=NUM_POINTS_RCWA,
        verbose=False,
    )
    scattering_physical = Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(
                _legacy_port_fields(stack.eps_substrate, stack.kappa_normalized),
                solve_data.substrate_continuity_fields,
            ),
            solve_data.total_scattering,
        ]
    )
    return complex(np.asarray(scattering_physical[0])[0, 0])


def _corrected_modal_te_reflection(eps_tensor: np.ndarray, kx_m_inv: complex) -> complex:
    """Use the RCWA modal TE reflection with air as the incident half-space."""
    stack = _build_stack(
        eps_incident=EPS_SUPERSTRATE,
        eps_exit=EPS_SUBSTRATE,
        eps_tensor=eps_tensor,
        kx_m_inv=kx_m_inv,
    )
    S11, _, _, _ = Solver.total_scattering_matrix(
        stack,
        0,
        num_points=NUM_POINTS_RCWA,
        verbose=False,
    )
    te_idx = Solver.zero_order_mode_index(0, "TE")
    return complex(np.asarray(S11)[te_idx, te_idx])


def _build_pygtm_system(eps_tensor: np.ndarray) -> GTM.System:
    system = GTM.System()
    system.set_superstrate(GTM.Layer(epsilon1=_const_eps(EPS_SUPERSTRATE)))
    system.set_substrate(GTM.Layer(epsilon1=_const_eps(EPS_SUBSTRATE)))
    system.add_layer(
        GTM.Layer(
            thickness=THICKNESS_M,
            epsilon1=_const_eps(eps_tensor[0, 0]),
            epsilon2=_const_eps(eps_tensor[1, 1]),
            epsilon3=_const_eps(eps_tensor[2, 2]),
        )
    )
    system.initialize_sys(C_M_PER_S / WAVELENGTH_M)
    return system


def _pygtm_rss(system: GTM.System, kx_m_inv: complex) -> complex:
    zeta = complex(kx_m_inv) / K0_M_INV
    system.calculate_GammaStar(C_M_PER_S / WAVELENGTH_M, zeta)
    r_out, _, _, _ = system.calculate_r_t(zeta)
    return complex(r_out[2])


def _compute_map(
    label: str,
    evaluator,
    kx_values: np.ndarray,
) -> np.ndarray:
    values = np.empty(kx_values.shape[0], dtype=np.complex128)
    start_time = time.time()
    for idx, kx_m_inv in enumerate(kx_values):
        try:
            values[idx] = evaluator(complex(kx_m_inv))
        except Exception:
            values[idx] = np.nan + 1j * np.nan
        if idx % max(1, kx_values.shape[0] // 10) == 0:
            print(f"[NbOCl2Waveguiding] {label}: {idx}/{kx_values.shape[0]}")
    print(
        f"[NbOCl2Waveguiding] {label} finished in {time.time() - start_time:.1f}s"
    )
    return values


def _magnitude_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    candidate_abs = np.abs(candidate)
    reference_abs = np.abs(reference)
    mask = np.isfinite(candidate_abs) & np.isfinite(reference_abs)
    if not np.any(mask):
        return {
            "finite_count": 0.0,
            "rmse_abs": np.nan,
            "mae_abs": np.nan,
            "p99_abs_diff": np.nan,
            "max_abs_diff": np.nan,
        }
    diff = np.abs(candidate_abs[mask] - reference_abs[mask])
    return {
        "finite_count": float(mask.sum()),
        "rmse_abs": float(np.sqrt(np.mean(diff**2))),
        "mae_abs": float(np.mean(diff)),
        "p99_abs_diff": float(np.percentile(diff, 99.0)),
        "max_abs_diff": float(np.max(diff)),
    }


def _log10_magnitude(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    out = np.full(magnitude.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(magnitude) & (magnitude > 0.0)
    out[mask] = np.log10(magnitude[mask])
    return out


def _plot_maps(
    legacy_map: np.ndarray,
    modal_map: np.ndarray,
    pygtm_map: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    legacy_log = _log10_magnitude(legacy_map)
    modal_log = _log10_magnitude(modal_map)
    pygtm_log = _log10_magnitude(pygtm_map)

    common = np.concatenate(
        [
            legacy_log[np.isfinite(legacy_log)],
            modal_log[np.isfinite(modal_log)],
            pygtm_log[np.isfinite(pygtm_log)],
        ]
    )
    vmin = float(np.percentile(common, 2.0))
    vmax = float(np.percentile(common, 98.0))
    extent = (
        KX_RE_RANGE_M_INV[0] * 1e-6,
        KX_RE_RANGE_M_INV[1] * 1e-6,
        KX_IM_RANGE_M_INV[0] * 1e-6,
        KX_IM_RANGE_M_INV[1] * 1e-6,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    images = [
        axes[0].imshow(
            legacy_log.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        ),
        axes[1].imshow(
            modal_log.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        ),
        axes[2].imshow(
            pygtm_log.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="hot",
            vmin=vmin,
            vmax=vmax,
        ),
    ]
    titles = [
        "RCWA legacy sweep path",
        "RCWA corrected modal TE",
        "pyGTM r_ss",
    ]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel(r"Re($k_x$) ($\mu m^{-1}$)")
        ax.set_ylabel(r"Im($k_x$) ($\mu m^{-1}$)")
    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.95)
    cbar.set_label(r"$\log_{10}|r|$")
    fig.suptitle(
        rf"NbOCl$_2$ theta=0 complex-$k_x$ map, $\lambda={WAVELENGTH_NM:.0f}$ nm, "
        rf"$d={THICKNESS_NM:.0f}$ nm"
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parent
    eps_tensor = rotate_eps_xy(EPS_DIAG_405, 0.0)

    kx_re = np.linspace(*KX_RE_RANGE_M_INV, NUM_KX_RE)
    kx_im = np.linspace(*KX_IM_RANGE_M_INV, NUM_KX_IM)
    kx_re_2d, kx_im_2d = np.meshgrid(kx_re, kx_im, indexing="ij")
    kx_flat = (kx_re_2d + 1j * kx_im_2d).ravel()

    pygtm_system = _build_pygtm_system(eps_tensor)

    legacy_flat = _compute_map(
        "Legacy RCWA",
        lambda kx: _legacy_rcwa_rss(eps_tensor, kx),
        kx_flat,
    )
    modal_flat = _compute_map(
        "Corrected RCWA modal TE",
        lambda kx: _corrected_modal_te_reflection(eps_tensor, kx),
        kx_flat,
    )
    pygtm_flat = _compute_map(
        "pyGTM r_ss",
        lambda kx: _pygtm_rss(pygtm_system, kx),
        kx_flat,
    )

    legacy_map = legacy_flat.reshape(NUM_KX_RE, NUM_KX_IM)
    modal_map = modal_flat.reshape(NUM_KX_RE, NUM_KX_IM)
    pygtm_map = pygtm_flat.reshape(NUM_KX_RE, NUM_KX_IM)

    legacy_metrics = _magnitude_metrics(legacy_map, pygtm_map)
    modal_metrics = _magnitude_metrics(modal_map, pygtm_map)
    print("[NbOCl2Waveguiding] Legacy RCWA vs pyGTM:", legacy_metrics)
    print("[NbOCl2Waveguiding] Modal RCWA vs pyGTM:", modal_metrics)

    np.savez_compressed(
        out_dir / "nbocl2_theta0_complex_kx_compare.npz",
        kx_re_m_inv=kx_re,
        kx_im_m_inv=kx_im,
        legacy_rcwa=legacy_map,
        corrected_modal_rcwa=modal_map,
        pygtm=pygtm_map,
        legacy_rmse_abs=legacy_metrics["rmse_abs"],
        modal_rmse_abs=modal_metrics["rmse_abs"],
        legacy_mae_abs=legacy_metrics["mae_abs"],
        modal_mae_abs=modal_metrics["mae_abs"],
    )
    _plot_maps(
        legacy_map=legacy_map,
        modal_map=modal_map,
        pygtm_map=pygtm_map,
        out_path=out_dir / "nbocl2_theta0_complex_kx_compare.png",
    )
    print(
        f"[NbOCl2Waveguiding] Wrote "
        f"{out_dir / 'nbocl2_theta0_complex_kx_compare.npz'}"
    )
    print(
        f"[NbOCl2Waveguiding] Wrote "
        f"{out_dir / 'nbocl2_theta0_complex_kx_compare.png'}"
    )


if __name__ == "__main__":
    main()
