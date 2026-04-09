"""Compare RCWAFFF and pyGTM reflection amplitudes for homogeneous stacks.

This script builds the same physical stack in both solvers and reports the
external reflection matrix in the p/s basis used by pyGTM.

Two bookkeeping details matter:

1. RCWAFFF propagates from ``Stack.eps_substrate`` to ``Stack.eps_superstrate``,
   while pyGTM is incident from ``System.superstrate`` toward ``System.substrate``.
2. For an in-plane tensor rotation ``R @ eps_local @ R.T`` in RCWAFFF, pyGTM
   needs ``phi = -rotation_angle``.

The isotropic calibration row verifies that the port-basis conversion is
correct before the anisotropic comparisons are printed.
"""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "pyGTM"))

from rcwa import Layer, Solver, Stack
import GTM.GTMcore as GTM


C_M_PER_S = 299792458.0
DEFAULT_X_DOMAIN_NM = (0.0, 500.0)


@dataclass(frozen=True)
class LayerSpec:
    thickness_nm: float
    eps1: complex
    eps2: complex
    eps3: complex
    angle_deg: float = 0.0


@dataclass(frozen=True)
class CaseSpec:
    name: str
    wavelength_nm: float
    zeta: complex
    eps_incident: complex
    eps_exit: complex
    layers: tuple[LayerSpec, ...] = ()


def _const_eps(value: complex):
    return lambda frequency_hz, _value=complex(value): _value


def _rotate_in_plane_eps(eps_local: np.ndarray, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    return rotation @ eps_local @ rotation.T


def _forward_q(eps: complex, kappa_normalized: complex) -> complex:
    q = np.sqrt(complex(eps) - complex(kappa_normalized) ** 2)
    if np.imag(q) < 0.0 or (abs(np.imag(q)) < 1e-14 and np.real(q) < 0.0):
        q = -q
    return q


def _pygtm_like_port_fields(eps: complex, kappa_normalized: complex) -> np.ndarray:
    """Return the isotropic external p/s port basis compatible with pyGTM."""
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


def _build_rcwa_stack(case: CaseSpec) -> Stack:
    kappa_inv_nm = complex(case.zeta) * (2.0 * np.pi / case.wavelength_nm)
    stack = Stack(
        wavelength_nm=case.wavelength_nm,
        kappa_inv_nm=kappa_inv_nm,
        eps_substrate=case.eps_incident,
        eps_superstrate=case.eps_exit,
    )

    if not case.layers:
        stack.add_layer(
            Layer.uniform(
                thickness_nm=0.0,
                eps_tensor=np.asarray(case.eps_incident, dtype=np.complex128) * np.eye(3, dtype=np.complex128),
                x_domain_nm=DEFAULT_X_DOMAIN_NM,
            )
        )
        return stack

    for layer in case.layers:
        eps_local = np.diag(
            np.array([layer.eps1, layer.eps2, layer.eps3], dtype=np.complex128)
        )
        stack.add_layer(
            Layer.uniform(
                thickness_nm=layer.thickness_nm,
                eps_tensor=_rotate_in_plane_eps(eps_local, layer.angle_deg),
                x_domain_nm=DEFAULT_X_DOMAIN_NM,
            )
        )
    return stack


def _build_pygtm_system(case: CaseSpec) -> GTM.System:
    system = GTM.System()
    system.set_superstrate(GTM.Layer(epsilon1=_const_eps(case.eps_incident)))
    system.set_substrate(GTM.Layer(epsilon1=_const_eps(case.eps_exit)))

    for layer in case.layers:
        system.add_layer(
            GTM.Layer(
                thickness=layer.thickness_nm * 1e-9,
                epsilon1=_const_eps(layer.eps1),
                epsilon2=_const_eps(layer.eps2),
                epsilon3=_const_eps(layer.eps3),
                phi=np.deg2rad(-layer.angle_deg),
            )
        )
    return system


def _rcwa_reflection_matrix_in_pygtm_basis(
    case: CaseSpec,
    num_points: int = 256,
) -> np.ndarray:
    stack = _build_rcwa_stack(case)
    scattering_solver = Solver.total_scattering_matrix(
        stack,
        0,
        num_points=num_points,
        verbose=False,
    )
    substrate_solver_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(0, num_points=num_points),
        0,
    )
    superstrate_solver_fields = Solver.isotropic_mode_fields(
        stack.get_Q_superstrate_normalized(0, num_points=num_points),
        0,
    )
    substrate_tangential_fields = Solver.reduced_to_tangential_fields(
        substrate_solver_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_substrate,
            0,
        ),
    )
    superstrate_tangential_fields = Solver.reduced_to_tangential_fields(
        superstrate_solver_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_superstrate,
            0,
        ),
    )

    substrate_pygtm_fields = _pygtm_like_port_fields(
        stack.eps_substrate,
        stack.kappa_normalized,
    )
    superstrate_pygtm_fields = _pygtm_like_port_fields(
        stack.eps_superstrate,
        stack.kappa_normalized,
    )

    scattering_physical = Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(
                substrate_pygtm_fields,
                substrate_tangential_fields,
            ),
            scattering_solver,
            Solver.basis_change_scattering_matrix(
                superstrate_tangential_fields,
                superstrate_pygtm_fields,
            ),
        ]
    )
    return np.asarray(scattering_physical[0])


def _pygtm_reflection_matrix(case: CaseSpec) -> np.ndarray:
    system = _build_pygtm_system(case)
    frequency_hz = C_M_PER_S / (case.wavelength_nm * 1e-9)
    system.initialize_sys(frequency_hz)
    system.calculate_GammaStar(frequency_hz, case.zeta)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_out, _, _, _ = system.calculate_r_t(case.zeta)
    return np.array(
        [
            [r_out[2], r_out[1]],
            [r_out[3], r_out[0]],
        ],
        dtype=np.complex128,
    )


def _format_complex(value: complex) -> str:
    return f"{np.real(value):+0.6f}{np.imag(value):+0.6f}j"


def _compare_case(case: CaseSpec) -> dict[str, object]:
    rcwa_r = _rcwa_reflection_matrix_in_pygtm_basis(case)
    pygtm_r = _pygtm_reflection_matrix(case)
    diff = np.abs(rcwa_r - pygtm_r)
    return {
        "case": case,
        "rcwa_r": rcwa_r,
        "pygtm_r": pygtm_r,
        "rpp_diff": float(diff[1, 1]),
        "rss_diff": float(diff[0, 0]),
        "cross_diff_max": float(max(diff[0, 1], diff[1, 0])),
    }


def _default_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            name="isotropic_interface_calibration",
            wavelength_nm=600.0,
            zeta=0.5,
            eps_incident=1.0,
            eps_exit=1.5**2,
        ),
        CaseSpec(
            name="axis_aligned_biaxial_film",
            wavelength_nm=633.0,
            zeta=np.sin(np.deg2rad(32.0)),
            eps_incident=1.0,
            eps_exit=1.7**2,
            layers=(
                LayerSpec(
                    thickness_nm=120.0,
                    eps1=(1.6 + 0.02j) ** 2,
                    eps2=(2.05 + 0.03j) ** 2,
                    eps3=(1.8 + 0.01j) ** 2,
                    angle_deg=0.0,
                ),
            ),
        ),
        CaseSpec(
            name="nbocl2_hbn_axis",
            wavelength_nm=780.0,
            zeta=1.4,
            eps_incident=1.0,
            eps_exit=3.674**2,
            layers=(
                LayerSpec(
                    thickness_nm=122.0,
                    eps1=4.0,
                    eps2=6.0,
                    eps3=1.6**2,
                    angle_deg=0.0,
                ),
                LayerSpec(
                    thickness_nm=90.0,
                    eps1=1.4531**2,
                    eps2=1.4531**2,
                    eps3=1.4531**2,
                    angle_deg=0.0,
                ),
            ),
        ),
        CaseSpec(
            name="nbocl2_hbn_rot27",
            wavelength_nm=780.0,
            zeta=1.4,
            eps_incident=1.0,
            eps_exit=3.674**2,
            layers=(
                LayerSpec(
                    thickness_nm=122.0,
                    eps1=4.0,
                    eps2=6.0,
                    eps3=1.6**2,
                    angle_deg=27.0,
                ),
                LayerSpec(
                    thickness_nm=90.0,
                    eps1=1.4531**2,
                    eps2=1.4531**2,
                    eps3=1.4531**2,
                    angle_deg=0.0,
                ),
            ),
        ),
    ]


def main() -> None:
    results = [_compare_case(case) for case in _default_cases()]

    print("RCWAFFF vs pyGTM reflection comparison")
    print("RCWAFFF amplitudes are first converted into the same external p/s basis used by pyGTM.")
    print()
    print(
        f"{'case':<30} {'wl_nm':>8} {'zeta':>8} "
        f"{'rcwa_rpp':>28} {'pygtm_rpp':>28} "
        f"{'|drpp|':>12} {'|drss|':>12} {'|dcross|max':>14}"
    )
    print("-" * 150)

    for result in results:
        case = result["case"]
        rcwa_r = result["rcwa_r"]
        pygtm_r = result["pygtm_r"]
        print(
            f"{case.name:<30} "
            f"{case.wavelength_nm:8.1f} "
            f"{np.real(case.zeta):8.3f} "
            f"{_format_complex(rcwa_r[1, 1]):>28} "
            f"{_format_complex(pygtm_r[1, 1]):>28} "
            f"{result['rpp_diff']:12.4e} "
            f"{result['rss_diff']:12.4e} "
            f"{result['cross_diff_max']:14.4e}"
        )


if __name__ == "__main__":
    main()
