from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rcwa.layer import Layer
from rcwa.solver import Solver, ScatteringMatrix
from rcwa.stack import Stack


def physical_port_fields_matrix_harmonic_major(
    stack: Stack,
    eps: complex,
    N: int,
) -> np.ndarray:
    """Return the analytic isotropic port fields matrix.

    Important:
    - rows are left in harmonic-major tangential ordering
      ``[-H_y(n), H_x(n), E_y(n), E_x(n)]`` per harmonic
    - columns are in the solver modal layout
      ``[FTE(-N..N), FTM(-N..N), BTE(-N..N), BTM(-N..N)]``

    The RCWA tangential fields produced by the current stack helpers are in
    component-major row ordering, so callers that want to compare against those
    fields must reorder these rows with
    ``Solver.harmonic_to_component_major_rows(...)`` first.
    """
    num_h = Stack.num_harmonics(N)
    fields = np.zeros((4 * num_h, 4 * num_h), dtype=np.complex128)
    eps_c = np.asarray(eps, dtype=np.complex128)

    for h, order in enumerate(Stack.harmonic_orders(N)):
        kx = stack.kappa_normalized + order * stack.G_normalized
        q = np.sqrt(eps_c - kx**2)
        if np.imag(q) < 0.0 or (np.isclose(np.imag(q), 0.0) and np.real(q) < 0.0):
            q = -q

        block = np.stack(
            [
                np.array([0.0, -q, 1.0, 0.0], dtype=np.complex128),
                np.array([eps_c / q, 0.0, 0.0, -1.0], dtype=np.complex128),
                np.array([0.0, q, 1.0, 0.0], dtype=np.complex128),
                np.array([eps_c / q, 0.0, 0.0, 1.0], dtype=np.complex128),
            ],
            axis=1,
        )
        fields[4 * h : 4 * (h + 1), 4 * h : 4 * (h + 1)] = block

    modal_reorder = np.array(
        [4 * h + 0 for h in range(num_h)]
        + [4 * h + 1 for h in range(num_h)]
        + [4 * h + 2 for h in range(num_h)]
        + [4 * h + 3 for h in range(num_h)],
        dtype=np.int32,
    )
    return fields[:, modal_reorder]


def physical_basis_scattering(
    stack: Stack,
    N: int,
    *,
    component_major_physical_rows: bool,
    num_points: int = 256,
) -> ScatteringMatrix:
    """Convert the modal stack S-matrix into the analytic isotropic physical basis.

    The reduced solver uses ``D_x`` internally, while the physical port basis
    uses ``E_x``. That part is handled by the reduced-to-tangential transforms.
    The diagnostic bug we are checking here is different: a row-order mismatch
    between harmonic-major and component-major tangential field matrices.
    """
    S_modal = Solver.total_scattering_matrix(stack, N, num_points=num_points, verbose=False)

    substrate_reduced = Solver.harmonic_to_component_major_rows(
        Solver.get_substrate_mode_to_field(stack, N, num_points)
    )
    superstrate_reduced = Solver.harmonic_to_component_major_rows(
        Solver.get_superstrate_mode_to_field(stack, N, num_points)
    )
    substrate_tangential = (
        stack.substrate_reduced_to_tangential_field_transform_component_major(N)
        @ substrate_reduced
    )
    superstrate_tangential = (
        stack.superstrate_reduced_to_tangential_field_transform_component_major(N)
        @ superstrate_reduced
    )

    substrate_physical = physical_port_fields_matrix_harmonic_major(
        stack,
        stack.eps_substrate,
        N,
    )
    superstrate_physical = physical_port_fields_matrix_harmonic_major(
        stack,
        stack.eps_superstrate,
        N,
    )
    if component_major_physical_rows:
        substrate_physical = Solver.harmonic_to_component_major_rows(substrate_physical)
        superstrate_physical = Solver.harmonic_to_component_major_rows(superstrate_physical)

    return Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(substrate_physical, substrate_tangential),
            S_modal,
            Solver.basis_change_scattering_matrix(superstrate_tangential, superstrate_physical),
        ]
    )


def fresnel_interface(n1: complex, n2: complex) -> tuple[complex, complex]:
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def normal_incidence_characteristic_rt(
    n_incident: float,
    n_exit: float,
    layer_indices: list[float],
    layer_thicknesses_nm: list[float],
    wavelength_nm: float,
) -> tuple[complex, complex]:
    matrix = np.eye(2, dtype=np.complex128)

    for n_layer, thickness_nm in zip(layer_indices, layer_thicknesses_nm):
        phase = 2 * np.pi * n_layer * thickness_nm / wavelength_nm
        layer_matrix = np.array(
            [
                [np.cos(phase), 1j * np.sin(phase) / n_layer],
                [1j * n_layer * np.sin(phase), np.cos(phase)],
            ],
            dtype=np.complex128,
        )
        matrix = matrix @ layer_matrix

    B = matrix[0, 0] + matrix[0, 1] * n_exit
    C = matrix[1, 0] + matrix[1, 1] * n_exit
    reflection = (n_incident * B - C) / (n_incident * B + C)
    transmission = 2 * n_incident / (n_incident * B + C)
    return reflection, transmission


def basis_alignment_report(stack: Stack, N: int, num_points: int = 256) -> None:
    reduced = Solver.harmonic_to_component_major_rows(
        Solver.get_substrate_mode_to_field(stack, N, num_points)
    )
    tangential = stack.substrate_reduced_to_tangential_field_transform_component_major(N) @ reduced
    physical_harmonic = physical_port_fields_matrix_harmonic_major(stack, stack.eps_substrate, N)
    physical_component = Solver.harmonic_to_component_major_rows(physical_harmonic)

    wrong = np.linalg.solve(physical_harmonic, tangential)
    right = np.linalg.solve(physical_component, tangential)
    wrong_offdiag = np.linalg.norm(wrong - np.diag(np.diag(wrong)))
    right_offdiag = np.linalg.norm(right - np.diag(np.diag(right)))

    print(f"Basis alignment, N={N}")
    print(f"  offdiag ||B_wrong|| = {wrong_offdiag:.6g}")
    print(f"  offdiag ||B_right|| = {right_offdiag:.6g}")
    print(f"  diag(B_right) = {np.round(np.diag(right), 6)}")


def quarter_wave_report(N: int, use_component_major_physical_rows: bool) -> None:
    wavelength_nm = 550.0
    period_nm = 200.0
    n_incident = 1.0
    n_film = 2.0
    n_exit = 1.5
    thickness_nm = wavelength_nm / (4 * n_film)

    stack = Stack(
        layers=[
            Layer.uniform(
                thickness_nm=thickness_nm,
                eps_tensor=n_film**2 * np.eye(3, dtype=np.complex128),
                x_domain_nm=(0.0, period_nm),
            )
        ],
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_incident**2,
        eps_superstrate=n_exit**2,
    )
    S = physical_basis_scattering(
        stack,
        N,
        component_major_physical_rows=use_component_major_physical_rows,
    )
    te = Solver.zero_order_mode_index(N, "TE")
    r = S[0][te, te]
    t = S[2][te, te]
    r_expected, t_expected = normal_incidence_characteristic_rt(
        n_incident,
        n_exit,
        [n_film],
        [thickness_nm],
        wavelength_nm,
    )
    label = "component-major physical rows" if use_component_major_physical_rows else "harmonic-major physical rows"
    print(f"Quarter-wave film, N={N}, {label}")
    print(f"  rcwa     r = {r:.12g}")
    print(f"  rcwa     t = {t:.12g}")
    print(f"  analytic r = {r_expected:.12g}")
    print(f"  analytic t = {t_expected:.12g}")
    print(f"  rcwa     R = {abs(r)**2:.12g}")
    print(f"  analytic R = {abs(r_expected)**2:.12g}")


def main() -> None:
    wavelength_nm = 550.0
    period_nm = 200.0
    stack = Stack(
        layers=[
            Layer.uniform(
                thickness_nm=0.0,
                eps_tensor=np.eye(3, dtype=np.complex128),
                x_domain_nm=(0.0, period_nm),
            )
        ],
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=1.0,
        eps_superstrate=1.5**2,
    )

    print("Checking the isotropic half-space tangential basis against the analytic port basis")
    basis_alignment_report(stack, N=1)
    print()
    quarter_wave_report(N=1, use_component_major_physical_rows=False)
    print()
    quarter_wave_report(N=1, use_component_major_physical_rows=True)


if __name__ == "__main__":
    main()
