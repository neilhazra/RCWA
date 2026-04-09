from __future__ import annotations

import pathlib
import sys

import numpy as jnp
import pytest
import scipy.linalg

from rcwa import Layer, Solver, Stack


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "pyGTM"))
GTM = pytest.importorskip("GTM.GTMcore")


def _fresnel_interface(n1: complex, n2: complex) -> tuple[complex, complex]:
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def _forward_q(eps: complex, kappa_normalized: complex) -> complex:
    q = jnp.sqrt(complex(eps) - complex(kappa_normalized) ** 2)
    if jnp.imag(q) < 0.0 or (abs(jnp.imag(q)) < 1e-14 and jnp.real(q) < 0.0):
        q = -q
    return q


def _physical_port_fields_matrix(stack: Stack, eps: complex, N: int) -> jnp.ndarray:
    num_h = Stack.num_harmonics(N)
    fields = jnp.zeros((4 * num_h, 4 * num_h), dtype=jnp.complex128)

    for h, order in enumerate(Stack.harmonic_orders(N)):
        kappa_normalized = stack.kappa_normalized + order * stack.G_normalized
        q = _forward_q(eps, kappa_normalized)
        block = jnp.stack(
            [
                jnp.array([0.0, -q, 1.0, 0.0], dtype=jnp.complex128),
                jnp.array([eps / q, 0.0, 0.0, -1.0], dtype=jnp.complex128),
                jnp.array([0.0, q, 1.0, 0.0], dtype=jnp.complex128),
                jnp.array([eps / q, 0.0, 0.0, 1.0], dtype=jnp.complex128),
            ],
            axis=1,
        )
        fields[4 * h : 4 * (h + 1), 4 * h : 4 * (h + 1)] = block

    return fields[:, Solver.mode_reorder_indices(N)]


def _zero_order_field_rt(stack: Stack, N: int, pol: str, num_points: int = 256) -> tuple[complex, complex]:
    scattering_modal = Solver.total_scattering_matrix(
        stack,
        N,
        num_points=num_points,
        verbose=False,
    )
    substrate_reduced_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=num_points),
        N,
    )
    superstrate_reduced_fields = Solver.isotropic_mode_fields(
        stack.get_Q_superstrate_normalized(N, num_points=num_points),
        N,
    )
    substrate_tangential_fields = Solver.reduced_to_tangential_fields(
        substrate_reduced_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_substrate,
            N,
        ),
    )
    superstrate_tangential_fields = Solver.reduced_to_tangential_fields(
        superstrate_reduced_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_superstrate,
            N,
        ),
    )
    substrate_physical_fields = _physical_port_fields_matrix(stack, stack.eps_substrate, N)
    superstrate_physical_fields = _physical_port_fields_matrix(stack, stack.eps_superstrate, N)

    scattering_physical = Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(
                substrate_physical_fields,
                substrate_tangential_fields,
            ),
            scattering_modal,
            Solver.basis_change_scattering_matrix(
                superstrate_tangential_fields,
                superstrate_physical_fields,
            ),
        ]
    )

    zero_mode = Solver.zero_order_mode_index(N, pol)
    return scattering_physical[0][zero_mode, zero_mode], scattering_physical[2][zero_mode, zero_mode]


def _normal_incidence_characteristic_rt(
    n_incident: float,
    n_exit: float,
    layer_indices: list[float],
    layer_thicknesses_nm: list[float],
    wavelength_nm: float,
) -> tuple[complex, complex]:
    matrix = jnp.eye(2, dtype=jnp.complex128)

    for n_layer, thickness_nm in zip(layer_indices, layer_thicknesses_nm):
        phase = 2 * jnp.pi * n_layer * thickness_nm / wavelength_nm
        layer_matrix = jnp.array(
            [
                [jnp.cos(phase), 1j * jnp.sin(phase) / n_layer],
                [1j * n_layer * jnp.sin(phase), jnp.cos(phase)],
            ],
            dtype=jnp.complex128,
        )
        matrix = matrix @ layer_matrix

    B = matrix[0, 0] + matrix[0, 1] * n_exit
    C = matrix[1, 0] + matrix[1, 1] * n_exit
    reflection = (n_incident * B - C) / (n_incident * B + C)
    transmission = 2 * n_incident / (n_incident * B + C)
    return reflection, transmission


def _normal_incidence_reflectance_transmittance(
    n_incident: float,
    n_exit: float,
    layer_indices: list[float],
    layer_thicknesses_nm: list[float],
    wavelength_nm: float,
) -> tuple[float, float]:
    reflection, transmission = _normal_incidence_characteristic_rt(
        n_incident,
        n_exit,
        layer_indices,
        layer_thicknesses_nm,
        wavelength_nm,
    )
    reflectance = jnp.abs(reflection) ** 2
    transmittance = jnp.abs(transmission) ** 2 * (n_exit / n_incident)
    return reflectance, transmittance


def _rotate_in_plane_eps(eps_local: jnp.ndarray, theta_rad: float) -> jnp.ndarray:
    c = jnp.cos(theta_rad)
    s = jnp.sin(theta_rad)
    rotation = jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    return rotation @ eps_local @ rotation.T


def _rotated_in_plane_birefringent_eps(no: float, ne: float, theta_rad: float) -> jnp.ndarray:
    eps_local = jnp.diag(jnp.array([no**2, ne**2, 1.0], dtype=jnp.complex128))
    return _rotate_in_plane_eps(eps_local, theta_rad)


def _const_eps_fn(value: complex):
    return lambda frequency_hz, _value=complex(value): _value


def _pygtm_rpp(
    wavelength_nm: float,
    zeta: complex,
    eps_incident: complex,
    eps_exit: complex,
    layers: list[tuple[float, jnp.ndarray, float]],
) -> complex:
    system = GTM.System()
    system.set_superstrate(GTM.Layer(epsilon1=_const_eps_fn(eps_incident)))
    system.set_substrate(GTM.Layer(epsilon1=_const_eps_fn(eps_exit)))

    for thickness_nm, eps_local, angle_deg in layers:
        system.add_layer(
            GTM.Layer(
                thickness=thickness_nm * 1e-9,
                epsilon1=_const_eps_fn(eps_local[0, 0]),
                epsilon2=_const_eps_fn(eps_local[1, 1]),
                epsilon3=_const_eps_fn(eps_local[2, 2]),
                phi=-jnp.deg2rad(angle_deg),
            )
        )

    frequency_hz = GTM.c_const / (wavelength_nm * 1e-9)
    system.initialize_sys(frequency_hz)
    system.calculate_GammaStar(frequency_hz, zeta)
    with jnp.errstate(divide="ignore", invalid="ignore"):
        r_out, _, _, _ = system.calculate_r_t(zeta)
    return complex(r_out[0])


def _physical_port_fields(eps: complex, kappa_normalized: complex) -> jnp.ndarray:
    q = jnp.sqrt(complex(eps) - complex(kappa_normalized) ** 2)
    if jnp.imag(q) < 0.0 or (abs(jnp.imag(q)) < 1e-14 and jnp.real(q) < 0.0):
        q = -q

    return jnp.stack(
        [
            jnp.array([0.0, -q, 1.0, 0.0], dtype=jnp.complex128),
            jnp.array([eps / q, 0.0, 0.0, -1.0], dtype=jnp.complex128),
            jnp.array([0.0, q, 1.0, 0.0], dtype=jnp.complex128),
            jnp.array([eps / q, 0.0, 0.0, 1.0], dtype=jnp.complex128),
        ],
        axis=1,
    )


def _rcwa_reflection_matrix_in_physical_ps_basis(
    stack: Stack,
    num_points: int = 256,
) -> jnp.ndarray:
    scattering_modal = Solver.total_scattering_matrix(
        stack,
        0,
        num_points=num_points,
        verbose=False,
    )
    substrate_reduced_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(0, num_points=num_points),
        0,
    )
    superstrate_reduced_fields = Solver.isotropic_mode_fields(
        stack.get_Q_superstrate_normalized(0, num_points=num_points),
        0,
    )
    substrate_tangential_fields = Solver.reduced_to_tangential_fields(
        substrate_reduced_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_substrate,
            0,
        ),
    )
    superstrate_tangential_fields = Solver.reduced_to_tangential_fields(
        superstrate_reduced_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_superstrate,
            0,
        ),
    )

    scattering_physical = Solver.chain_scattering_matrices(
        [
            Solver.basis_change_scattering_matrix(
                _physical_port_fields(stack.eps_substrate, stack.kappa_normalized),
                substrate_tangential_fields,
            ),
            scattering_modal,
            Solver.basis_change_scattering_matrix(
                superstrate_tangential_fields,
                _physical_port_fields(stack.eps_superstrate, stack.kappa_normalized),
            ),
        ]
    )
    return scattering_physical[0]


def _uniform_anisotropic_layer_q_and_reference_fields(
    N: int,
    num_points: int = 256,
) -> tuple[Stack, jnp.ndarray, jnp.ndarray]:
    wavelength_nm = 633.0
    period_nm = 500.0
    kappa_inv_nm = 0.002
    theta_rad = jnp.deg2rad(27.0)
    eps_local = jnp.diag(jnp.array([2.05**2, 1.52**2, 1.67**2], dtype=jnp.complex128))
    eps_tensor = _rotate_in_plane_eps(eps_local, theta_rad)

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=kappa_inv_nm,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=120.0,
            eps_tensor=eps_tensor,
            x_domain_nm=(0.0, period_nm),
        )
    )

    q_matrix = Solver.component_to_harmonic_major(
        stack.layer_Q_matrix_normalized(0, N, num_points=num_points)
    )
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=num_points),
        N,
    )
    return stack, q_matrix, substrate_fields


def _zero_order_outgoing_electric_components(
    stack: Stack,
    modal_coeffs: jnp.ndarray,
    side: str,
    direction: str,
    num_points: int = 256,
) -> tuple[complex, complex]:
    if side == "substrate":
        q_matrix = stack.get_Q_substrate_normalized(0, num_points=num_points)
        eps = stack.eps_substrate
    elif side == "superstrate":
        q_matrix = stack.get_Q_superstrate_normalized(0, num_points=num_points)
        eps = stack.eps_superstrate
    else:
        raise ValueError(f"Unknown side={side!r}")

    if direction == "forward":
        te_col, tm_col = 0, 1
    elif direction == "backward":
        te_col, tm_col = 2, 3
    else:
        raise ValueError(f"Unknown direction={direction!r}")

    _, evecs = Solver.diagonalize_sort_isotropic_modes(q_matrix)
    zero = Stack.zero_harmonic_index(0)
    te_mode = Solver.zero_order_mode_index(0, "TE")
    tm_mode = Solver.zero_order_mode_index(0, "TM")
    E_y = modal_coeffs[te_mode] * evecs[zero, 2, te_col]
    E_x = modal_coeffs[tm_mode] * evecs[zero, 3, tm_col] / eps
    return E_x, E_y


def _transmitted_superstrate_electric_components_for_te_incidence(
    stack: Stack,
    num_points: int = 256,
) -> tuple[complex, complex]:
    _, transmitted = Solver.reflection_transmission(stack, 0, "TE", num_points=num_points)
    return _zero_order_outgoing_electric_components(
        stack,
        transmitted,
        side="superstrate",
        direction="forward",
        num_points=num_points,
    )


def _assert_modal_propagation_blocks_match_eigenvalue_halves(
    eigenvalues: jnp.ndarray,
    thickness: float,
    atol: float = 1e-10,
) -> None:
    half = eigenvalues.shape[0] // 2
    expected_forward = jnp.diag(jnp.exp(eigenvalues[:half] * thickness))
    expected_backward = jnp.diag(jnp.exp(-eigenvalues[half:] * thickness))
    S11, S12, S21, S22 = Solver.modal_propagation_scattering_matrix(eigenvalues, thickness)

    assert jnp.allclose(S11, 0, atol=atol)
    assert jnp.allclose(S22, 0, atol=atol)
    assert jnp.allclose(S12, expected_backward, atol=atol)
    assert jnp.allclose(S21, expected_forward, atol=atol)


def _explicit_transfer_to_scattering(T: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    half = T.shape[0] // 2
    T11 = T[:half, :half]
    T12 = T[:half, half:]
    T21 = T[half:, :half]
    T22 = T[half:, half:]
    T22_inv = jnp.linalg.inv(T22)
    return (
        -(T22_inv @ T21),
        T22_inv,
        T11 - T12 @ T22_inv @ T21,
        T12 @ T22_inv,
    )


def _explicit_redheffer_star_product(
    Sa: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    Sb: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A11, A12, A21, A22 = Sa
    B11, B12, B21, B22 = Sb
    half = A11.shape[0]
    I = jnp.eye(half, dtype=A11.dtype)

    inv_a = jnp.linalg.solve(I - A22 @ B11, I)
    inv_b = jnp.linalg.solve(I - B11 @ A22, I)
    return (
        A11 + A12 @ inv_b @ B11 @ A21,
        A12 @ inv_b @ B12,
        B21 @ inv_a @ A21,
        B22 + B21 @ inv_a @ A22 @ B12,
    )


def test_reorder_matrix_layout() -> None:
    N = 2
    num_h = Stack.num_harmonics(N)
    P = Solver.reorder_matrix(N)
    vec = jnp.arange(4 * num_h, dtype=jnp.complex128)
    out = P @ vec

    for h in range(num_h):
        for p in range(4):
            assert jnp.isclose(out[4 * h + p], vec[p * num_h + h])


def test_component_to_harmonic_major_matches_reorder_similarity_transform() -> None:
    N = 2
    num_h = Stack.num_harmonics(N)
    matrix = jnp.arange((4 * num_h) ** 2, dtype=jnp.complex128).reshape(4 * num_h, 4 * num_h)
    reorder = Solver.reorder_matrix(N)
    expected = reorder @ matrix @ reorder.T

    assert jnp.array_equal(Solver.component_to_harmonic_major(matrix), expected)


def test_mode_reorder_indices_put_all_forward_modes_before_backward_modes() -> None:
    N = 2
    num_h = Stack.num_harmonics(N)
    indices = Solver.mode_reorder_indices(N)
    expected = jnp.array(
        [4 * h for h in range(num_h)]
        + [4 * h + 1 for h in range(num_h)]
        + [4 * h + 2 for h in range(num_h)]
        + [4 * h + 3 for h in range(num_h)],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(indices, expected)


def test_zero_order_mode_index_matches_forward_te_tm_layout() -> None:
    for N in [0, 1, 2]:
        zero = Stack.zero_harmonic_index(N)
        assert Solver.zero_order_mode_index(N, "TE") == zero
        assert Solver.zero_order_mode_index(N, "TM") == Stack.num_harmonics(N) + zero


def test_diagonalize_sort_isotropic_modes_reconstructs_blocks(uniform_interface_stack: Stack) -> None:
    N = 2
    for Q_iso in [
        uniform_interface_stack.get_Q_substrate_normalized(N),
        uniform_interface_stack.get_Q_superstrate_normalized(N),
    ]:
        evals, evecs = Solver.diagonalize_sort_isotropic_modes(Q_iso)
        diag_blocks = Solver._isotropic_diag_blocks(Q_iso)

        assert evals.shape == (Stack.num_harmonics(N), 4)
        assert evecs.shape == (Stack.num_harmonics(N), 4, 4)

        for h in range(evals.shape[0]):
            Q_recon = evecs[h] @ jnp.diag(evals[h]) @ jnp.linalg.inv(evecs[h])
            assert jnp.allclose(Q_recon, diag_blocks[h], atol=1e-8)


def test_diagonalize_sort_isotropic_modes_accepts_preextracted_harmonic_blocks(
    uniform_interface_stack: Stack,
) -> None:
    N = 2
    Q_iso = uniform_interface_stack.get_Q_substrate_normalized(N)
    diag_blocks = Solver._isotropic_diag_blocks(Q_iso)

    evals, evecs = Solver.diagonalize_sort_isotropic_modes(diag_blocks)

    assert evals.shape == (Stack.num_harmonics(N), 4)
    assert evecs.shape == (Stack.num_harmonics(N), 4, 4)

    for h in range(evals.shape[0]):
        reconstructed = evecs[h] @ jnp.diag(evals[h]) @ jnp.linalg.inv(evecs[h])
        assert jnp.allclose(reconstructed, diag_blocks[h], atol=1e-8)


def test_isotropic_mode_fields_have_global_forward_backward_split(
    uniform_interface_stack: Stack,
) -> None:
    N = 2
    reorder = Solver.reorder_matrix(N)
    q_matrix = uniform_interface_stack.layer_Q_matrix_normalized(0, N)
    q_matrix = reorder @ q_matrix @ reorder.T

    substrate_fields = Solver.isotropic_mode_fields(
        uniform_interface_stack.get_Q_substrate_normalized(N),
        N,
    )
    _, layer_fields = Solver.layer_mode_fields(q_matrix, substrate_fields)
    T = jnp.linalg.solve(layer_fields, substrate_fields)
    half = T.shape[0] // 2

    assert jnp.max(jnp.abs(T[:half, half:])) < 1e-10
    assert jnp.max(jnp.abs(T[half:, :half])) < 1e-10


def test_layer_mode_sort_uses_overlap_to_break_zeroed_direction_metric_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eigenvalues = jnp.array([0.4j, 0.0, 0.0, -0.4j], dtype=jnp.complex128)
    eigenvectors = jnp.eye(4, dtype=jnp.complex128)

    def fake_eig(_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return eigenvalues.copy(), eigenvectors.copy()

    monkeypatch.setattr(
        Solver,
        "_harmonic_diag_blocks_if_block_diagonal",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(scipy.linalg, "eig", fake_eig)

    # In the reference basis, columns 0 and 2 are forward-like while columns 1
    # and 3 are backward-like. The two zero-metric modes therefore need the
    # overlap score to decide their relative order.
    reference_coeffs = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    reference_fields = jnp.linalg.inv(reference_coeffs)

    _, sorted_vectors = Solver.diagonalize_sort_layer_modes(
        jnp.eye(4, dtype=jnp.complex128),
        reference_fields=reference_fields,
        tol=1e-9,
    )

    expected = jnp.eye(4, dtype=jnp.complex128)[:, [0, 2, 1, 3]]
    assert jnp.array_equal(sorted_vectors, expected)


def test_diagonalize_sort_layer_modes_uses_harmonic_block_fast_path_for_uniform_anisotropic_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, q_matrix, substrate_fields = _uniform_anisotropic_layer_q_and_reference_fields(N=2)

    def fail_eig(_: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        raise AssertionError("dense scipy.linalg.eig should not be called for a harmonic block-diagonal layer")

    monkeypatch.setattr(scipy.linalg, "eig", fail_eig)

    eigenvalues, eigenvectors = Solver.diagonalize_sort_layer_modes(
        q_matrix,
        reference_fields=substrate_fields,
    )
    reconstructed = eigenvectors @ jnp.diag(eigenvalues) @ jnp.linalg.inv(eigenvectors)

    assert jnp.allclose(reconstructed, q_matrix, atol=1e-8)


def test_diagonalize_sort_layer_modes_uses_dense_fallback_for_patterned_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack = Stack(
        wavelength_nm=633.0,
        kappa_inv_nm=0.0,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=120.0,
            x_domain_nm=(0.0, 500.0),
            segments=[
                (0.0, 180.0, 2.4 * jnp.eye(3, dtype=jnp.complex128)),
                (180.0, 500.0, 1.2 * jnp.eye(3, dtype=jnp.complex128)),
            ],
        )
    )

    N = 2
    num_points = 256
    q_matrix = Solver.component_to_harmonic_major(
        stack.layer_Q_matrix_normalized(0, N, num_points=num_points)
    )
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=num_points),
        N,
    )

    original_eig = scipy.linalg.eig
    calls = {"count": 0}

    def tracking_eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        calls["count"] += 1
        return original_eig(matrix)

    monkeypatch.setattr(scipy.linalg, "eig", tracking_eig)

    eigenvalues, eigenvectors = Solver.diagonalize_sort_layer_modes(
        q_matrix,
        reference_fields=substrate_fields,
    )
    reconstructed = eigenvectors @ jnp.diag(eigenvalues) @ jnp.linalg.inv(eigenvectors)

    assert calls["count"] == 1
    assert jnp.allclose(reconstructed, q_matrix, atol=1e-8)


def test_uniform_anisotropic_layer_total_scattering_matches_dense_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack, _, _ = _uniform_anisotropic_layer_q_and_reference_fields(N=2)

    fast = Solver.total_scattering_matrix(stack, 2, num_points=256)

    monkeypatch.setattr(
        Solver,
        "_harmonic_diag_blocks_if_block_diagonal",
        lambda *_args, **_kwargs: None,
    )
    dense = Solver.total_scattering_matrix(stack, 2, num_points=256)

    for fast_block, dense_block in zip(fast, dense):
        assert jnp.allclose(fast_block, dense_block, atol=1e-10)


def test_modes_to_fields_matrix_reorders_columns_into_solver_modal_layout() -> None:
    evecs = jnp.arange(3 * 4 * 4, dtype=jnp.complex128).reshape(3, 4, 4)
    fields = Solver.modes_to_fields_matrix(evecs)
    block = scipy.linalg.block_diag(*evecs)
    expected = block[:, Solver.mode_reorder_indices(1)]
    assert jnp.array_equal(fields, expected)


def test_basis_change_scattering_matrix_is_identity_for_identical_bases(
    uniform_interface_stack: Stack,
) -> None:
    N = 2
    fields = Solver.isotropic_mode_fields(uniform_interface_stack.get_Q_substrate_normalized(N), N)
    S11, S12, S21, S22 = Solver.basis_change_scattering_matrix(fields, fields)
    eye = jnp.eye(S12.shape[0], dtype=S12.dtype)

    assert jnp.allclose(S11, 0, atol=1e-12)
    assert jnp.allclose(S22, 0, atol=1e-12)
    assert jnp.allclose(S12, eye, atol=1e-12)
    assert jnp.allclose(S21, eye, atol=1e-12)


def test_transfer_to_scattering_identity_is_identity() -> None:
    T = jnp.eye(6, dtype=jnp.complex128)
    S11, S12, S21, S22 = Solver.transfer_to_scattering(T)
    eye = jnp.eye(3, dtype=jnp.complex128)

    assert jnp.allclose(S11, 0, atol=1e-12)
    assert jnp.allclose(S22, 0, atol=1e-12)
    assert jnp.allclose(S12, eye, atol=1e-12)
    assert jnp.allclose(S21, eye, atol=1e-12)


def test_transfer_to_scattering_matches_explicit_block_formula_for_generic_transfer_matrix() -> None:
    T11 = jnp.array(
        [[1.3 + 0.2j, -0.4 + 0.1j], [0.15 - 0.3j, 0.8 + 0.05j]],
        dtype=jnp.complex128,
    )
    T12 = jnp.array(
        [[0.2 - 0.1j, 0.05 + 0.03j], [-0.1 + 0.2j, 0.4 - 0.15j]],
        dtype=jnp.complex128,
    )
    T21 = jnp.array(
        [[-0.3 + 0.2j, 0.07 - 0.05j], [0.12 + 0.09j, -0.18 + 0.04j]],
        dtype=jnp.complex128,
    )
    T22 = jnp.array(
        [[1.6 - 0.4j, 0.2 + 0.1j], [-0.25 + 0.05j, 1.2 + 0.3j]],
        dtype=jnp.complex128,
    )
    T = jnp.block([[T11, T12], [T21, T22]])

    expected = _explicit_transfer_to_scattering(T)
    actual = Solver.transfer_to_scattering(T)

    for expected_block, actual_block in zip(expected, actual):
        assert jnp.allclose(actual_block, expected_block, atol=1e-12)


def test_modal_propagation_scattering_matrix_is_reflectionless() -> None:
    eigenvalues = jnp.array([0.2j, 0.5j, -0.2j, -0.5j], dtype=jnp.complex128)
    thickness = 1.7
    S11, S12, S21, S22 = Solver.modal_propagation_scattering_matrix(eigenvalues, thickness)
    expected_forward = jnp.diag(jnp.exp(eigenvalues[:2] * thickness))
    expected_backward = jnp.diag(jnp.exp(-eigenvalues[2:] * thickness))

    assert jnp.allclose(S11, 0, atol=1e-12)
    assert jnp.allclose(S22, 0, atol=1e-12)
    assert jnp.allclose(expected_forward, expected_backward, atol=1e-12)
    assert jnp.allclose(S12, expected_backward, atol=1e-12)
    assert jnp.allclose(S21, expected_forward, atol=1e-12)


def test_modal_propagation_scattering_matrix_uses_isotropic_forward_and_backward_halves_directly(
    uniform_interface_stack: Stack,
) -> None:
    N = 0
    thickness = 0.73
    eigenvalues, _ = Solver.diagonalize_sort_isotropic_modes(
        uniform_interface_stack.get_Q_substrate_normalized(N)
    )
    zero_harmonic_eigenvalues = eigenvalues[Stack.zero_harmonic_index(N)]
    _assert_modal_propagation_blocks_match_eigenvalue_halves(
        zero_harmonic_eigenvalues,
        thickness,
        atol=1e-12,
    )


@pytest.mark.parametrize("N", [1, 2])
def test_modal_propagation_scattering_matrix_uses_anisotropic_forward_and_backward_halves_directly(
    N: int,
) -> None:
    wavelength_nm = 633.0
    period_nm = 500.0
    kappa_inv_nm = 0.002
    theta_rad = jnp.deg2rad(27.0)
    c = jnp.cos(theta_rad)
    s = jnp.sin(theta_rad)
    rotation = jnp.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=jnp.complex128,
    )
    eps_local = jnp.diag(jnp.array([2.05**2, 1.52**2, 1.67**2], dtype=jnp.complex128))
    eps_tensor = rotation @ eps_local @ rotation.T

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=kappa_inv_nm,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=120.0,
            eps_tensor=eps_tensor,
            x_domain_nm=(0.0, period_nm),
        )
    )

    reorder = Solver.reorder_matrix(N)
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=256),
        N,
    )
    q_matrix = reorder @ stack.layer_Q_matrix_normalized(0, N, num_points=256) @ reorder.T
    eigenvalues, _ = Solver.layer_mode_fields(q_matrix, substrate_fields)

    _assert_modal_propagation_blocks_match_eigenvalue_halves(
        eigenvalues,
        stack.thickness_normalized(0),
        atol=1e-10,
    )


@pytest.mark.parametrize("N", [1, 2])
def test_modal_propagation_scattering_matrix_uses_lossy_anisotropic_forward_and_backward_halves_directly(
    N: int,
) -> None:
    wavelength_nm = 610.0
    period_nm = 480.0
    kappa_inv_nm = 0.0018
    theta_rad = jnp.deg2rad(31.0)
    eps_local = jnp.diag(
        jnp.array(
            [
                (1.82 + 0.04j) ** 2,
                (1.46 + 0.08j) ** 2,
                (1.61 + 0.03j) ** 2,
            ],
            dtype=jnp.complex128,
        )
    )
    eps_tensor = _rotate_in_plane_eps(eps_local, theta_rad)

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=kappa_inv_nm,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=115.0,
            eps_tensor=eps_tensor,
            x_domain_nm=(0.0, period_nm),
        )
    )

    reorder = Solver.reorder_matrix(N)
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=256),
        N,
    )
    q_matrix = reorder @ stack.layer_Q_matrix_normalized(0, N, num_points=256) @ reorder.T
    eigenvalues, _ = Solver.layer_mode_fields(q_matrix, substrate_fields)

    _assert_modal_propagation_blocks_match_eigenvalue_halves(
        eigenvalues,
        stack.thickness_normalized(0),
        atol=1e-10,
    )


def test_redheffer_star_product_multiplies_reflectionless_transmissions() -> None:
    X1 = jnp.diag(jnp.array([0.8 + 0.1j, 0.7 - 0.2j], dtype=jnp.complex128))
    X2 = jnp.diag(jnp.array([0.9 - 0.05j, 0.6 + 0.15j], dtype=jnp.complex128))
    Sa = (jnp.zeros_like(X1), X1, X1, jnp.zeros_like(X1))
    Sb = (jnp.zeros_like(X2), X2, X2, jnp.zeros_like(X2))

    S11, S12, S21, S22 = Solver.redheffer_star_product(Sa, Sb)
    expected = X1 @ X2

    assert jnp.allclose(S11, 0, atol=1e-12)
    assert jnp.allclose(S22, 0, atol=1e-12)
    assert jnp.allclose(S12, expected, atol=1e-12)
    assert jnp.allclose(S21, expected, atol=1e-12)


def test_redheffer_star_product_matches_explicit_two_solve_formula_for_generic_blocks() -> None:
    Sa = (
        jnp.array([[0.12 + 0.04j, -0.03 + 0.01j], [0.02 - 0.05j, 0.15 + 0.02j]], dtype=jnp.complex128),
        jnp.array([[0.81 - 0.02j, 0.05 + 0.03j], [-0.04 + 0.01j, 0.78 + 0.06j]], dtype=jnp.complex128),
        jnp.array([[0.76 + 0.01j, -0.02 + 0.04j], [0.03 - 0.01j, 0.83 - 0.02j]], dtype=jnp.complex128),
        jnp.array([[0.08 + 0.03j, 0.01 - 0.02j], [-0.02 + 0.01j, 0.09 + 0.04j]], dtype=jnp.complex128),
    )
    Sb = (
        jnp.array([[0.11 - 0.02j, 0.04 + 0.01j], [-0.01 + 0.02j, 0.07 - 0.03j]], dtype=jnp.complex128),
        jnp.array([[0.79 + 0.05j, -0.03 + 0.02j], [0.02 + 0.01j, 0.75 - 0.04j]], dtype=jnp.complex128),
        jnp.array([[0.82 - 0.01j, 0.01 + 0.03j], [-0.02 + 0.02j, 0.77 + 0.05j]], dtype=jnp.complex128),
        jnp.array([[0.06 + 0.02j, -0.01 + 0.01j], [0.02 - 0.03j, 0.10 + 0.01j]], dtype=jnp.complex128),
    )

    expected = _explicit_redheffer_star_product(Sa, Sb)
    actual = Solver.redheffer_star_product(Sa, Sb)

    for expected_block, actual_block in zip(expected, actual):
        assert jnp.allclose(actual_block, expected_block, atol=1e-12)


def test_uniform_interface_total_scattering_is_finite_and_diagonal(
    uniform_interface_stack: Stack,
) -> None:
    for N in [0, 1, 2]:
        S11, _, S21, _ = Solver.total_scattering_matrix(uniform_interface_stack, N, num_points=256)
        assert jnp.all(jnp.isfinite(S11))
        assert jnp.all(jnp.isfinite(S21))
        assert jnp.max(jnp.abs(S11 - jnp.diag(jnp.diag(S11)))) < 1e-10
        assert jnp.max(jnp.abs(S21 - jnp.diag(jnp.diag(S21)))) < 1e-10


def test_verbose_total_scattering_matrix_emits_progress_messages(
    uniform_interface_stack: Stack,
    capsys: pytest.CaptureFixture[str],
) -> None:
    Solver.total_scattering_matrix(uniform_interface_stack, 1, num_points=128, verbose=True)
    captured = capsys.readouterr()

    assert "[Solver] Building total scattering matrix" in captured.out
    assert "[Solver] Diagonalizing isotropic substrate modes" in captured.out
    assert "block-diagonal harmonic layer Q" in captured.out
    assert "[Solver] Concatenating" in captured.out


def test_reflection_transmission_only_populates_zero_order_for_uniform_interface(
    uniform_interface_stack: Stack,
) -> None:
    for N in [0, 1, 2]:
        for pol in ["TE", "TM"]:
            r, t = Solver.reflection_transmission(
                uniform_interface_stack,
                N,
                incident_pol=pol,
                num_points=256,
            )
            zero_mode = Solver.zero_order_mode_index(N, pol)
            mask = jnp.ones_like(r, dtype=bool)
            mask[zero_mode] = False

            assert jnp.allclose(r[mask], 0, atol=1e-10)
            assert jnp.allclose(t[mask], 0, atol=1e-10)


def test_uniform_interface_matches_fresnel_coefficients(
    uniform_interface_stack: Stack,
) -> None:
    n_sub = jnp.sqrt(uniform_interface_stack.eps_substrate)
    n_sup = jnp.sqrt(uniform_interface_stack.eps_superstrate)
    r_te_expected, t_te_expected = _fresnel_interface(n_sub, n_sup)
    r_tm_expected = -r_te_expected
    t_tm_expected = t_te_expected

    for N in [0, 1, 2]:
        r0_te, t0_te = _zero_order_field_rt(uniform_interface_stack, N, "TE")
        r0_tm, t0_tm = _zero_order_field_rt(uniform_interface_stack, N, "TM")

        assert jnp.isclose(r0_te, r_te_expected, atol=1e-10)
        assert jnp.isclose(t0_te, t_te_expected, atol=1e-10)
        assert jnp.isclose(r0_tm, r_tm_expected, atol=1e-10)
        assert jnp.isclose(t0_tm, t_tm_expected, atol=1e-10)


def test_uniform_interface_conserves_energy(uniform_interface_stack: Stack) -> None:
    n_sub = jnp.sqrt(uniform_interface_stack.eps_substrate)
    n_sup = jnp.sqrt(uniform_interface_stack.eps_superstrate)

    for N in [0, 1, 2]:
        r0_te, t0_te = _zero_order_field_rt(uniform_interface_stack, N, "TE")
        r0_tm, t0_tm = _zero_order_field_rt(uniform_interface_stack, N, "TM")

        R_te = jnp.abs(r0_te) ** 2
        T_te = jnp.abs(t0_te) ** 2 * (n_sup / n_sub)
        R_tm = jnp.abs(r0_tm) ** 2
        T_tm = jnp.abs(t0_tm) ** 2 * (n_sup / n_sub)

        assert jnp.isclose(R_te + T_te, 1.0, atol=1e-10)
        assert jnp.isclose(R_tm + T_tm, 1.0, atol=1e-10)


def test_multilayer_uniform_stack_total_scattering_is_finite() -> None:
    stack = Stack(
        wavelength_nm=405.0,
        kappa_inv_nm=0.0,
        eps_substrate=1.4696**2,
        eps_superstrate=1.0,
    )
    period_nm = 500.0
    stack.add_layer(Layer.uniform(50.0, 1.4696**2 * jnp.eye(3), x_domain_nm=(0.0, period_nm)))
    stack.add_layer(Layer.uniform(20.0, 2.0 * jnp.eye(3), x_domain_nm=(0.0, period_nm)))
    stack.add_layer(Layer.uniform(10.0, 1.0 * jnp.eye(3), x_domain_nm=(0.0, period_nm)))

    S = Solver.total_scattering_matrix(stack, 1, num_points=256)
    for block in S:
        assert jnp.all(jnp.isfinite(block))


def test_single_isotropic_quarter_wave_film_matches_analytic_reflectance() -> None:
    wavelength_nm = 550.0
    n_incident = 1.0
    n_film = 2.0
    n_exit = 1.5
    thickness_nm = wavelength_nm / (4 * n_film)

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_incident**2,
        eps_superstrate=n_exit**2,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm,
            n_film**2 * jnp.eye(3),
            x_domain_nm=(0.0, 500.0),
        )
    )

    reflectance_expected, transmittance_expected = _normal_incidence_reflectance_transmittance(
        n_incident,
        n_exit,
        [n_film],
        [thickness_nm],
        wavelength_nm,
    )

    for pol in ["TE", "TM"]:
        r0, t0 = _zero_order_field_rt(stack, 0, pol)
        reflectance = jnp.abs(r0) ** 2
        transmittance = (
            jnp.abs(t0) ** 2 * (n_exit / n_incident)
            if pol == "TE"
            else jnp.abs(t0) ** 2 * (n_exit / n_incident)
        )

        assert jnp.isclose(reflectance, reflectance_expected, atol=1e-10)
        assert jnp.isclose(transmittance, transmittance_expected, atol=1e-10)


def test_dielectric_mirror_reflectance_matches_analytic_stack_and_grows_with_periods() -> None:
    wavelength_nm = 550.0
    n_incident = 1.0
    n_exit = 1.5
    n_high = 2.2
    n_low = 1.45
    thickness_high_nm = wavelength_nm / (4 * n_high)
    thickness_low_nm = wavelength_nm / (4 * n_low)

    reflectances: list[float] = []
    for periods in [1, 4]:
        stack = Stack(
            wavelength_nm=wavelength_nm,
            kappa_inv_nm=0.0,
            eps_substrate=n_incident**2,
            eps_superstrate=n_exit**2,
        )
        layer_indices: list[float] = []
        layer_thicknesses_nm: list[float] = []

        for _ in range(periods):
            stack.add_layer(
                Layer.uniform(
                    thickness_high_nm,
                    n_high**2 * jnp.eye(3),
                    x_domain_nm=(0.0, 500.0),
                )
            )
            stack.add_layer(
                Layer.uniform(
                    thickness_low_nm,
                    n_low**2 * jnp.eye(3),
                    x_domain_nm=(0.0, 500.0),
                )
            )
            layer_indices.extend([n_high, n_low])
            layer_thicknesses_nm.extend([thickness_high_nm, thickness_low_nm])

        reflectance_expected, transmittance_expected = _normal_incidence_reflectance_transmittance(
            n_incident,
            n_exit,
            layer_indices,
            layer_thicknesses_nm,
            wavelength_nm,
        )

        r0_te, t0_te = _zero_order_field_rt(stack, 0, "TE")
        r0_tm, t0_tm = _zero_order_field_rt(stack, 0, "TM")

        reflectance_te = jnp.abs(r0_te) ** 2
        reflectance_tm = jnp.abs(r0_tm) ** 2
        transmittance_te = jnp.abs(t0_te) ** 2 * (n_exit / n_incident)
        transmittance_tm = jnp.abs(t0_tm) ** 2 * (n_exit / n_incident)

        assert jnp.isclose(reflectance_te, reflectance_expected, atol=1e-10)
        assert jnp.isclose(reflectance_tm, reflectance_expected, atol=1e-10)
        assert jnp.isclose(transmittance_te, transmittance_expected, atol=1e-10)
        assert jnp.isclose(transmittance_tm, transmittance_expected, atol=1e-10)

        reflectances.append(float(reflectance_te))

    assert reflectances[1] > reflectances[0]
    assert reflectances[1] > 0.9


def test_rotated_quarter_wave_plate_mixes_polarization_when_axis_turns() -> None:
    wavelength_nm = 633.0
    n_ordinary = 1.5
    n_extraordinary = 1.6
    n_ambient = jnp.sqrt(n_ordinary * n_extraordinary)
    thickness_nm = wavelength_nm / (4 * (n_extraordinary - n_ordinary))

    transmitted_cross_components: list[float] = []
    transmitted_phase_shifts: list[float] = []

    for degrees in [0.0, 22.5, 45.0]:
        theta_rad = jnp.deg2rad(degrees)
        stack = Stack(
            wavelength_nm=wavelength_nm,
            kappa_inv_nm=0.0,
            eps_substrate=n_ambient**2,
            eps_superstrate=n_ambient**2,
        )
        stack.add_layer(
            Layer.uniform(
                thickness_nm,
                _rotated_in_plane_birefringent_eps(n_ordinary, n_extraordinary, theta_rad),
                x_domain_nm=(0.0, 500.0),
            )
        )

        E_x, E_y = _transmitted_superstrate_electric_components_for_te_incidence(stack)
        phase = jnp.angle(E_x) - jnp.angle(E_y)
        phase = (phase + jnp.pi) % (2 * jnp.pi) - jnp.pi

        transmitted_cross_components.append(float(jnp.abs(E_x)))
        transmitted_phase_shifts.append(float(phase))

    assert transmitted_cross_components[0] < 1e-10
    assert transmitted_cross_components[0] < transmitted_cross_components[1] < transmitted_cross_components[2]
    assert jnp.isclose(transmitted_cross_components[2], 0.3833762304577446, atol=5e-3)
    assert jnp.isclose(transmitted_phase_shifts[2], -jnp.pi / 2, atol=5e-2)


@pytest.mark.parametrize(("pol", "cross_pol"), [("TE", "TM"), ("TM", "TE")])
def test_axis_aligned_lossy_anisotropic_film_preserves_polarization_and_is_passive(
    pol: str,
    cross_pol: str,
) -> None:
    wavelength_nm = 532.0
    n_ambient = 1.27
    thickness_nm = 140.0
    eps_tensor = jnp.diag(
        jnp.array(
            [
                (1.83 + 0.07j) ** 2,
                (1.56 + 0.03j) ** 2,
                (1.67 + 0.05j) ** 2,
            ],
            dtype=jnp.complex128,
        )
    )

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm,
            eps_tensor,
            x_domain_nm=(0.0, 500.0),
        )
    )

    reflected, transmitted = Solver.reflection_transmission(stack, 0, pol, num_points=256)
    cross_mode = Solver.zero_order_mode_index(0, cross_pol)
    reflected_E_x, reflected_E_y = _zero_order_outgoing_electric_components(
        stack,
        reflected,
        side="substrate",
        direction="backward",
    )
    transmitted_E_x, transmitted_E_y = _zero_order_outgoing_electric_components(
        stack,
        transmitted,
        side="superstrate",
        direction="forward",
    )
    reflectance = jnp.abs(reflected_E_x) ** 2 + jnp.abs(reflected_E_y) ** 2
    transmittance = jnp.abs(transmitted_E_x) ** 2 + jnp.abs(transmitted_E_y) ** 2

    assert jnp.abs(reflected[cross_mode]) < 1e-12
    assert jnp.abs(transmitted[cross_mode]) < 1e-12
    assert jnp.isfinite(reflectance)
    assert jnp.isfinite(transmittance)
    assert reflectance + transmittance < 0.98


def test_ninety_degree_rotation_of_lossy_anisotropic_film_swaps_te_tm_responses() -> None:
    wavelength_nm = 532.0
    n_ambient = 1.27
    thickness_nm = 140.0
    eps_local = jnp.diag(
        jnp.array(
            [
                (1.83 + 0.07j) ** 2,
                (1.56 + 0.03j) ** 2,
                (1.67 + 0.05j) ** 2,
            ],
            dtype=jnp.complex128,
        )
    )

    stack_unrotated = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack_unrotated.add_layer(
        Layer.uniform(
            thickness_nm,
            _rotate_in_plane_eps(eps_local, 0.0),
            x_domain_nm=(0.0, 500.0),
        )
    )

    stack_rotated = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack_rotated.add_layer(
        Layer.uniform(
            thickness_nm,
            _rotate_in_plane_eps(eps_local, jnp.pi / 2),
            x_domain_nm=(0.0, 500.0),
        )
    )

    r_te_unrot, t_te_unrot = _zero_order_field_rt(stack_unrotated, 0, "TE")
    r_tm_unrot, t_tm_unrot = _zero_order_field_rt(stack_unrotated, 0, "TM")
    r_te_rot, t_te_rot = _zero_order_field_rt(stack_rotated, 0, "TE")
    r_tm_rot, t_tm_rot = _zero_order_field_rt(stack_rotated, 0, "TM")

    assert jnp.isclose(t_te_rot, t_tm_unrot, atol=1e-10)
    assert jnp.isclose(t_tm_rot, t_te_unrot, atol=1e-10)
    assert jnp.isclose(jnp.abs(r_te_rot) ** 2, jnp.abs(r_tm_unrot) ** 2, atol=1e-10)
    assert jnp.isclose(jnp.abs(r_tm_rot) ** 2, jnp.abs(r_te_unrot) ** 2, atol=1e-10)


def test_rotated_lossy_anisotropic_film_mixes_polarization_and_absorbs_power() -> None:
    wavelength_nm = 633.0
    n_ambient = 1.35
    thickness_nm = 180.0
    eps_local = jnp.diag(
        jnp.array(
            [
                (1.50 + 0.01j) ** 2,
                (1.72 + 0.05j) ** 2,
                (1.62 + 0.03j) ** 2,
            ],
            dtype=jnp.complex128,
        )
    )

    stack_aligned = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack_aligned.add_layer(
        Layer.uniform(
            thickness_nm,
            eps_local,
            x_domain_nm=(0.0, 500.0),
        )
    )

    stack_rotated = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack_rotated.add_layer(
        Layer.uniform(
            thickness_nm,
            _rotate_in_plane_eps(eps_local, jnp.deg2rad(33.0)),
            x_domain_nm=(0.0, 500.0),
        )
    )

    _, transmitted_aligned = Solver.reflection_transmission(
        stack_aligned,
        0,
        "TE",
        num_points=256,
    )
    reflected_rotated, transmitted_rotated = Solver.reflection_transmission(
        stack_rotated,
        0,
        "TE",
        num_points=256,
    )

    transmitted_aligned_E_x, _ = _zero_order_outgoing_electric_components(
        stack_aligned,
        transmitted_aligned,
        side="superstrate",
        direction="forward",
    )
    reflected_rotated_E_x, reflected_rotated_E_y = _zero_order_outgoing_electric_components(
        stack_rotated,
        reflected_rotated,
        side="substrate",
        direction="backward",
    )
    transmitted_rotated_E_x, transmitted_rotated_E_y = _zero_order_outgoing_electric_components(
        stack_rotated,
        transmitted_rotated,
        side="superstrate",
        direction="forward",
    )
    reflectance = jnp.abs(reflected_rotated_E_x) ** 2 + jnp.abs(reflected_rotated_E_y) ** 2
    transmittance = jnp.abs(transmitted_rotated_E_x) ** 2 + jnp.abs(transmitted_rotated_E_y) ** 2

    assert jnp.abs(transmitted_aligned_E_x) < 1e-12
    assert jnp.abs(transmitted_rotated_E_x) > 5e-2
    assert reflectance + transmittance < 0.9


def test_oblique_axis_aligned_anisotropic_film_rpp_matches_pygtm() -> None:
    wavelength_nm = 633.0
    zeta = jnp.sin(jnp.deg2rad(32.0))
    eps_incident = 1.0
    eps_exit = 1.7**2
    eps_local = jnp.diag(
        jnp.array(
            [
                (1.60 + 0.02j) ** 2,
                (2.05 + 0.03j) ** 2,
                (1.80 + 0.01j) ** 2,
            ],
            dtype=jnp.complex128,
        )
    )

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=zeta * 2 * jnp.pi / wavelength_nm,
        eps_substrate=eps_incident,
        eps_superstrate=eps_exit,
    )
    stack.add_layer(
        Layer.uniform(
            120.0,
            eps_local,
            x_domain_nm=(0.0, 500.0),
        )
    )

    reflection_matrix_rcwa = _rcwa_reflection_matrix_in_physical_ps_basis(stack)
    rpp_rcwa = reflection_matrix_rcwa[1, 1]
    rpp_pygtm = _pygtm_rpp(
        wavelength_nm,
        zeta,
        eps_incident,
        eps_exit,
        [(120.0, eps_local, 0.0)],
    )

    assert jnp.isclose(rpp_rcwa, rpp_pygtm, atol=1e-10)


@pytest.mark.parametrize("angle_deg", [0.0, 27.0])
def test_oblique_nbocl2_hbn_stack_rpp_matches_pygtm(angle_deg: float) -> None:
    wavelength_nm = 780.0
    zeta = 1.4
    eps_incident = 1.0
    eps_exit = 3.674**2
    eps_nbocl2_local = jnp.diag(
        jnp.array([4.0, 6.0, 1.6**2], dtype=jnp.complex128)
    )
    eps_hbn = (1.4531**2) * jnp.eye(3, dtype=jnp.complex128)

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=zeta * 2 * jnp.pi / wavelength_nm,
        eps_substrate=eps_incident,
        eps_superstrate=eps_exit,
    )
    stack.add_layer(
        Layer.uniform(
            122.0,
            _rotate_in_plane_eps(eps_nbocl2_local, jnp.deg2rad(angle_deg)),
            x_domain_nm=(0.0, 500.0),
        )
    )
    stack.add_layer(
        Layer.uniform(
            90.0,
            eps_hbn,
            x_domain_nm=(0.0, 500.0),
        )
    )

    reflection_matrix_rcwa = _rcwa_reflection_matrix_in_physical_ps_basis(stack)
    rpp_rcwa = reflection_matrix_rcwa[1, 1]
    rpp_pygtm = _pygtm_rpp(
        wavelength_nm,
        zeta,
        eps_incident,
        eps_exit,
        [
            (122.0, eps_nbocl2_local, angle_deg),
            (90.0, eps_hbn, 0.0),
        ],
    )

    assert jnp.isclose(rpp_rcwa, rpp_pygtm, atol=1e-10)


def test_quarter_wave_plate_at_45_degrees_produces_nearly_equal_transmitted_field_components() -> None:
    wavelength_nm = 633.0
    n_ordinary = 1.5
    n_extraordinary = 1.6
    n_ambient = jnp.sqrt(n_ordinary * n_extraordinary)
    thickness_nm = wavelength_nm / (4 * (n_extraordinary - n_ordinary))

    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_ambient**2,
        eps_superstrate=n_ambient**2,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm,
            _rotated_in_plane_birefringent_eps(
                n_ordinary,
                n_extraordinary,
                jnp.deg2rad(45.0),
            ),
            x_domain_nm=(0.0, 500.0),
        )
    )

    reflected, _ = Solver.reflection_transmission(stack, 0, "TE", num_points=256)
    E_x, E_y = _transmitted_superstrate_electric_components_for_te_incidence(stack)

    assert float(jnp.sum(jnp.abs(reflected) ** 2)) < 5e-3
    assert jnp.isclose(jnp.abs(E_x), jnp.abs(E_y), rtol=2e-2, atol=2e-3)


def test_piecewise_stack_response_is_finite(sample_problem: dict) -> None:
    stack = sample_problem["stack"]
    for pol in ["TE", "TM"]:
        r, t = Solver.reflection_transmission(stack, 1, incident_pol=pol, num_points=256)
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(t))
