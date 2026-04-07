from __future__ import annotations

import jax
import jax.numpy as jnp

from rcwa import Layer, Solver, Stack


def _fresnel_interface(n1: complex, n2: complex) -> tuple[complex, complex]:
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def _zero_order_field_rt(stack: Stack, N: int, pol: str, num_points: int = 256) -> tuple[complex, complex]:
    r, t = Solver.reflection_transmission(stack, N, incident_pol=pol, num_points=num_points)
    Q_sub = stack.get_Q_substrate_normalized(N, num_points=num_points)
    Q_sup = stack.get_Q_superstrate_normalized(N, num_points=num_points)
    _, evecs_sub = Solver.diagonalize_sort_isotropic_modes(Q_sub)
    _, evecs_sup = Solver.diagonalize_sort_isotropic_modes(Q_sup)
    h = Stack.zero_harmonic_index(N)

    if pol == "TE":
        E_sub_fwd = evecs_sub[h, 2, 0]
        E_sub_bwd = evecs_sub[h, 2, 2]
        E_sup_fwd = evecs_sup[h, 2, 0]
        zero_mode = Solver.zero_order_mode_index(N, "TE")
    else:
        E_sub_fwd = evecs_sub[h, 3, 1]
        E_sub_bwd = evecs_sub[h, 3, 3]
        E_sup_fwd = evecs_sup[h, 3, 1]
        zero_mode = Solver.zero_order_mode_index(N, "TM")

    r0 = r[zero_mode] * (E_sub_bwd / E_sub_fwd)
    t0 = t[zero_mode] * (E_sup_fwd / E_sub_fwd)
    return r0, t0


def test_reorder_matrix_layout() -> None:
    N = 2
    num_h = Stack.num_harmonics(N)
    P = Solver.reorder_matrix(N)
    vec = jnp.arange(4 * num_h, dtype=jnp.complex128)
    out = P @ vec

    for h in range(num_h):
        for p in range(4):
            assert jnp.isclose(out[4 * h + p], vec[p * num_h + h])


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


def test_modes_to_fields_matrix_reorders_columns_into_solver_modal_layout() -> None:
    evecs = jnp.arange(3 * 4 * 4, dtype=jnp.complex128).reshape(3, 4, 4)
    fields = Solver.modes_to_fields_matrix(evecs)
    block = jax.scipy.linalg.block_diag(*evecs)
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


def test_modal_propagation_scattering_matrix_is_reflectionless() -> None:
    eigenvalues = jnp.array([0.2j, 0.5j, -0.2j, -0.5j], dtype=jnp.complex128)
    thickness = 1.7
    S11, S12, S21, S22 = Solver.modal_propagation_scattering_matrix(eigenvalues, thickness)
    expected = jnp.diag(jnp.exp(eigenvalues[:2] * thickness))

    assert jnp.allclose(S11, 0, atol=1e-12)
    assert jnp.allclose(S22, 0, atol=1e-12)
    assert jnp.allclose(S12, expected, atol=1e-12)
    assert jnp.allclose(S21, expected, atol=1e-12)


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


def test_uniform_interface_total_scattering_is_finite_and_diagonal(
    uniform_interface_stack: Stack,
) -> None:
    for N in [0, 1, 2]:
        S11, _, S21, _ = Solver.total_scattering_matrix(uniform_interface_stack, N, num_points=256)
        assert jnp.all(jnp.isfinite(S11))
        assert jnp.all(jnp.isfinite(S21))
        assert jnp.max(jnp.abs(S11 - jnp.diag(jnp.diag(S11)))) < 1e-10
        assert jnp.max(jnp.abs(S21 - jnp.diag(jnp.diag(S21)))) < 1e-10


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
            mask = jnp.ones_like(r, dtype=bool).at[zero_mode].set(False)

            assert jnp.allclose(r[mask], 0, atol=1e-10)
            assert jnp.allclose(t[mask], 0, atol=1e-10)


def test_uniform_interface_matches_fresnel_coefficients(
    uniform_interface_stack: Stack,
) -> None:
    n_sub = jnp.sqrt(uniform_interface_stack.eps_substrate)
    n_sup = jnp.sqrt(uniform_interface_stack.eps_superstrate)
    r_te_expected, t_te_expected = _fresnel_interface(n_sub, n_sup)
    r_tm_expected = -r_te_expected
    t_tm_expected = 2 * n_sup / (n_sub + n_sup)

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
        T_tm = jnp.abs(t0_tm) ** 2 * (n_sub / n_sup)

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


def test_piecewise_stack_response_is_finite(sample_problem: dict) -> None:
    stack = sample_problem["stack"]
    for pol in ["TE", "TM"]:
        r, t = Solver.reflection_transmission(stack, 1, incident_pol=pol, num_points=256)
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(t))
