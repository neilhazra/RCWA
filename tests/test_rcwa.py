from __future__ import annotations

import jax.numpy as jnp

from rcwa import Layer, Solver, Stack
from scripts.grating_coupler_convergence import get_field_rt


def _diag_blocks(Q_iso: jnp.ndarray) -> jnp.ndarray:
    diag_blocks = jnp.diagonal(Q_iso, axis1=0, axis2=1)
    return jnp.moveaxis(diag_blocks, -1, 0)


def _fresnel_interface(n1: complex, n2: complex) -> tuple[complex, complex]:
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def _uniform_interface_stack(n_sub: float = 1.4696, n_sup: float = 1.0) -> Stack:
    stack = Stack(
        wavelength_nm=405.0,
        kappa_inv_nm=0.0,
        eps_substrate=n_sub**2,
        eps_superstrate=n_sup**2,
    )
    stack.add_layer(Layer.uniform(0.0, n_sub**2 * jnp.eye(3), x_domain_nm=(0.0, 500.0)))
    return stack


def test_layer_eps_folds_periodically() -> None:
    layer = Layer.piecewise(
        thickness_nm=10.0,
        x_domain_nm=(0.0, 100.0),
        segments=[
            (0.0, 40.0, 2.0 * jnp.eye(3)),
            (40.0, 100.0, 3.0 * jnp.eye(3)),
        ],
    )
    x = jnp.array([5.0, 25.0, 75.0])
    assert jnp.allclose(layer.eps(x), layer.eps(x + layer.period_nm))


def test_zero_harmonic_index_even_and_odd_N() -> None:
    for N in [2, 3]:
        orders = Stack.harmonic_orders(N)
        h0 = Stack.zero_harmonic_index(N)
        assert Stack.num_harmonics(N) == 2 * N + 1
        assert orders[h0] == 0
        assert Solver.zero_order_mode_index(N, "TE") == h0
        assert Solver.zero_order_mode_index(N, "TM") == Stack.num_harmonics(N) + h0


def test_layer_q_tensor_center_block_uses_zero_harmonic_for_even_and_odd_N() -> None:
    stack = _uniform_interface_stack()
    for N in [2, 3]:
        layer = stack.layers[0]
        coeffs = layer.fourier_coefficients(N)
        orders = Stack.harmonic_orders(N)
        Q = Layer.build_Q_tensor_normalized(
            orders,
            orders,
            stack.kappa_normalized,
            stack.G_normalized,
            coeffs,
            N,
        )
        h0 = Stack.zero_harmonic_index(N)
        Q00 = Layer.build_4x4_Q_normalized_single(
            jnp.array(0),
            jnp.array(0),
            stack.kappa_normalized,
            stack.G_normalized,
            coeffs,
            N,
        )
        assert jnp.allclose(Q[h0, h0], Q00, atol=1e-12)


def test_reorder_matrix_layout() -> None:
    N = 3
    num_h = Stack.num_harmonics(N)
    P = Solver.reorder_matrix(N)
    vec = jnp.arange(4 * num_h, dtype=jnp.complex128)
    out = P @ vec
    for h in range(num_h):
        for p in range(4):
            assert jnp.isclose(out[4 * h + p], vec[p * num_h + h])


def test_diagonalize_sort_isotropic_modes_reconstructs_blocks(sample_problem: dict) -> None:
    stack = sample_problem["stack"]
    N = 3
    for Q_iso in [stack.get_Q_substrate_normalized(N), stack.get_Q_superstrate_normalized(N)]:
        evals, evecs = Stack.diagonalize_sort_isotropic_modes(Q_iso)
        diag_blocks = _diag_blocks(Q_iso)
        assert evals.shape == (Stack.num_harmonics(N), 4)
        assert evecs.shape == (Stack.num_harmonics(N), 4, 4)

        max_recon_err = 0.0
        for h in range(evals.shape[0]):
            Q_recon = evecs[h] @ jnp.diag(evals[h]) @ jnp.linalg.inv(evecs[h])
            err = jnp.max(jnp.abs(Q_recon - diag_blocks[h]))
            max_recon_err = max(max_recon_err, float(err))
        assert max_recon_err < 1e-5


def test_basis_change_scattering_matrix_is_identity_for_identical_bases(sample_problem: dict) -> None:
    stack = sample_problem["stack"]
    fields = Solver.isotropic_mode_fields(stack.get_Q_substrate_normalized(2), 2)
    S11, S12, S21, S22 = Solver.basis_change_scattering_matrix(fields, fields)
    eye = jnp.eye(S12.shape[0], dtype=S12.dtype)
    assert jnp.allclose(S11, 0, atol=1e-10)
    assert jnp.allclose(S22, 0, atol=1e-10)
    assert jnp.allclose(S12, eye, atol=1e-10)
    assert jnp.allclose(S21, eye, atol=1e-10)


def test_modal_sort_uses_decaying_forward_modes_for_grating_coupler() -> None:
    n_wg = 2.0
    n_sub = 1.5
    n_sup = 1.0
    period_nm = 370.0
    d_wg_nm = 200.0
    duty_cycle = 0.5
    design_wl = 633.0
    N = 8

    stack = Stack(
        wavelength_nm=design_wl,
        kappa_inv_nm=0.0,
        eps_substrate=n_sub**2,
        eps_superstrate=n_sup**2,
    )
    fill_width = duty_cycle * period_nm
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=d_wg_nm,
            x_domain_nm=(0.0, period_nm),
            segments=[
                (0.0, fill_width, n_wg**2 * jnp.eye(3)),
                (fill_width, period_nm, n_sup**2 * jnp.eye(3)),
            ],
        )
    )

    substrate_fields = Solver.isotropic_mode_fields(stack.get_Q_substrate_normalized(N), N)
    q_matrix = stack.build_all_Q_matrices_normalized(N)[0]
    eigenvalues, _ = Solver.diagonalize_sort_layer_modes(q_matrix, reference_fields=substrate_fields)
    half = eigenvalues.shape[0] // 2
    growth = jnp.abs(jnp.exp(eigenvalues[:half] * stack.thickness_normalized(0)))
    assert float(jnp.max(growth)) <= 1.0 + 1e-8


def test_uniform_interface_has_only_zero_order_response_for_even_and_odd_N() -> None:
    stack = _uniform_interface_stack()
    for N in [2, 3]:
        num_h = Stack.num_harmonics(N)
        h0_te = Solver.zero_order_mode_index(N, "TE")
        h0_tm = Solver.zero_order_mode_index(N, "TM")
        for pol, h0 in [("TE", h0_te), ("TM", h0_tm)]:
            r, t = Solver.reflection_transmission(stack, N, incident_pol=pol)
            mask = jnp.ones_like(r, dtype=bool).at[h0].set(False)
            assert jnp.allclose(r[mask], 0, atol=1e-10)
            assert jnp.allclose(t[:h0], 0, atol=1e-10)
            assert jnp.allclose(t[h0 + 1 :], 0, atol=1e-10)
            assert r.shape == (2 * num_h,)
            assert t.shape == (2 * num_h,)


def test_get_field_rt_matches_fresnel_interface_for_even_and_odd_N() -> None:
    n_sub = 1.4696
    n_sup = 1.0
    stack = _uniform_interface_stack(n_sub=n_sub, n_sup=n_sup)
    r_expected, t_expected = _fresnel_interface(n_sub, n_sup)

    for N in [2, 3]:
        for pol in ["TE", "TM"]:
            r0, t0 = get_field_rt(stack, N, pol)
            assert jnp.isclose(r0, r_expected, atol=1e-10)
            assert jnp.isclose(t0, t_expected, atol=1e-10)


def test_modal_uniform_interface_energy_is_conserved_for_even_and_odd_N() -> None:
    n_sub = 1.4696
    n_sup = 1.0
    stack = _uniform_interface_stack(n_sub=n_sub, n_sup=n_sup)

    for N in [2, 3]:
        for pol in ["TE", "TM"]:
            r0, t0 = get_field_rt(stack, N, pol)
            R = jnp.abs(r0) ** 2
            T = jnp.abs(t0) ** 2 * (n_sup / n_sub)
            assert jnp.isclose(R + T, 1.0, atol=1e-10)


def test_modal_stack_response_is_finite(sample_problem: dict) -> None:
    stack = sample_problem["stack"]
    for pol in ["TE", "TM"]:
        r, t = Solver.reflection_transmission(stack, 3, incident_pol=pol)
        assert jnp.all(jnp.isfinite(r))
        assert jnp.all(jnp.isfinite(t))
