from __future__ import annotations

import numpy as jnp
import pytest
import scipy.linalg

from rcwa import Layer, Solver, Stack


def _reconstruct_from_centered_coeffs(
    coeffs: jnp.ndarray,
    x_nm: jnp.ndarray,
    x_domain_nm: tuple[float, float],
) -> jnp.ndarray:
    x_min_nm, x_max_nm = x_domain_nm
    period_nm = x_max_nm - x_min_nm
    max_order = (coeffs.shape[0] - 1) // 2
    orders = jnp.arange(-max_order, max_order + 1)
    phase = 2j * jnp.pi * (x_nm[:, None] - x_min_nm) * orders[None, :] / period_nm
    return jnp.sum(coeffs[None, :] * jnp.exp(phase), axis=1)


def _smooth_diagonal_layer_with_known_spectrum() -> Layer:
    x_domain_nm = (0.0, 100.0)
    period_nm = x_domain_nm[1] - x_domain_nm[0]

    def eps_fn(x_nm: jnp.ndarray) -> jnp.ndarray:
        theta = 2 * jnp.pi * x_nm / period_nm
        eps_xx = (
            2.5
            + 0.2 * jnp.cos(theta)
            + 0.12 * jnp.sin(3 * theta)
            + 0.08 * jnp.cos(5 * theta)
        )
        eps_yy = 1.8 + 0.05 * jnp.cos(2 * theta)
        eps_zz = 2.2 + 0.03 * jnp.sin(theta)
        zeros = jnp.zeros_like(theta)

        return jnp.stack(
            [
                jnp.stack([eps_xx, zeros, zeros], axis=-1),
                jnp.stack([zeros, eps_yy, zeros], axis=-1),
                jnp.stack([zeros, zeros, eps_zz], axis=-1),
            ],
            axis=-2,
        )

    return Layer(thickness_nm=10.0, x_domain_nm=x_domain_nm, eps_fn=eps_fn)


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


def test_uniform_layer_toeplitz_matrices_are_diagonal_with_constant_entry() -> None:
    layer = Layer.uniform(
        thickness_nm=50.0,
        eps_tensor=jnp.array(
            [
                [4.0, 0.3, 0.2],
                [0.1, 3.0, 0.25],
                [0.05, 0.15, 2.5],
            ],
            dtype=jnp.complex128,
        ),
        x_domain_nm=(0.0, 100.0),
    )

    N = 2
    toeplitz = layer.build_toeplitz_fourier_matrices(N, num_points=128)
    quantities = layer.field_quantities(num_points=128)
    eye = jnp.eye(2 * N + 1, dtype=jnp.complex128)

    for key, values in quantities.items():
        expected = values[0] * eye
        assert toeplitz[key].shape == (2 * N + 1, 2 * N + 1)
        assert jnp.allclose(toeplitz[key], expected, atol=1e-12)


def test_toeplitz_matrices_match_manual_harmonic_difference_indexing() -> None:
    layer = Layer.piecewise(
        thickness_nm=80.0,
        x_domain_nm=(0.0, 120.0),
        segments=[
            (
                0.0,
                45.0,
                jnp.array(
                    [
                        [4.0, 0.3, 0.2],
                        [0.1, 3.0, 0.25],
                        [0.05, 0.15, 2.5],
                    ],
                    dtype=jnp.complex128,
                ),
            ),
            (
                45.0,
                120.0,
                jnp.array(
                    [
                        [2.2, -0.4, 0.35],
                        [0.2, 1.8, -0.1],
                        [-0.25, 0.12, 3.1],
                    ],
                    dtype=jnp.complex128,
                ),
            ),
        ],
    )

    N = 2
    toeplitz = layer.build_toeplitz_fourier_matrices(N, num_points=256)
    coeffs = layer.fourier_coefficients(N, num_points=256)
    orders = jnp.arange(-N, N + 1)

    for key, coeff_array in coeffs.items():
        expected = jnp.array(
            [[coeff_array[int(n - m + 2 * N)] for m in orders] for n in orders],
            dtype=jnp.complex128,
        )
        assert toeplitz[key].shape == (2 * N + 1, 2 * N + 1)
        assert jnp.allclose(toeplitz[key], expected, atol=1e-12)


def test_truncated_fourier_reconstruction_error_decreases_as_more_harmonics_are_kept() -> None:
    layer = _smooth_diagonal_layer_with_known_spectrum()
    num_points = 512
    x_nm = layer.sample_points(num_points)
    reference = layer.field_quantities(num_points=num_points)["hat_eps_xx"]

    errors = []
    for N in [1, 2, 3]:
        coeffs = layer.fourier_coefficients(N, num_points=num_points)["hat_eps_xx"]
        reconstruction = _reconstruct_from_centered_coeffs(coeffs, x_nm, layer.x_domain_nm)
        rms_error = jnp.sqrt(jnp.mean(jnp.abs(reconstruction - reference) ** 2))
        errors.append(float(rms_error))

    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-12


def test_stack_rejects_mismatched_periods() -> None:
    stack = Stack(wavelength_nm=405.0, kappa_inv_nm=0.0, eps_substrate=1.0, eps_superstrate=1.0)
    stack.add_layer(Layer.uniform(10.0, 2.0 * jnp.eye(3), x_domain_nm=(0.0, 100.0)))

    with pytest.raises(ValueError, match="does not match stack period"):
        stack.add_layer(Layer.uniform(10.0, 2.0 * jnp.eye(3), x_domain_nm=(0.0, 120.0)))


def test_stack_normalized_parameters_follow_geometry(uniform_interface_stack: Stack) -> None:
    stack = uniform_interface_stack
    assert jnp.isclose(stack.period_nm, 500.0)
    assert jnp.isclose(stack.G_normalized, stack.wavelength_nm / stack.period_nm)
    assert jnp.isclose(stack.kappa_normalized, 0.0)
    assert jnp.isclose(stack.thickness_normalized(0), 0.0)


def test_uniform_q_matrix_reorders_to_harmonic_block_diagonal_form(
    uniform_interface_stack: Stack,
) -> None:
    N = 2
    reorder = Solver.reorder_matrix(N)
    Q_component_major = uniform_interface_stack.get_Q_substrate_normalized(N)
    Q_harmonic_major = reorder @ Q_component_major @ reorder.T
    diag_blocks = Solver._isotropic_diag_blocks(Q_component_major)
    expected = scipy.linalg.block_diag(*diag_blocks)

    assert Q_component_major.shape == (4 * Stack.num_harmonics(N), 4 * Stack.num_harmonics(N))
    assert jnp.allclose(Q_harmonic_major, expected, atol=1e-12)


def test_layer_q_harmonic_major_builder_matches_component_major_reorder() -> None:
    layer = _smooth_diagonal_layer_with_known_spectrum()
    stack = Stack(wavelength_nm=633.0, kappa_inv_nm=0.07, eps_substrate=1.0, eps_superstrate=1.0)
    stack.add_layer(layer)
    N = 2

    q_component_major = stack.layer_Q_matrix_normalized(0, N, num_points=256)
    q_harmonic_major = stack.layer_Q_matrix_harmonic_major_normalized(0, N, num_points=256)

    assert jnp.allclose(
        q_harmonic_major,
        Solver.component_to_harmonic_major(q_component_major),
        atol=1e-12,
    )


def test_layer_tangential_transform_harmonic_major_matches_component_major_reorder() -> None:
    layer = _smooth_diagonal_layer_with_known_spectrum()
    N = 2
    toeplitz = layer.build_toeplitz_fourier_matrices(N, num_points=256)
    transform_component_major = Layer.build_reduced_to_tangential_field_transform_component_major(
        toeplitz,
        N,
    )
    transform_harmonic_major = Layer.build_reduced_to_tangential_field_transform_harmonic_major(
        toeplitz,
        N,
    )

    assert jnp.allclose(
        transform_harmonic_major,
        Solver.component_to_harmonic_major(transform_component_major),
        atol=1e-12,
    )


def test_isotropic_q_matrix_matches_closed_form_uniform_medium_expression(
    uniform_interface_stack: Stack,
) -> None:
    N = 2
    num_h = Stack.num_harmonics(N)
    eye = jnp.eye(num_h, dtype=jnp.complex128)
    zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)

    for eps, q_matrix in [
        (
            uniform_interface_stack.eps_substrate,
            uniform_interface_stack.get_Q_substrate_normalized(N),
        ),
        (
            uniform_interface_stack.eps_superstrate,
            uniform_interface_stack.get_Q_superstrate_normalized(N),
        ),
    ]:
        K_x = Layer.build_K_x_diag_matrix(
            uniform_interface_stack.kappa_normalized,
            uniform_interface_stack.G_normalized,
            N,
        )
        K_x_squared = K_x @ K_x
        expected_coupling = 1j * (K_x_squared - eps * eye)

        expected = jnp.block(
            [
                [zero, zero, zero, -1j * eye],
                [zero, zero, expected_coupling, zero],
                [zero, -1j * eye, zero, zero],
                [expected_coupling, zero, zero, zero],
            ]
        )

        assert jnp.allclose(q_matrix, expected, atol=1e-12)
