from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from rcwa import Layer, Solver, Stack


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
    expected = jax.scipy.linalg.block_diag(*diag_blocks)

    assert Q_component_major.shape == (4 * Stack.num_harmonics(N), 4 * Stack.num_harmonics(N))
    assert jnp.allclose(Q_harmonic_major, expected, atol=1e-12)
