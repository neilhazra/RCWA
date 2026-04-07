from __future__ import annotations

import jax.numpy as jnp

from rcwa import Layer


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
            [
                [coeff_array[int(n - m + 2 * N)] for m in orders]
                for n in orders
            ],
            dtype=jnp.complex128,
        )
        assert toeplitz[key].shape == (2 * N + 1, 2 * N + 1)
        assert jnp.allclose(toeplitz[key], expected, atol=1e-12)
