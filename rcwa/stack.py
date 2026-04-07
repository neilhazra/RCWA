from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from . import _config  # noqa: F401
from .layer import Layer


@dataclass
class Stack:
    """A stack of x-periodic layers between isotropic substrate and superstrate."""

    layers: list[Layer] = field(default_factory=list)
    wavelength_nm: float = 0.0
    kappa_inv_nm: float = 0.0
    eps_substrate: complex = 1.0
    eps_superstrate: complex = 1.0

    @property
    def period_nm(self) -> float:
        if not self.layers:
            raise ValueError("Stack has no layers")
        return self.layers[0].period_nm

    @property
    def G_normalized(self) -> float:
        return self.wavelength_nm / self.period_nm

    @property
    def kappa_normalized(self) -> float:
        return self.kappa_inv_nm * self.wavelength_nm / (2 * jnp.pi)

    def thickness_normalized(self, layer_index: int) -> float:
        return 2 * jnp.pi * self.layers[layer_index].thickness_nm / self.wavelength_nm

    @staticmethod
    def num_harmonics(N: int) -> int:
        return 2 * N + 1

    @staticmethod
    def harmonic_orders(N: int) -> jnp.ndarray:
        return jnp.arange(-N, N + 1)

    @staticmethod
    def zero_harmonic_index(N: int) -> int:
        return N

    def add_layer(self, layer: Layer) -> None:
        if self.layers and abs(layer.period_nm - self.period_nm) > 1e-10:
            raise ValueError(
                f"Layer period {layer.period_nm} nm does not match stack period {self.period_nm} nm"
            )
        self.layers.append(layer)

    def layer_Q_tensor_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        orders = self.harmonic_orders(N)
        coeffs = self.layers[layer_index].fourier_coefficients(N, num_points=num_points)
        return Layer.build_Q_tensor_normalized(
            orders,
            orders,
            self.kappa_normalized,
            self.G_normalized,
            coeffs,
            N,
        )

    def build_all_Q_matrices_normalized(self, N: int, num_points: int = 512) -> list[jnp.ndarray]:
        return [
            self.Q_tensor_to_matrix(self.layer_Q_tensor_normalized(i, N, num_points=num_points))
            for i in range(len(self.layers))
        ]

    def _build_Q_isotropic_normalized(self, eps: complex, N: int) -> jnp.ndarray:
        eps = jnp.asarray(eps, dtype=jnp.complex128)
        num_harmonics = self.num_harmonics(N)
        q_tensor = jnp.zeros((num_harmonics, num_harmonics, 4, 4), dtype=jnp.complex128)

        for idx, n in enumerate(self.harmonic_orders(N)):
            k_normalized = self.kappa_normalized + n * self.G_normalized
            block = jnp.array(
                [
                    [0, 0, 0, 1j * (-eps)],
                    [0, 0, 1j * (-eps + k_normalized**2), 0],
                    [0, -1j, 0, 0],
                    [1j * (-1 + k_normalized**2 / eps), 0, 0, 0],
                ],
                dtype=jnp.complex128,
            )
            q_tensor = q_tensor.at[idx, idx].set(block)

        return q_tensor

    def get_Q_substrate_normalized(self, N: int) -> jnp.ndarray:
        return self._build_Q_isotropic_normalized(self.eps_substrate, N)

    def get_Q_superstrate_normalized(self, N: int) -> jnp.ndarray:
        return self._build_Q_isotropic_normalized(self.eps_superstrate, N)

    @staticmethod
    @jax.jit
    def diagonalize_sort_isotropic_modes(Q_iso: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize isotropic half-space blocks harmonic-by-harmonic."""
        diag_blocks = jnp.diagonal(Q_iso, axis1=0, axis2=1)
        diag_blocks = jnp.moveaxis(diag_blocks, -1, 0)

        def _resolve_pair(v1: jnp.ndarray, v2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            w1 = v2[3] * v1 - v1[3] * v2
            w1_norm = jnp.linalg.norm(w1)
            w1 = jnp.where(w1_norm > 1e-14, w1 / w1_norm, v1)

            w2 = v2[2] * v1 - v1[2] * v2
            w2_norm = jnp.linalg.norm(w2)
            w2 = jnp.where(w2_norm > 1e-14, w2 / w2_norm, v2)
            return w1, w2

        def _eig_sort_single(block: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            eigenvalues, eigenvectors = jnp.linalg.eig(block)
            tol = 1e-8
            im_rounded = jnp.round(jnp.imag(eigenvalues) / tol) * tol
            re_rounded = jnp.round(jnp.real(eigenvalues) / tol) * tol
            sort_idx = jnp.lexsort((-re_rounded, -im_rounded))
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            w1_fwd, w2_fwd = _resolve_pair(eigenvectors[:, 0], eigenvectors[:, 1])
            w1_bwd, w2_bwd = _resolve_pair(eigenvectors[:, 2], eigenvectors[:, 3])
            eigenvectors = jnp.column_stack([w1_fwd, w2_fwd, w1_bwd, w2_bwd])
            return eigenvalues, eigenvectors

        return jax.vmap(_eig_sort_single)(diag_blocks)
