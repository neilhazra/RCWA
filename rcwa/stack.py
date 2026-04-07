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

    def layer_Q_matrix_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Build the full layer Q matrix directly from Toeplitz material operators."""
        toeplitz_matrices = self.layers[layer_index].build_toeplitz_fourier_matrices(
            N,
            num_points=num_points,
        )
        return Layer.build_Q_matrix_normalized(
            self.harmonic_orders(N),
            self.harmonic_orders(N),
            self.kappa_normalized,
            self.G_normalized,
            toeplitz_matrices,
            N,
        )

    def layer_Q_tensor_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Compatibility wrapper while the rest of the rewrite is in progress."""
        return self.layer_Q_matrix_normalized(layer_index, N, num_points=num_points)

    def build_all_Q_matrices_normalized(self, N: int, num_points: int = 512) -> list[jnp.ndarray]:
        """Build each layer's full Q matrix without an intermediate tensor reshape."""
        return [
            self.layer_Q_matrix_normalized(i, N, num_points=num_points)
            for i in range(len(self.layers))
        ]

    def _build_uniform_medium_Q_normalized(
        self,
        eps: complex,
        N: int,
        num_points: int = 512,
    ) -> jnp.ndarray:
        """Build a uniform-medium Q matrix through the same layer-level Q pipeline."""
        x_domain_nm = self.layers[0].x_domain_nm if self.layers else (0.0, 1.0)
        uniform_layer = Layer.uniform(
            thickness_nm=0.0,
            eps_tensor=jnp.asarray(eps, dtype=jnp.complex128) * jnp.eye(3, dtype=jnp.complex128),
            x_domain_nm=x_domain_nm,
        )
        toeplitz_matrices = uniform_layer.build_toeplitz_fourier_matrices(
            N,
            num_points=num_points,
        )
        return Layer.build_Q_matrix_normalized(
            self.harmonic_orders(N),
            self.harmonic_orders(N),
            self.kappa_normalized,
            self.G_normalized,
            toeplitz_matrices,
            N,
        )

    def get_Q_substrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        return self._build_uniform_medium_Q_normalized(
            self.eps_substrate,
            N,
            num_points=num_points,
        )

    def get_Q_superstrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        return self._build_uniform_medium_Q_normalized(
            self.eps_superstrate,
            N,
            num_points=num_points,
        )

    @staticmethod
    def _isotropic_diag_blocks(Q_iso: jnp.ndarray) -> jnp.ndarray:
        """Extract per-harmonic 4x4 isotropic blocks from either Q layout.

        The original implementation used a tensor with shape
        ``(num_h, num_h, 4, 4)`` and only the diagonal ``(h, h)`` blocks were
        populated for isotropic media.

        The rewritten code builds a full matrix with shape
        ``(4 * num_h, 4 * num_h)`` in component-major ordering:

            [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)].

        For a uniform isotropic medium this full matrix is still decoupled
        harmonic-by-harmonic. Regrouping the component-major indices by
        harmonic recovers the same independent 4x4 blocks.
        """
        if Q_iso.ndim == 4:
            diag_blocks = jnp.diagonal(Q_iso, axis1=0, axis2=1)
            return jnp.moveaxis(diag_blocks, -1, 0)

        if Q_iso.ndim == 2:
            if Q_iso.shape[0] != Q_iso.shape[1]:
                raise ValueError(f"Expected a square Q matrix, got shape={Q_iso.shape}.")
            if Q_iso.shape[0] % 4 != 0:
                raise ValueError(
                    "Expected a full isotropic Q matrix with size divisible by 4, "
                    f"got shape={Q_iso.shape}."
                )

            num_h = Q_iso.shape[0] // 4

            # Reshape the component-major matrix into explicit harmonic/component
            # block indices:
            #   Q[h_row, comp_row, h_col, comp_col].
            q_blocks = Q_iso.reshape(4, num_h, 4, num_h).transpose(1, 0, 3, 2)
            diag_blocks = jnp.diagonal(q_blocks, axis1=0, axis2=2)
            return jnp.moveaxis(diag_blocks, -1, 0)

        raise ValueError(
            "Expected isotropic Q data with ndim 2 or 4, "
            f"got ndim={Q_iso.ndim} and shape={Q_iso.shape}."
        )

    @staticmethod
    @jax.jit
    def diagonalize_sort_isotropic_modes(Q_iso: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize isotropic half-space modes harmonic-by-harmonic.

        Accepts either the original diagonal-block tensor layout or the rewritten
        full matrix layout and reduces both to per-harmonic 4x4 blocks before
        diagonalization.
        """
        diag_blocks = Stack._isotropic_diag_blocks(Q_iso)

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
    

