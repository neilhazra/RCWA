from __future__ import annotations

from dataclasses import dataclass, field

import numpy as jnp

from . import _config  # noqa: F401
from .layer import Layer


@dataclass
class Stack:
    """A stack of x-periodic layers between isotropic substrate and superstrate.

    This class mainly orchestrates the construction of layer and half-space
    RCWA operators. In the current rewrite, the layer-level operator is a full
    matrix in the component-major basis

        [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T

    so any Q matrix returned by this file has shape

        (4 * (2N + 1), 4 * (2N + 1)).
    """

    layers: list[Layer] = field(default_factory=list)
    wavelength_nm: float = 0.0
    kappa_inv_nm: float = 0.0
    eps_substrate: complex = 1.0
    eps_superstrate: complex = 1.0
    _uniform_q_cache: dict[tuple[complex, int, int], jnp.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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
        self._uniform_q_cache.clear()

    def layer_Q_matrix_normalized(
        self, layer_index: int, N: int, num_points: int = 512
    ) -> jnp.ndarray:
        """Build one layer's full normalized Q matrix.

        Inputs:
            layer_index:
                Integer index into ``self.layers``.
            N:
                Fourier truncation order, so the harmonic orders are
                ``n = -N, ..., N`` and the number of harmonics is
                ``num_h = 2N + 1``.
            num_points:
                Number of real-space sample points used to compute the Fourier
                coefficients of the material functions before those coefficients
                are assembled into Toeplitz convolution matrices.

        Output:
            A dense complex matrix with shape ``(4 * num_h, 4 * num_h)``.

        Contents of the output matrix:
            This is the layer propagation operator in the component-major basis

                [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

            The matrix is built from:
            - Toeplitz convolution matrices of shape ``(num_h, num_h)`` for the
              reduced material quantities, and
            - the diagonal harmonic wavevector matrix
              ``K_x = diag(kappa_normalized + n * G_normalized)``.

            Each of the 16 logical blocks in the 4x4 operator layout is itself
            a ``(num_h, num_h)`` matrix acting on one field component across all
            retained harmonics.
        """
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
        """Compatibility wrapper that currently returns the full Q matrix.

        Historically this method returned a tensor with shape
        ``(num_h, num_h, 4, 4)`` whose ``(n, m)`` entry was the 4x4 block
        coupling harmonic ``m`` into harmonic ``n``.

        In the rewrite, the layer code directly constructs the flattened full
        matrix instead, so this method now returns the same object as
        :meth:`layer_Q_matrix_normalized`:

            shape ``(4 * num_h, 4 * num_h)``

        in the component-major basis

            [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.
        """
        return self.layer_Q_matrix_normalized(layer_index, N, num_points=num_points)

    def build_all_Q_matrices_normalized(self, N: int, num_points: int = 512) -> list[jnp.ndarray]:
        """Build the full normalized Q matrix for every physical layer.

        Input:
            N sets ``num_h = 2N + 1`` retained Fourier harmonics.

        Output:
            A Python list with ``len(self.layers)`` entries.

            Each entry is a complex matrix with shape
            ``(4 * num_h, 4 * num_h)`` in the basis

                [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

            There is no intermediate tensor representation in this path.
        """
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
        """Build the full normalized Q matrix for a uniform isotropic medium.

        Input:
            eps:
                Scalar dielectric constant for an isotropic medium. It is
                converted to the 3x3 tensor ``eps * I`` before entering the
                layer-level material pipeline.
            N:
                Fourier truncation order giving ``num_h = 2N + 1`` harmonics.

        Output:
            A dense complex matrix with shape ``(4 * num_h, 4 * num_h)`` in the
            same component-major basis used for layer Q matrices:

                [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

        Structure of the output:
            Because the medium is uniform, every Toeplitz material matrix is
            diagonal and the resulting full Q matrix is decoupled
            harmonic-by-harmonic. After regrouping rows and columns by
            harmonic, it consists of ``num_h`` independent 4x4 blocks.
        """
        cache_key = (complex(eps), N, num_points)
        if cache_key not in self._uniform_q_cache:
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
            self._uniform_q_cache[cache_key] = Layer.build_Q_matrix_normalized(
                self.harmonic_orders(N),
                self.harmonic_orders(N),
                self.kappa_normalized,
                self.G_normalized,
                toeplitz_matrices,
                N,
            )
        return self._uniform_q_cache[cache_key]

    def get_Q_substrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        """Return the substrate half-space Q matrix.

        Output:
            A complex matrix with shape ``(4 * num_h, 4 * num_h)``, where
            ``num_h = 2N + 1``. The basis is

                [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

            Since the substrate is uniform and isotropic, this matrix is
            harmonic-decoupled even though it is stored in the full flattened
            form.
        """
        return self._build_uniform_medium_Q_normalized(
            self.eps_substrate,
            N,
            num_points=num_points,
        )

    def get_Q_superstrate_normalized(self, N: int, num_points: int = 512) -> jnp.ndarray:
        """Return the superstrate half-space Q matrix.

        Output:
            A complex matrix with shape ``(4 * num_h, 4 * num_h)``, where
            ``num_h = 2N + 1``. The basis is

                [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

            Since the superstrate is uniform and isotropic, this matrix is
            harmonic-decoupled even though it is stored in the full flattened
            form.
        """
        return self._build_uniform_medium_Q_normalized(
            self.eps_superstrate,
            N,
            num_points=num_points,
        )
