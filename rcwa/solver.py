from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as jnp
import scipy.linalg

from . import _config  # noqa: F401
from .layer import Layer
from .stack import Stack


ScatteringMatrix = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


@dataclass
class StackSolveData:
    """Shared stack decomposition reused by the solver and visualization code."""

    substrate_fields: jnp.ndarray
    superstrate_fields: jnp.ndarray
    substrate_continuity_fields: jnp.ndarray
    superstrate_continuity_fields: jnp.ndarray
    layer_modes: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]
    total_scattering: ScatteringMatrix


class Solver:
    """Modal RCWA solver using adjacent-port scattering matrices."""

    @staticmethod
    def _log(verbose: bool, message: str) -> None:
        """Print a solver progress message when verbose output is enabled."""
        if verbose:
            print(f"[Solver] {message}")

    @staticmethod
    @lru_cache(maxsize=None)
    def reorder_matrix(N: int) -> jnp.ndarray:
        """Map component-major field ordering into harmonic-major ordering."""
        num_h = Stack.num_harmonics(N)
        size = 4 * num_h
        src = jnp.array([p * num_h + h for h in range(num_h) for p in range(4)])
        return jnp.eye(size, dtype=jnp.complex128)[src]

    @staticmethod
    @lru_cache(maxsize=None)
    def mode_reorder_indices(N: int) -> jnp.ndarray:
        """Return the global modal column order [fwd, bwd] from per-harmonic blocks.

        The isotropic single-harmonic eigensolver produces columns in each 4x4
        block as

            [forward_TE, forward_TM, backward_TE, backward_TM].

        The stack solver, however, assumes a global modal basis whose first half
        contains every forward mode and whose second half contains every
        backward mode. Within each half we keep the historical ordering

            [TE(-N..N), TM(-N..N)].
        """
        num_h = Stack.num_harmonics(N)
        return jnp.array(
            [4 * h + 0 for h in range(num_h)]
            + [4 * h + 1 for h in range(num_h)]
            + [4 * h + 2 for h in range(num_h)]
            + [4 * h + 3 for h in range(num_h)],
            dtype=jnp.int32,
        )

    @staticmethod
    def component_to_harmonic_major(matrix: jnp.ndarray) -> jnp.ndarray:
        """Reorder a full 4*num_h square matrix from component-major to harmonic-major."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] % 4 != 0:
            raise ValueError(f"Expected a square matrix with size divisible by 4, got {matrix.shape}.")

        num_h = matrix.shape[0] // 4
        return matrix.reshape(4, num_h, 4, num_h).transpose(1, 0, 3, 2).reshape(matrix.shape)

    @staticmethod
    def reduced_to_tangential_fields(
        reduced_fields: jnp.ndarray,
        reduced_to_tangential_component_major: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert modal fields from the reduced D-field basis to tangential fields.

        Internal propagation continues to use the reduced basis

            [-H_y, H_x, E_y, D_x]

        because that basis is needed for the Fourier-factorized layer operator.
        Only the zero-thickness interface basis changes use tangential
        continuity, so before matching adjacent media we convert rows to

            [-H_y, H_x, E_y, E_x].
        """
        transform_harmonic_major = Solver.component_to_harmonic_major(
            reduced_to_tangential_component_major
        )
        return transform_harmonic_major @ reduced_fields

    @staticmethod
    def reduced_to_tangential_fields_harmonic_major(
        reduced_fields: jnp.ndarray,
        reduced_to_tangential_harmonic_major: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert modal fields to tangential fields with a harmonic-major transform."""
        return reduced_to_tangential_harmonic_major @ reduced_fields

    @staticmethod
    def isotropic_reduced_to_tangential_transform_component_major(
        eps: complex,
        N: int,
    ) -> jnp.ndarray:
        """Return the isotropic reduced-to-tangential interface map in component-major order."""
        num_h = Stack.num_harmonics(N)
        identity = jnp.eye(num_h, dtype=jnp.complex128)
        zero = jnp.zeros((num_h, num_h), dtype=jnp.complex128)
        inv_eps = identity / jnp.asarray(eps, dtype=jnp.complex128)

        return jnp.block(
            [
                [identity, zero, zero, zero],
                [zero, identity, zero, zero],
                [zero, zero, identity, zero],
                [zero, zero, zero, inv_eps],
            ]
        )

    @staticmethod
    def isotropic_reduced_to_tangential_transform_harmonic_major(
        eps: complex,
        N: int,
    ) -> jnp.ndarray:
        """Return the isotropic reduced-to-tangential interface map in harmonic-major order."""
        block = jnp.diag(
            jnp.array([1.0, 1.0, 1.0, 1.0 / complex(eps)], dtype=jnp.complex128)
        )
        return jnp.kron(jnp.eye(Stack.num_harmonics(N), dtype=jnp.complex128), block)

    @staticmethod
    def zero_order_mode_index(N: int, incident_pol: str) -> int:
        """Return the forward zero-order modal index for TE or TM incidence."""
        pol = incident_pol.upper()
        zero = Stack.zero_harmonic_index(N)
        if pol == "TE":
            return zero
        if pol == "TM":
            return Stack.num_harmonics(N) + zero
        raise ValueError(f"Unknown incident_pol={incident_pol!r}")

    @staticmethod
    def _isotropic_diag_blocks(Q_iso: jnp.ndarray) -> jnp.ndarray:
        """Extract the per-harmonic 4x4 isotropic operators from a full Q matrix."""
        if Q_iso.ndim == 3 and Q_iso.shape[1:] == (4, 4):
            return Q_iso

        if Q_iso.ndim != 2:
            raise ValueError(
                "Expected isotropic Q data with ndim 2 or 3, "
                f"got ndim={Q_iso.ndim} and shape={Q_iso.shape}."
            )

        if Q_iso.shape[0] != Q_iso.shape[1]:
            raise ValueError(f"Expected a square Q matrix, got shape={Q_iso.shape}.")
        if Q_iso.shape[0] % 4 != 0:
            raise ValueError(
                "Expected a full isotropic Q matrix with size divisible by 4, "
                f"got shape={Q_iso.shape}."
            )

        num_h = Q_iso.shape[0] // 4
        q_blocks = Q_iso.reshape(4, num_h, 4, num_h).transpose(1, 0, 3, 2)
        diag_blocks = jnp.diagonal(q_blocks, axis1=0, axis2=2)
        return jnp.moveaxis(diag_blocks, -1, 0)

    @staticmethod
    def _harmonic_diag_blocks_if_block_diagonal(
        Q: jnp.ndarray,
        tol: float,
    ) -> jnp.ndarray | None:
        """Return per-harmonic 4x4 blocks when a layer Q is harmonic-block diagonal.

        ``diagonalize_sort_layer_modes()`` receives layer operators in
        harmonic-major ordering, so rows/columns are grouped as

            [-H_y(n), H_x(n), E_y(n), D_x(n)]

        for one harmonic ``n`` at a time. A spatially homogeneous layer has no
        harmonic coupling, so in this basis the full Q matrix is block diagonal
        with independent 4x4 blocks along the harmonic axis.
        """
        if Q.ndim != 2:
            raise ValueError(
                "Expected layer Q data with ndim 2, "
                f"got ndim={Q.ndim} and shape={Q.shape}."
            )
        if Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Expected a square Q matrix, got shape={Q.shape}.")
        if Q.shape[0] % 4 != 0:
            raise ValueError(
                "Expected a layer Q matrix with size divisible by 4, "
                f"got shape={Q.shape}."
            )

        num_h = Q.shape[0] // 4
        q_blocks = Q.reshape(num_h, 4, num_h, 4)
        harmonic_blocks = q_blocks.transpose(0, 2, 1, 3)
        block_idx = jnp.arange(num_h)
        diag_blocks = harmonic_blocks[block_idx, block_idx]

        if num_h == 1:
            return diag_blocks

        offdiag_mask = jnp.ones((num_h, num_h), dtype=bool)
        offdiag_mask[block_idx, block_idx] = False
        max_abs_q = float(jnp.max(jnp.abs(Q)))
        max_offdiag = float(jnp.max(jnp.abs(harmonic_blocks[offdiag_mask])))

        if max_offdiag <= tol * max(1.0, max_abs_q):
            return diag_blocks
        return None

    @staticmethod
    def _block_diagonal_matrix(blocks: jnp.ndarray) -> jnp.ndarray:
        """Assemble a dense block-diagonal matrix from blocks[block, row, col]."""
        num_blocks, block_size, _ = blocks.shape
        out = jnp.zeros((num_blocks * block_size, num_blocks * block_size), dtype=blocks.dtype)
        for i in range(num_blocks):
            row_slice = slice(i * block_size, (i + 1) * block_size)
            out[row_slice, row_slice] = blocks[i]
        return out

    @staticmethod
    def _mode_normal_poynting_flux(tangential_fields: jnp.ndarray) -> jnp.ndarray:
        """Return Re(Sz) for modal fields in tangential ordering [-H_y, H_x, E_y, E_x]."""
        if tangential_fields.ndim == 2:
            if tangential_fields.shape[0] % 4 != 0:
                raise ValueError(
                    "Expected tangential fields with row count divisible by 4, "
                    f"got shape={tangential_fields.shape}."
                )
            fields = tangential_fields.reshape(tangential_fields.shape[0] // 4, 4, tangential_fields.shape[1])
            minus_hy = fields[:, 0, :]
            hx = fields[:, 1, :]
            ey = fields[:, 2, :]
            ex = fields[:, 3, :]
            return jnp.real(jnp.sum(-ex * jnp.conj(minus_hy) - ey * jnp.conj(hx), axis=0))

        if tangential_fields.ndim == 3 and tangential_fields.shape[1] == 4:
            minus_hy = tangential_fields[:, 0, :]
            hx = tangential_fields[:, 1, :]
            ey = tangential_fields[:, 2, :]
            ex = tangential_fields[:, 3, :]
            return jnp.real(-ex * jnp.conj(minus_hy) - ey * jnp.conj(hx))

        raise ValueError(
            "Expected tangential fields with shape (4*num_h, num_modes) or (num_h, 4, 4), "
            f"got {tangential_fields.shape}."
        )

    @staticmethod
    def _sort_layer_eigensystem(
        eigenvalues: jnp.ndarray,
        eigenvectors: jnp.ndarray,
        reference_fields: jnp.ndarray | None,
        tangential_transform: jnp.ndarray | None,
        tol: float,
        verbose: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Sort layer modes into the solver-wide [all forward, all backward] convention."""
        eigenvectors = Solver._normalize_columns(eigenvectors)
        direction_metric = jnp.where(
            jnp.abs(jnp.imag(eigenvalues)) > tol,
            jnp.imag(eigenvalues),
            -jnp.real(eigenvalues),
        )
        if tangential_transform is not None:
            poynting_score = Solver._mode_normal_poynting_flux(
                tangential_transform @ eigenvectors
            )
        else:
            poynting_score = jnp.zeros(eigenvalues.shape, dtype=jnp.float64)

        if reference_fields is None:
            sort_idx = jnp.lexsort((-direction_metric, -poynting_score))
            Solver._log(
                verbose,
                "Finished direction/poynting sorting without reference-basis overlap",
            )
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

        Solver._log(verbose, "Computing reference-basis overlaps for layer mode sorting")
        ref_coeffs = jnp.linalg.solve(reference_fields, eigenvectors)
        half = eigenvectors.shape[0] // 2
        forward_weight = jnp.linalg.norm(ref_coeffs[:half, :], axis=0)
        backward_weight = jnp.linalg.norm(ref_coeffs[half:, :], axis=0)
        overlap_score = forward_weight - backward_weight
        direction_metric_for_sort = jnp.where(
            jnp.abs(direction_metric) > tol,
            direction_metric,
            0.0,
        )

        # numpy.lexsort uses the last key as the primary key. This sorts first
        # by descending Poynting flux, then by descending direction metric, and
        # finally uses reference-basis overlap to break residual ties.
        sort_idx = jnp.lexsort(
            (-overlap_score, -direction_metric_for_sort, -poynting_score)
        )
        Solver._log(verbose, "Finished lexicographic forward/backward sorting with Poynting tie-break")
        return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

    @staticmethod
    def _normalize_columns(vectors: jnp.ndarray, tol: float = 1e-14) -> jnp.ndarray:
        norms = jnp.linalg.norm(vectors, axis=-2, keepdims=True)
        safe_norms = jnp.where(norms > tol, norms, 1.0)
        return vectors / safe_norms

    @staticmethod
    def _resolve_isotropic_pair(
        v1: jnp.ndarray,
        v2: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Resolve a degenerate isotropic mode pair into a stable TE/TM basis.

        The inputs may be a single pair with shape ``(4,)`` or a batched set of
        pairs with shape ``(..., 4)``.
        """
        w_te = v2[..., 3, None] * v1 - v1[..., 3, None] * v2
        w_tm = v2[..., 2, None] * v1 - v1[..., 2, None] * v2
        pair = jnp.stack([w_te, w_tm], axis=-1)
        pair = Solver._normalize_columns(pair)

        fallback = Solver._normalize_columns(jnp.stack([v1, v2], axis=-1))
        valid = jnp.linalg.norm(jnp.stack([w_te, w_tm], axis=-1), axis=-2) > 1e-14
        te = jnp.where(valid[..., 0, None], pair[..., 0], fallback[..., 0])
        tm = jnp.where(valid[..., 1, None], pair[..., 1], fallback[..., 1])
        return te, tm

    @staticmethod
    def diagonalize_sort_isotropic_modes(
        Q_iso: jnp.ndarray,
        tangential_transform: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize isotropic half-space modes harmonic-by-harmonic.

        Returns
            eigenvalues: shape ``(num_h, 4)``
            eigenvectors: shape ``(num_h, 4, 4)``

        with per-harmonic column order

            [forward_TE, forward_TM, backward_TE, backward_TM].
        """
        diag_blocks = Solver._isotropic_diag_blocks(Q_iso)

        eigenvalues, eigenvectors = jnp.linalg.eig(diag_blocks)
        tol = 1e-8
        direction_metric = jnp.where(
            jnp.abs(jnp.imag(eigenvalues)) > tol,
            jnp.imag(eigenvalues),
            -jnp.real(eigenvalues),
        )
        if tangential_transform is None:
            sort_idx = jnp.argsort(-direction_metric, axis=1)
        else:
            transform_blocks = Solver._isotropic_diag_blocks(tangential_transform)
            poynting_score = jnp.stack(
                [
                    Solver._mode_normal_poynting_flux(
                        transform_blocks[block_idx] @ eigenvectors[block_idx]
                    )
                    for block_idx in range(eigenvalues.shape[0])
                ],
                axis=0,
            )
            # Keep the existing direction metric as the primary classifier and
            # only use Re(Sz) to break residual ties inside degenerate blocks.
            sort_idx = jnp.stack(
                [
                    jnp.lexsort(
                        (
                            -poynting_score[block_idx],
                            -direction_metric[block_idx],
                        )
                    )
                    for block_idx in range(eigenvalues.shape[0])
                ],
                axis=0,
            )

        eigenvalues = jnp.take_along_axis(eigenvalues, sort_idx, axis=1)
        eigenvectors = jnp.take_along_axis(eigenvectors, sort_idx[:, None, :], axis=2)

        te_fwd, tm_fwd = Solver._resolve_isotropic_pair(eigenvectors[:, :, 0], eigenvectors[:, :, 1])
        te_bwd, tm_bwd = Solver._resolve_isotropic_pair(eigenvectors[:, :, 2], eigenvectors[:, :, 3])
        eigenvectors = jnp.stack([te_fwd, tm_fwd, te_bwd, tm_bwd], axis=2)
        return eigenvalues, Solver._normalize_columns(eigenvectors)

    @staticmethod
    def modes_to_fields_matrix(evecs: jnp.ndarray) -> jnp.ndarray:
        """Assemble the full isotropic modes-to-fields matrix.

        Rows are in harmonic-major field ordering

            [-H_y(n), H_x(n), E_y(n), D_x(n)]

        while columns are in the global modal ordering

            [forward_TE(-N..N), forward_TM(-N..N),
             backward_TE(-N..N), backward_TM(-N..N)].
        """
        if evecs.ndim != 3 or evecs.shape[1:] != (4, 4):
            raise ValueError(f"Expected eigenvectors with shape (num_h, 4, 4), got {evecs.shape}.")

        num_h = int(evecs.shape[0])
        size = 4 * num_h
        fields = jnp.zeros((size, size), dtype=evecs.dtype)
        for h in range(num_h):
            row_slice = slice(4 * h, 4 * (h + 1))
            fields[row_slice, h] = evecs[h, :, 0]
            fields[row_slice, num_h + h] = evecs[h, :, 1]
            fields[row_slice, 2 * num_h + h] = evecs[h, :, 2]
            fields[row_slice, 3 * num_h + h] = evecs[h, :, 3]
        return fields

    @staticmethod
    def diagonalize_sort_layer_modes(
        Q: jnp.ndarray,
        reference_fields: jnp.ndarray | None = None,
        tangential_transform: jnp.ndarray | None = None,
        tol: float = 1e-9,
        verbose: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize a full layer Q matrix in harmonic-major basis.

        The input ``Q`` is in harmonic-major ordering, so a spatially uniform
        layer appears as a block-diagonal matrix with one independent 4x4 block
        per harmonic. In that case we avoid a full dense eigendecomposition and
        diagonalize the 4x4 blocks directly before reconstructing the full
        eigensystem. Both the fast path and the dense fallback are then sorted
        into the solver-wide modal ordering whose first half contains all
        forward/right-going modes and whose second half contains all
        backward/left-going modes.
        """
        diag_blocks = Solver._harmonic_diag_blocks_if_block_diagonal(Q, tol=tol)
        if diag_blocks is not None:
            Solver._log(
                verbose,
                (
                    "Detected block-diagonal harmonic layer Q; using per-harmonic "
                    "4x4 fast path with numpy.linalg.eig"
                ),
            )
            block_eigenvalues, block_eigenvectors = jnp.linalg.eig(diag_blocks)
            eigenvalues = block_eigenvalues.reshape(-1)
            eigenvectors = Solver._block_diagonal_matrix(block_eigenvectors)
            Solver._log(verbose, "Finished harmonic block fast path; normalizing layer eigenvectors")
        else:
            Solver._log(verbose, f"Starting dense scipy.linalg.eig fallback for layer matrix with shape {Q.shape}")
            eigenvalues, eigenvectors = scipy.linalg.eig(Q)
            Solver._log(verbose, "Finished scipy.linalg.eig; normalizing layer eigenvectors")

        Solver._log(verbose, "Starting layer mode direction classification and sorting")
        return Solver._sort_layer_eigensystem(
            eigenvalues,
            eigenvectors,
            reference_fields=reference_fields,
            tangential_transform=tangential_transform,
            tol=tol,
            verbose=verbose,
        )

    @staticmethod
    def basis_change_transfer_matrix(left_fields: jnp.ndarray, right_fields: jnp.ndarray) -> jnp.ndarray:
        """Return the coefficient map from the left modal basis into the right modal basis."""
        return jnp.linalg.solve(right_fields, left_fields)

    @staticmethod
    def transfer_to_scattering(T: jnp.ndarray) -> ScatteringMatrix:
        """Convert a 2x2-block transfer matrix into an S-matrix.

        Throughout this solver we use the standard two-port convention

            [a_L^-]   [S11  S12] [a_L^+]
            [a_R^+] = [S21  S22] [a_R^-]

        where:
        - ``a_L^+`` are forward/right-going modes incident from the left
        - ``a_L^-`` are backward/left-going modes leaving on the left
        - ``a_R^-`` are backward/left-going modes incident from the right
        - ``a_R^+`` are forward/right-going modes leaving on the right

        The corresponding transfer matrix uses the convention

            [a_R^+]   [T11  T12] [a_L^+]
            [a_R^-] = [T21  T22] [a_L^-]

        so ``S21`` is always the left-to-right transmission block and ``S12``
        is always the right-to-left transmission block.
        """
        half = T.shape[0] // 2
        T11 = T[:half, :half]
        T12 = T[:half, half:]
        T21 = T[half:, :half]
        T22 = T[half:, half:]

        rhs = jnp.concatenate([T21, jnp.eye(half, dtype=T22.dtype)], axis=1)
        solution = jnp.linalg.solve(T22, rhs)
        T22_inv_T21 = solution[:, :half]
        T22_inv = solution[:, half:]

        S11 = -T22_inv_T21
        S12 = T22_inv
        S21 = T11 - T12 @ T22_inv_T21
        S22 = T12 @ T22_inv
        return S11, S12, S21, S22

    @staticmethod
    def basis_change_scattering_matrix(
        left_fields: jnp.ndarray,
        right_fields: jnp.ndarray,
    ) -> ScatteringMatrix:
        """Return the zero-thickness S-matrix for a change of adjacent modal basis."""
        return Solver.transfer_to_scattering(
            Solver.basis_change_transfer_matrix(left_fields, right_fields)
        )

    @staticmethod
    def isotropic_mode_fields(
        Q_iso: jnp.ndarray,
        N: int,
        tangential_transform: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return the isotropic half-space fields matrix in the solver's modal ordering."""
        _, evecs = Solver.diagonalize_sort_isotropic_modes(
            Q_iso,
            tangential_transform=tangential_transform,
        )
        if int(evecs.shape[0]) != Stack.num_harmonics(N):
            raise ValueError(
                f"Expected {Stack.num_harmonics(N)} harmonics for N={N}, got {evecs.shape[0]}."
            )
        return Solver.modes_to_fields_matrix(evecs)

    @staticmethod
    def layer_mode_fields(
        q_matrix: jnp.ndarray,
        reference_fields: jnp.ndarray,
        tangential_transform: jnp.ndarray | None = None,
        verbose: bool = False,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return sorted layer eigenvalues and the corresponding fields matrix."""
        return Solver.diagonalize_sort_layer_modes(
            q_matrix,
            reference_fields=reference_fields,
            tangential_transform=tangential_transform,
            verbose=verbose,
        )

    @staticmethod
    def modal_propagation_scattering_matrix(
        eigenvalues: jnp.ndarray,
        thickness: float,
    ) -> ScatteringMatrix:
        """Return the diagonal propagation S-matrix inside one layer's modal basis.

        Modal ordering convention:
        - the first half of ``eigenvalues`` are forward/right-going layer modes
        - the second half are backward/left-going layer modes

        Scattering-port convention:

            [a_L^-]   [S11  S12] [a_L^+]
            [a_R^+] = [S21  S22] [a_R^-]

        For a homogeneous layer there is no internal reflection, so ``S11`` and
        ``S22`` vanish.

        A forward mode incident from the left propagates to the right face, so
        the forward propagation factor belongs in ``S21``.

        A backward mode incident from the right propagates to the left face, so
        the backward propagation factor belongs in ``S12``.
        """
        n = eigenvalues.shape[0]
        if n % 2 != 0:
            raise ValueError(f"Expected an even number of eigenvalues, got shape[0]={n}.")

        half = n // 2
        X_forward = jnp.diag(jnp.exp(eigenvalues[:half] * thickness))
        X_backward = jnp.diag(jnp.exp(-eigenvalues[half:] * thickness))
        Z = jnp.zeros_like(X_forward)
        return Z, X_backward, X_forward, Z

    @staticmethod
    def redheffer_star_product(Sa: ScatteringMatrix, Sb: ScatteringMatrix) -> ScatteringMatrix:
        """Redheffer star product for two compatible adjacent-port S-matrices."""
        A11, A12, A21, A22 = Sa
        B11, B12, B21, B22 = Sb
        half = A11.shape[0]
        I = jnp.eye(half, dtype=A11.dtype)

        system = I - A22 @ B11
        rhs = jnp.concatenate([A21, A22 @ B12], axis=1)
        solution = jnp.linalg.solve(system, rhs)
        inv_a_A21 = solution[:, :half]
        inv_a_A22_B12 = solution[:, half:]

        B11_inv_a_A21 = B11 @ inv_a_A21
        B11_inv_a_A22_B12 = B11 @ inv_a_A22_B12
        B21_inv_a_A21 = B21 @ inv_a_A21
        B21_inv_a_A22_B12 = B21 @ inv_a_A22_B12

        C11 = A11 + A12 @ B11_inv_a_A21
        C12 = A12 @ (B12 + B11_inv_a_A22_B12)
        C21 = B21_inv_a_A21
        C22 = B22 + B21_inv_a_A22_B12
        return C11, C12, C21, C22

    @staticmethod
    def chain_scattering_matrices(S_list: list[ScatteringMatrix]) -> ScatteringMatrix:
        """Chain a list of compatible S-matrices with the Redheffer star product."""
        if not S_list:
            raise ValueError("Expected at least one scattering matrix to chain.")

        result = S_list[0]
        for S in S_list[1:]:
            result = Solver.redheffer_star_product(result, S)
        return result

    @staticmethod
    def identity_scattering_matrix(num_modes: int) -> ScatteringMatrix:
        """Return the transparent two-port S-matrix on one modal basis."""
        identity = jnp.eye(num_modes, dtype=jnp.complex128)
        zero = jnp.zeros_like(identity)
        return zero, identity, identity, zero

    @staticmethod
    def stack_scattering_blocks(
        stack: Stack,
        stack_data: StackSolveData,
    ) -> list[ScatteringMatrix]:
        """Return the interface and propagation S-matrices for one stack decomposition."""
        if not stack_data.layer_modes:
            return [
                Solver.basis_change_scattering_matrix(
                    stack_data.substrate_continuity_fields,
                    stack_data.superstrate_continuity_fields,
                )
            ]

        blocks: list[ScatteringMatrix] = [
            Solver.basis_change_scattering_matrix(
                stack_data.substrate_continuity_fields,
                stack_data.layer_modes[0][2],
            )
        ]

        for i, (eigenvalues, _, layer_continuity_fields) in enumerate(stack_data.layer_modes):
            blocks.append(
                Solver.modal_propagation_scattering_matrix(
                    eigenvalues,
                    stack.thickness_normalized(i),
                )
            )
            right_fields = (
                stack_data.superstrate_continuity_fields
                if i == len(stack_data.layer_modes) - 1
                else stack_data.layer_modes[i + 1][2]
            )
            blocks.append(
                Solver.basis_change_scattering_matrix(
                    layer_continuity_fields,
                    right_fields,
                )
            )

        return blocks

    @staticmethod
    def scattering_prefix_suffix_chains(
        blocks: list[ScatteringMatrix],
    ) -> tuple[list[ScatteringMatrix], list[ScatteringMatrix]]:
        """Return cumulative left-to-right and right-to-left scattering chains."""
        if not blocks:
            raise ValueError("Expected at least one scattering block.")

        num_modes = blocks[0][0].shape[0]
        prefix_scattering: list[ScatteringMatrix] = [
            Solver.identity_scattering_matrix(num_modes)
        ]
        for block in blocks:
            prefix_scattering.append(
                Solver.redheffer_star_product(prefix_scattering[-1], block)
            )

        suffix_scattering: list[ScatteringMatrix] = [
            Solver.identity_scattering_matrix(num_modes) for _ in range(len(blocks) + 1)
        ]
        for i in range(len(blocks) - 1, -1, -1):
            suffix_scattering[i] = Solver.redheffer_star_product(
                blocks[i],
                suffix_scattering[i + 1],
            )
        return prefix_scattering, suffix_scattering

    @staticmethod
    def build_stack_solve_data(
        stack: Stack,
        N: int,
        num_points: int = 512,
        verbose: bool = False,
    ) -> StackSolveData:
        """Build and cache all modal data needed for one stack solve."""
        Solver._log(
            verbose,
            (
                f"Building shared stack solve data: N={N}, num_points={num_points}, "
                f"num_layers={len(stack.layers)}"
            ),
        )
        layer_toeplitz_matrices = [
            layer.build_toeplitz_fourier_matrices(N, num_points=num_points)
            for layer in stack.layers
        ]
        q_matrices_harmonic_major = stack.build_all_Q_matrices_harmonic_major_normalized(
            N,
            num_points=num_points,
        )
        Solver._log(verbose, f"Built {len(q_matrices_harmonic_major)} harmonic-major layer Q matrices")

        Solver._log(verbose, "Diagonalizing isotropic substrate modes")
        substrate_tangential_transform = Solver.isotropic_reduced_to_tangential_transform_harmonic_major(
            stack.eps_substrate,
            N,
        )
        substrate_fields = Solver.isotropic_mode_fields(
            stack.get_Q_substrate_normalized(N, num_points=num_points),
            N,
            tangential_transform=substrate_tangential_transform,
        )
        substrate_continuity_fields = Solver.reduced_to_tangential_fields_harmonic_major(
            substrate_fields,
            substrate_tangential_transform,
        )
        Solver._log(verbose, "Diagonalizing isotropic superstrate modes")
        superstrate_tangential_transform = Solver.isotropic_reduced_to_tangential_transform_harmonic_major(
            stack.eps_superstrate,
            N,
        )
        superstrate_fields = Solver.isotropic_mode_fields(
            stack.get_Q_superstrate_normalized(N, num_points=num_points),
            N,
            tangential_transform=superstrate_tangential_transform,
        )
        superstrate_continuity_fields = Solver.reduced_to_tangential_fields_harmonic_major(
            superstrate_fields,
            superstrate_tangential_transform,
        )

        layer_modes: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
        reference_fields = substrate_fields
        for i, (q_matrix, toeplitz_matrices) in enumerate(
            zip(q_matrices_harmonic_major, layer_toeplitz_matrices)
        ):
            layer_tangential_transform = Layer.build_reduced_to_tangential_field_transform_harmonic_major(
                toeplitz_matrices,
                N,
            )
            Solver._log(
                verbose,
                f"Diagonalizing layer {i} Q matrix with shape {q_matrix.shape}",
            )
            eigenvalues, layer_fields = Solver.layer_mode_fields(
                q_matrix,
                reference_fields,
                tangential_transform=layer_tangential_transform,
                verbose=verbose,
            )
            layer_continuity_fields = Solver.reduced_to_tangential_fields_harmonic_major(
                layer_fields,
                layer_tangential_transform,
            )
            layer_modes.append((eigenvalues, layer_fields, layer_continuity_fields))
            reference_fields = layer_fields
            Solver._log(
                verbose,
                f"Sorted layer {i} modes into forward/backward basis ({eigenvalues.shape[0]} eigenvalues)",
            )

        if not layer_modes:
            Solver._log(
                verbose,
                "No internal layers found; returning direct substrate/superstrate basis change",
            )
            total_scattering = Solver.basis_change_scattering_matrix(
                substrate_continuity_fields,
                superstrate_continuity_fields,
            )
        else:
            blocks = Solver.stack_scattering_blocks(
                stack,
                StackSolveData(
                    substrate_fields=substrate_fields,
                    superstrate_fields=superstrate_fields,
                    substrate_continuity_fields=substrate_continuity_fields,
                    superstrate_continuity_fields=superstrate_continuity_fields,
                    layer_modes=layer_modes,
                    total_scattering=Solver.identity_scattering_matrix(
                        2 * Stack.num_harmonics(N)
                    ),
                ),
            )
            Solver._log(
                verbose,
                f"Concatenating {len(blocks)} scattering matrices with Redheffer star products",
            )
            total_scattering = Solver.chain_scattering_matrices(blocks)

        return StackSolveData(
            substrate_fields=substrate_fields,
            superstrate_fields=superstrate_fields,
            substrate_continuity_fields=substrate_continuity_fields,
            superstrate_continuity_fields=superstrate_continuity_fields,
            layer_modes=layer_modes,
            total_scattering=total_scattering,
        )

    @staticmethod
    def total_scattering_matrix(
        stack: Stack,
        N: int,
        num_points: int = 512,
        verbose: bool = False,
        precomputed: StackSolveData | None = None,
    ) -> ScatteringMatrix:
        """Return the stack S-matrix in substrate/superstrate modal bases."""
        if precomputed is None:
            Solver._log(
                verbose,
                (
                    f"Building total scattering matrix: N={N}, num_points={num_points}, "
                    f"num_layers={len(stack.layers)}"
                ),
            )
            precomputed = Solver.build_stack_solve_data(
                stack,
                N,
                num_points=num_points,
                verbose=verbose,
            )
        else:
            Solver._log(verbose, "Reusing precomputed stack solve data for total scattering matrix")
        return precomputed.total_scattering

    @staticmethod
    def reflection_transmission(
        stack: Stack,
        N: int,
        incident_pol: str = "TE",
        num_points: int = 512,
        verbose: bool = True,
        precomputed: StackSolveData | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return reflected and transmitted modal amplitudes for unit normal incidence."""
        Solver._log(verbose, f"Computing reflection/transmission for {incident_pol.upper()} incidence")
        S11, _, S21, _ = Solver.total_scattering_matrix(
            stack,
            N,
            num_points=num_points,
            verbose=verbose,
            precomputed=precomputed,
        )
        half = 2 * Stack.num_harmonics(N)

        inc = jnp.zeros(half, dtype=jnp.complex128)
        inc[Solver.zero_order_mode_index(N, incident_pol)] = 1.0

        return S11 @ inc, S21 @ inc
