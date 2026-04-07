from __future__ import annotations

import jax
import jax.numpy as jnp

from . import _config  # noqa: F401
from .stack import Stack


ScatteringMatrix = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


class Solver:
    """Modal RCWA solver using adjacent port bases."""
    @staticmethod
    def reorder_matrix(N: int) -> jnp.ndarray:
        """Map polarization-major ordering into harmonic-major field ordering."""
        num_h = Stack.num_harmonics(N)
        size = 4 * num_h
        src = jnp.array([p * num_h + h for h in range(num_h) for p in range(4)])
        return jnp.eye(size, dtype=jnp.complex128)[src]

    @staticmethod
    def _isotropic_diag_blocks(Q_iso: jnp.ndarray) -> jnp.ndarray:
        """Extract per-harmonic 4x4 isotropic blocks from either Q layout.

        Accepted input layouts:
            1. Old tensor layout with shape ``(num_h, num_h, 4, 4)``.
               Only the diagonal ``(h, h)`` blocks are nonzero for isotropic
               media, and each diagonal entry is a 4x4 matrix for one harmonic.

            2. New flattened full-matrix layout with shape
               ``(4 * num_h, 4 * num_h)`` in the component-major ordering

                   [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

               In this representation, the field components are grouped first
               and the harmonic index varies inside each group.

        Output:
            A tensor with shape ``(num_h, 4, 4)``.

            Entry ``output[h]`` is the 4x4 matrix acting on the four reduced
            field components of harmonic ``h`` alone. The harmonic ordering is
            the same as everywhere else in the code:

                h = 0, 1, ..., num_h - 1  corresponds to  n = -N, ..., N.

        For a uniform isotropic medium the full matrix is decoupled
        harmonic-by-harmonic, so regrouping the component-major indices by
        harmonic recovers these same independent 4x4 blocks.
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

        Input:
            ``Q_iso`` may be either
            - the old tensor layout with shape ``(num_h, num_h, 4, 4)``, or
            - the new flattened layout with shape ``(4 * num_h, 4 * num_h)``.

            In both cases the matrix describes a uniform isotropic half-space.

        Internal reduction:
            The input is first converted into a tensor of shape
            ``(num_h, 4, 4)``, where each 4x4 block is the single-harmonic
            isotropic operator for one diffraction order.

        Output:
            ``(eigenvalues, eigenvectors)`` where
            - ``eigenvalues`` has shape ``(num_h, 4)``
            - ``eigenvectors`` has shape ``(num_h, 4, 4)``

            ``eigenvalues[h, :]`` are the four modal eigenvalues for harmonic
            ``h``.

            ``eigenvectors[h, :, :]`` is a 4x4 matrix whose columns are the
            corresponding modal field vectors in the single-harmonic basis

                [-H_y, H_x, E_y, D_x]^T

            for that harmonic.

        Sorting convention:
            Inside each 4x4 block, the modes are sorted into
            ``[forward_1, forward_2, backward_1, backward_2]`` using the same
            eigenvalue-based ordering logic as the previous implementation.
        """
        diag_blocks = Stack._isotropic_diag_blocks(Q_iso)

        def _resolve_pair(v1: jnp.ndarray, v2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # Each input vector has shape (4,) and lives in the single-harmonic
            # field basis [-H_y, H_x, E_y, D_x]^T. This helper recombines a
            # nearly degenerate pair into a more stable polarization basis.
            w1 = v2[3] * v1 - v1[3] * v2
            w1_norm = jnp.linalg.norm(w1)
            w1 = jnp.where(w1_norm > 1e-14, w1 / w1_norm, v1)

            w2 = v2[2] * v1 - v1[2] * v2
            w2_norm = jnp.linalg.norm(w2)
            w2 = jnp.where(w2_norm > 1e-14, w2 / w2_norm, v2)
            return w1, w2

        def _eig_sort_single(block: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # `block` has shape (4, 4) and is the isotropic operator for one
            # harmonic order only.
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
    

    @staticmethod
    def modes_to_fields_matrix(evecs: jnp.ndarray) -> jnp.ndarray:
        """Assemble a block-diagonal modes-to-fields matrix across harmonics."""
        return jax.scipy.linalg.block_diag(*evecs)

    @staticmethod
    def diagonalize_sort_layer_modes(
        Q: jnp.ndarray, reference_fields: jnp.ndarray | None = None, tol: float = 1e-9
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize a full layer Q matrix and sort it into [forward, backward] modes."""
        eigenvalues, eigenvectors = jnp.linalg.eig(Q)
        direction_metric = jnp.where(
            jnp.abs(jnp.imag(eigenvalues)) > tol,
            jnp.imag(eigenvalues),
            -jnp.real(eigenvalues),
        )
        if reference_fields is None:
            sort_idx = jnp.argsort(-direction_metric)
        else:
            ref_coeffs = jnp.linalg.solve(reference_fields, eigenvectors)
            half = Q.shape[0] // 2
            forward_weight = jnp.linalg.norm(ref_coeffs[:half, :], axis=0)
            backward_weight = jnp.linalg.norm(ref_coeffs[half:, :], axis=0)
            overlap_score = forward_weight - backward_weight

            forward_idx: list[int] = []
            backward_idx: list[int] = []
            ambiguous_idx: list[int] = []
            for i in range(int(Q.shape[0])):
                metric = float(direction_metric[i])
                if metric > tol:
                    forward_idx.append(i)
                elif metric < -tol:
                    backward_idx.append(i)
                else:
                    ambiguous_idx.append(i)

            ambiguous_idx.sort(key=lambda i: float(overlap_score[i]), reverse=True)
            while len(forward_idx) < half and ambiguous_idx:
                forward_idx.append(ambiguous_idx.pop(0))
            while len(backward_idx) < half and ambiguous_idx:
                backward_idx.append(ambiguous_idx.pop(0))

            if len(forward_idx) != half or len(backward_idx) != half:
                sort_idx = jnp.argsort(-direction_metric)
                eigenvalues = eigenvalues[sort_idx]
                eigenvectors = eigenvectors[:, sort_idx]
                return eigenvalues, eigenvectors

            forward_idx.sort(
                key=lambda i: (float(overlap_score[i]), float(direction_metric[i])),
                reverse=True,
            )
            backward_idx.sort(
                key=lambda i: (float(-overlap_score[i]), float(-direction_metric[i])),
                reverse=True,
            )
            sort_idx = jnp.array(forward_idx + backward_idx, dtype=jnp.int32)

        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        return eigenvalues, eigenvectors

    @staticmethod
    def basis_change_scattering_matrix(
        left_fields: jnp.ndarray, right_fields: jnp.ndarray
    ) -> ScatteringMatrix:
        def transfer_to_scattering(T: jnp.ndarray) -> ScatteringMatrix:
            """Convert a 2x2-block transfer matrix into an S-matrix."""
            half = T.shape[0] // 2
            T11 = T[:half, :half]
            T12 = T[:half, half:]
            T21 = T[half:, :half]
            T22 = T[half:, half:]

            T22_inv_T21 = jnp.linalg.solve(T22, T21)
            T22_inv = jnp.linalg.solve(T22, jnp.eye(half, dtype=T22.dtype))

            S11 = -T22_inv_T21
            S12 = T22_inv
            S21 = T11 - T12 @ T22_inv_T21
            S22 = T12 @ T22_inv
            return S11, S12, S21, S22

        """Return the zero-thickness S-matrix for a change of adjacent modal basis."""
        return transfer_to_scattering(
            jnp.linalg.solve(right_fields, left_fields)
        )

    @staticmethod
    def isotropic_mode_fields(Q_iso: jnp.ndarray, N: int) -> jnp.ndarray:
        """Return the isotropic half-space fields matrix in harmonic-major ordering."""
        _, evecs = Stack.diagonalize_sort_isotropic_modes(Q_iso)
        return Solver.modes_to_fields_matrix(evecs) @ Solver.reorder_matrix(N)

    @staticmethod
    def layer_mode_fields(
        q_matrix: jnp.ndarray,
        reference_fields: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return sorted layer eigenvalues and the corresponding fields matrix."""
        return Solver.diagonalize_sort_layer_modes(q_matrix, reference_fields=reference_fields)

    @staticmethod
    def modal_propagation_scattering_matrix(
        eigenvalues: jnp.ndarray,
        thickness: float,
    ) -> ScatteringMatrix:
        """Return the diagonal propagation S-matrix inside one layer's modal basis."""
        n = eigenvalues.shape[0]
        if n % 2 != 0:
            raise ValueError(f"Expected an even number of eigenvalues, got shape[0]={n}.")

        half = n // 2
        X = jnp.diag(jnp.exp(eigenvalues[:half] * thickness))
        Z = jnp.zeros_like(X)
        return Z, X, X, Z


    @staticmethod
    def redheffer_star_product(Sa: ScatteringMatrix, Sb: ScatteringMatrix) -> ScatteringMatrix:
        """Redheffer star product for two compatible adjacent-port S-matrices."""
        A11, A12, A21, A22 = Sa
        B11, B12, B21, B22 = Sb
        half = A11.shape[0]
        I = jnp.eye(half, dtype=A11.dtype)

        F_A21 = jnp.linalg.solve(I - A22 @ B11, A21)
        G_B12 = jnp.linalg.solve(I - B11 @ A22, B12)
        G_B11 = jnp.linalg.solve(I - B11 @ A22, B11)

        C11 = A11 + A12 @ G_B11 @ A21
        C12 = A12 @ G_B12
        C21 = B21 @ F_A21
        C22 = B22 + B21 @ jnp.linalg.solve(I - A22 @ B11, A22) @ B12
        return C11, C12, C21, C22

    @staticmethod
    def chain_scattering_matrices(S_list: list[ScatteringMatrix]) -> ScatteringMatrix:
        """Chain a list of compatible S-matrices with the Redheffer star product."""
        result = S_list[0]
        for S in S_list[1:]:
            result = Solver.redheffer_star_product(result, S)
        return result

    @staticmethod
    def total_scattering_matrix(stack: Stack, N: int, num_points: int = 512) -> ScatteringMatrix:
        """Return the stack S-matrix in substrate/superstrate modal bases."""
        q_matrices = stack.build_all_Q_matrices_normalized(N, num_points=num_points)
        substrate_fields = Solver.isotropic_mode_fields(stack.get_Q_substrate_normalized(N), N)
        superstrate_fields = Solver.isotropic_mode_fields(stack.get_Q_superstrate_normalized(N), N)

        if not q_matrices:
            return Solver.basis_change_scattering_matrix(substrate_fields, superstrate_fields)

        layer_modes: list[tuple[jnp.ndarray, jnp.ndarray]] = []
        reference_fields = substrate_fields
        for q_matrix in q_matrices:
            layer_modes.append(Solver.layer_mode_fields(q_matrix, reference_fields))
            reference_fields = layer_modes[-1][1]

        S_list: list[ScatteringMatrix] = []
        for i, (layer_eigenvalues, layer_fields) in enumerate(layer_modes):
            left_fields = substrate_fields if i == 0 else layer_modes[i - 1][1]
            right_fields = superstrate_fields if i == len(layer_modes) - 1 else layer_modes[i + 1][1]

            S_list.append(Solver.basis_change_scattering_matrix(left_fields, layer_fields))
            S_list.append(
                Solver.modal_propagation_scattering_matrix(
                    layer_eigenvalues,
                    stack.thickness_normalized(i),
                )
            )
            S_list.append(Solver.basis_change_scattering_matrix(layer_fields, right_fields))

        return Solver.chain_scattering_matrices(S_list)
