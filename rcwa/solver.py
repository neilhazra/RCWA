from __future__ import annotations

from functools import lru_cache

import numpy as jnp
import scipy.linalg

from . import _config  # noqa: F401
from .stack import Stack


ScatteringMatrix = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


class Solver:
    """Modal RCWA solver using adjacent-port scattering matrices."""

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
    def _normalize_columns(vectors: jnp.ndarray, tol: float = 1e-14) -> jnp.ndarray:
        norms = jnp.linalg.norm(vectors, axis=0, keepdims=True)
        safe_norms = jnp.where(norms > tol, norms, 1.0)
        return vectors / safe_norms

    def diagonalize_sort_isotropic_modes(Q_iso: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize isotropic half-space modes harmonic-by-harmonic.

        Returns
            eigenvalues: shape ``(num_h, 4)``
            eigenvectors: shape ``(num_h, 4, 4)``

        with per-harmonic column order

            [forward_TE, forward_TM, backward_TE, backward_TM].
        """
        diag_blocks = Solver._isotropic_diag_blocks(Q_iso)

        def _resolve_pair(v1: jnp.ndarray, v2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            # In an isotropic medium the forward and backward pairs are often
            # polarization-degenerate. Recombine each pair into a stable TE/TM
            # basis by constructing one vector with vanishing D_x and one with
            # vanishing E_y in the reduced field basis [-H_y, H_x, E_y, D_x]^T.
            w_te = v2[3] * v1 - v1[3] * v2
            w_tm = v2[2] * v1 - v1[2] * v2
            pair = jnp.column_stack([w_te, w_tm])
            pair = Solver._normalize_columns(pair)

            fallback = Solver._normalize_columns(jnp.column_stack([v1, v2]))
            valid = jnp.linalg.norm(jnp.column_stack([w_te, w_tm]), axis=0) > 1e-14
            return (
                jnp.where(valid[0], pair[:, 0], fallback[:, 0]),
                jnp.where(valid[1], pair[:, 1], fallback[:, 1]),
            )

        def _eig_sort_single(block: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            eigenvalues, eigenvectors = jnp.linalg.eig(block)
            tol = 1e-8
            direction_metric = jnp.where(
                jnp.abs(jnp.imag(eigenvalues)) > tol,
                jnp.imag(eigenvalues),
                -jnp.real(eigenvalues),
            )
            sort_idx = jnp.argsort(-direction_metric)
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]

            te_fwd, tm_fwd = _resolve_pair(eigenvectors[:, 0], eigenvectors[:, 1])
            te_bwd, tm_bwd = _resolve_pair(eigenvectors[:, 2], eigenvectors[:, 3])
            eigenvectors = jnp.column_stack([te_fwd, tm_fwd, te_bwd, tm_bwd])
            return eigenvalues, Solver._normalize_columns(eigenvectors)

        mode_data = [_eig_sort_single(block) for block in diag_blocks]
        eigenvalues = jnp.stack([vals for vals, _ in mode_data], axis=0)
        eigenvectors = jnp.stack([vecs for _, vecs in mode_data], axis=0)
        return eigenvalues, eigenvectors

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
        N = (num_h - 1) // 2
        size = 4 * num_h
        fields = jnp.zeros((size, size), dtype=evecs.dtype)
        for h in range(num_h):
            fields[4 * h : 4 * (h + 1), 4 * h : 4 * (h + 1)] = evecs[h]
        return fields[:, Solver.mode_reorder_indices(N)]

    @staticmethod
    def diagonalize_sort_layer_modes(
        Q: jnp.ndarray,
        reference_fields: jnp.ndarray | None = None,
        tol: float = 1e-9,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagonalize a full layer Q matrix and sort it into [forward, backward] modes."""
        eigenvalues, eigenvectors = jnp.linalg.eig(Q)
        eigenvectors = Solver._normalize_columns(eigenvectors)
        half = Q.shape[0] // 2
        eigvals = [complex(v) for v in eigenvalues]

        def _pair_backward_indices(forward_idx: list[int], backward_idx: list[int]) -> list[int]:
            paired_backward_idx: list[int] = []
            remaining_backward_idx = backward_idx.copy()
            for idx in forward_idx:
                match = min(
                    remaining_backward_idx,
                    key=lambda j: abs(eigvals[idx] + eigvals[j]),
                )
                paired_backward_idx.append(match)
                remaining_backward_idx.remove(match)
            return paired_backward_idx

        direction_metric = jnp.where(
            jnp.abs(jnp.imag(eigenvalues)) > tol,
            jnp.imag(eigenvalues),
            -jnp.real(eigenvalues),
        )

        if reference_fields is None:
            sort_idx = jnp.argsort(-direction_metric)
            forward_idx = [int(i) for i in sort_idx[:half]]
            backward_idx = [int(i) for i in sort_idx[half:]]
            paired_backward_idx = _pair_backward_indices(forward_idx, backward_idx)
            sort_idx = jnp.array(forward_idx + paired_backward_idx, dtype=jnp.int32)
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

        ref_coeffs = jnp.linalg.solve(reference_fields, eigenvectors)
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
            forward_idx = [int(i) for i in sort_idx[:half]]
            backward_idx = [int(i) for i in sort_idx[half:]]
            paired_backward_idx = _pair_backward_indices(forward_idx, backward_idx)
            sort_idx = jnp.array(forward_idx + paired_backward_idx, dtype=jnp.int32)
            return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

        forward_idx.sort(
            key=lambda i: (float(overlap_score[i]), float(direction_metric[i])),
            reverse=True,
        )
        paired_backward_idx = _pair_backward_indices(forward_idx, backward_idx)
        sort_idx = jnp.array(forward_idx + paired_backward_idx, dtype=jnp.int32)
        return eigenvalues[sort_idx], eigenvectors[:, sort_idx]

    @staticmethod
    def basis_change_transfer_matrix(left_fields: jnp.ndarray, right_fields: jnp.ndarray) -> jnp.ndarray:
        """Return the coefficient map from the left modal basis into the right modal basis."""
        return jnp.linalg.solve(right_fields, left_fields)

    @staticmethod
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
    def isotropic_mode_fields(Q_iso: jnp.ndarray, N: int) -> jnp.ndarray:
        """Return the isotropic half-space fields matrix in the solver's modal ordering."""
        _, evecs = Solver.diagonalize_sort_isotropic_modes(Q_iso)
        if int(evecs.shape[0]) != Stack.num_harmonics(N):
            raise ValueError(
                f"Expected {Stack.num_harmonics(N)} harmonics for N={N}, got {evecs.shape[0]}."
            )
        return Solver.modes_to_fields_matrix(evecs)

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
        X_forward = jnp.diag(jnp.exp(eigenvalues[:half] * thickness))
        X_backward = jnp.diag(jnp.exp(-eigenvalues[half:] * thickness))
        Z = jnp.zeros_like(X_forward)
        return Z, X_forward, X_backward, Z

    @staticmethod
    def redheffer_star_product(Sa: ScatteringMatrix, Sb: ScatteringMatrix) -> ScatteringMatrix:
        """Redheffer star product for two compatible adjacent-port S-matrices."""
        A11, A12, A21, A22 = Sa
        B11, B12, B21, B22 = Sb
        half = A11.shape[0]
        I = jnp.eye(half, dtype=A11.dtype)

        inv_a = jnp.linalg.solve(I - A22 @ B11, I)
        inv_b = jnp.linalg.solve(I - B11 @ A22, I)

        C11 = A11 + A12 @ inv_b @ B11 @ A21
        C12 = A12 @ inv_b @ B12
        C21 = B21 @ inv_a @ A21
        C22 = B22 + B21 @ inv_a @ A22 @ B12
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
    def total_scattering_matrix(stack: Stack, N: int, num_points: int = 512) -> ScatteringMatrix:
        """Return the stack S-matrix in substrate/superstrate modal bases."""
        q_matrices = stack.build_all_Q_matrices_normalized(N, num_points=num_points)
        q_matrices_harmonic_major = [
            Solver.component_to_harmonic_major(q_matrix) for q_matrix in q_matrices
        ]

        substrate_fields = Solver.isotropic_mode_fields(
            stack.get_Q_substrate_normalized(N, num_points=num_points),
            N,
        )
        superstrate_fields = Solver.isotropic_mode_fields(
            stack.get_Q_superstrate_normalized(N, num_points=num_points),
            N,
        )

        if not q_matrices_harmonic_major:
            return Solver.basis_change_scattering_matrix(substrate_fields, superstrate_fields)

        layer_modes: list[tuple[jnp.ndarray, jnp.ndarray]] = []
        reference_fields = substrate_fields
        for q_matrix in q_matrices_harmonic_major:
            mode_data = Solver.layer_mode_fields(q_matrix, reference_fields)
            layer_modes.append(mode_data)
            reference_fields = mode_data[1]

        S_list: list[ScatteringMatrix] = [
            Solver.basis_change_scattering_matrix(substrate_fields, layer_modes[0][1])
        ]

        for i, (layer_eigenvalues, layer_fields) in enumerate(layer_modes):
            S_list.append(
                Solver.modal_propagation_scattering_matrix(
                    layer_eigenvalues,
                    stack.thickness_normalized(i),
                )
            )
            right_fields = (
                superstrate_fields if i == len(layer_modes) - 1 else layer_modes[i + 1][1]
            )
            S_list.append(Solver.basis_change_scattering_matrix(layer_fields, right_fields))

        return Solver.chain_scattering_matrices(S_list)

    @staticmethod
    def reflection_transmission(
        stack: Stack,
        N: int,
        incident_pol: str = "TE",
        num_points: int = 512,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return reflected and transmitted modal amplitudes for unit normal incidence."""
        S11, _, S21, _ = Solver.total_scattering_matrix(stack, N, num_points=num_points)
        half = 2 * Stack.num_harmonics(N)

        inc = jnp.zeros(half, dtype=jnp.complex128)
        inc[Solver.zero_order_mode_index(N, incident_pol)] = 1.0

        return S11 @ inc, S21 @ inc
