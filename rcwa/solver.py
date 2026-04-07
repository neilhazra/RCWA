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
    def zero_order_mode_index(N: int, incident_pol: str) -> int:
        """Return the flat TE/TM modal index for the zero diffraction order."""
        zero = Stack.zero_harmonic_index(N)
        if incident_pol == "TE":
            return zero
        if incident_pol == "TM":
            return Stack.num_harmonics(N) + zero
        raise ValueError(f"Unknown incident_pol={incident_pol!r}")

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
    def _basis_change_transfer_matrix(left_fields: jnp.ndarray, right_fields: jnp.ndarray) -> jnp.ndarray:
        """Return the coefficient map from left basis amplitudes to right basis amplitudes."""
        return jnp.linalg.solve(right_fields, left_fields)

    @staticmethod
    def basis_change_scattering_matrix(
        left_fields: jnp.ndarray, right_fields: jnp.ndarray
    ) -> ScatteringMatrix:
        """Return the zero-thickness S-matrix for a change of adjacent modal basis."""
        return Solver.transfer_to_scattering(
            Solver._basis_change_transfer_matrix(left_fields, right_fields)
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

    @staticmethod
    def reflection_transmission(
        stack: Stack,
        N: int,
        incident_pol: str = "TE",
        num_points: int = 2048,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return reflected and transmitted modal amplitudes for normal incidence."""
        S11, _, S21, _ = Solver.total_scattering_matrix(stack, N, num_points=num_points)
        half = 2 * Stack.num_harmonics(N)

        inc = jnp.zeros(half, dtype=jnp.complex128)
        inc = inc.at[Solver.zero_order_mode_index(N, incident_pol)].set(1.0)

        r = S11 @ inc
        t = S21 @ inc
        return r, t
