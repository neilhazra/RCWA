"""Visualize reduced-basis RCWA fields inside stack layers on the x-z cross section.

This file works with three distinct linear spaces that are easy to conflate:

1. Component-major Q-matrix basis from :mod:`rcwa.stack`

       [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T

   Here each field component is grouped across all retained harmonics.

2. Harmonic-major reduced-field basis used for reconstruction in this file

       [-H_y(n), H_x(n), E_y(n), D_x(n)]^T

   stacked for n = -N, ..., N. This is the basis in which one can directly
   extract the Fourier coefficients of a single field component by taking every
   fourth entry of the vector.

3. Modal-coefficient bases

   - Isotropic substrate/superstrate port basis:

         [forward_TE(-N..N), forward_TM(-N..N),
          backward_TE(-N..N), backward_TM(-N..N)]^T

   - Internal layer modal basis:

         [all forward layer modes, all backward layer modes]^T

     The layer basis is not TE/TM labeled. It is whatever sorted eigenbasis
     ``Solver.layer_mode_fields(...)`` returns.

The matrices returned by :mod:`rcwa.solver` map modal coefficients into the
harmonic-major reduced-field basis, and this file composes those maps with
interface basis changes, propagation diagonals, and inverse Fourier sums.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from time import perf_counter

import numpy as jnp

from .layer import Layer
from .solver import ScatteringMatrix, Solver
from .stack import Stack


DEFAULT_NUM_POINTS_X = 1024
DEFAULT_NUM_POINTS_Z = 201
DEFAULT_NUM_POINTS_RCWA = 512
SUPPORTED_COMPONENTS = ("-H_y", "H_x", "E_y", "D_x")


def _log(verbose: bool, message: str) -> None:
    """Print a visualization progress message when verbose output is enabled."""
    if verbose:
        print(f"[visualize] {message}")


def _format_elapsed_seconds(elapsed_seconds: float) -> str:
    """Format elapsed wall-clock time for human-readable timing logs."""
    if elapsed_seconds < 1e-3:
        return f"{elapsed_seconds * 1e6:.1f} us"
    if elapsed_seconds < 1.0:
        return f"{elapsed_seconds * 1e3:.1f} ms"
    return f"{elapsed_seconds:.3f} s"


def _log_timing(verbose: bool, label: str, start_time: float) -> None:
    """Print one timing line when verbose output is enabled."""
    if verbose:
        _log(verbose, f"{label}: {_format_elapsed_seconds(perf_counter() - start_time)}")


@dataclass
class _StackModalData:
    """Cache the modal objects used to reconstruct fields inside the stack.

    Attributes
    ----------
    q_matrices_component_major:
        Full layer operators in the component-major basis

            [-H_y(-N..N), H_x(-N..N), E_y(-N..N), D_x(-N..N)]^T.

    q_matrices_harmonic_major:
        The same operators after the similarity transform

            Q_harmonic = P Q_component P^T

        with ``P = Solver.reorder_matrix(N)``. In this basis the rows and
        columns are grouped harmonic-by-harmonic as

            [-H_y(n), H_x(n), E_y(n), D_x(n)].

    substrate_fields, superstrate_fields:
        Modes-to-fields matrices whose columns are isotropic port modes in the
        solver ordering

            [forward_TE(-N..N), forward_TM(-N..N),
             backward_TE(-N..N), backward_TM(-N..N)]

        and whose rows are the harmonic-major reduced-field basis.

    substrate_continuity_fields, superstrate_continuity_fields:
        The corresponding port modal maps after converting rows into the
        harmonic-major tangential-field basis

            [-H_y(n), H_x(n), E_y(n), E_x(n)].

    layer_modes:
        One ``(eigenvalues, layer_fields, layer_continuity_fields)`` tuple per
        physical layer. Each ``layer_fields`` matrix maps layer modal
        coefficients in the ordering

            [all forward layer modes, all backward layer modes]

        into the harmonic-major reduced-field basis, while
        ``layer_continuity_fields`` maps the same modal coefficients into the
        harmonic-major tangential-field basis used only at z-interfaces.
    """
    q_matrices_component_major: list[jnp.ndarray]
    q_matrices_harmonic_major: list[jnp.ndarray]
    substrate_fields: jnp.ndarray
    superstrate_fields: jnp.ndarray
    substrate_continuity_fields: jnp.ndarray
    superstrate_continuity_fields: jnp.ndarray
    layer_modes: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


@dataclass
class LayerProfileData:
    """Precomputed field data for one physical layer and one incident polarization."""

    x_nm: jnp.ndarray
    z_nm: jnp.ndarray
    field_xz: jnp.ndarray
    left_coeffs: jnp.ndarray
    right_coeffs: jnp.ndarray


@dataclass
class IncidentVisualizationData:
    """All stored results for one incident polarization."""

    incident_pol: str
    component: str
    reflected: jnp.ndarray
    transmitted: jnp.ndarray
    layer_profiles: list[LayerProfileData]


@dataclass
class VisualizationBundle:
    """Serializable batch of reflection/transmission data and layer field profiles."""

    wavelength_nm: float
    kappa_inv_nm: complex
    N: int
    num_points_rcwa: int
    num_points_x: int
    num_points_z: int
    polarization_data: dict[str, IncidentVisualizationData]

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Save the bundle as a compressed ``.npz`` archive."""
        out_path = pathlib.Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "polarizations": list(self.polarization_data),
            "components": {
                pol: pol_data.component for pol, pol_data in self.polarization_data.items()
            },
            "num_layers": len(next(iter(self.polarization_data.values())).layer_profiles)
            if self.polarization_data
            else 0,
        }

        arrays: dict[str, jnp.ndarray] = {
            "metadata_json": jnp.asarray(json.dumps(metadata)),
            "wavelength_nm": jnp.asarray(self.wavelength_nm, dtype=jnp.float64),
            "kappa_inv_nm": jnp.asarray(self.kappa_inv_nm, dtype=jnp.complex128),
            "N": jnp.asarray(self.N, dtype=jnp.int64),
            "num_points_rcwa": jnp.asarray(self.num_points_rcwa, dtype=jnp.int64),
            "num_points_x": jnp.asarray(self.num_points_x, dtype=jnp.int64),
            "num_points_z": jnp.asarray(self.num_points_z, dtype=jnp.int64),
        }

        for pol, pol_data in self.polarization_data.items():
            arrays[f"reflected__{pol}"] = pol_data.reflected
            arrays[f"transmitted__{pol}"] = pol_data.transmitted
            for layer_index, layer_profile in enumerate(pol_data.layer_profiles):
                prefix = f"{pol}__layer{layer_index}"
                arrays[f"x_nm__{prefix}"] = layer_profile.x_nm
                arrays[f"z_nm__{prefix}"] = layer_profile.z_nm
                arrays[f"field_xz__{prefix}"] = layer_profile.field_xz
                arrays[f"left_coeffs__{prefix}"] = layer_profile.left_coeffs
                arrays[f"right_coeffs__{prefix}"] = layer_profile.right_coeffs

        jnp.savez_compressed(out_path, **arrays)
        return out_path

    @staticmethod
    def load(path: str | pathlib.Path) -> "VisualizationBundle":
        """Load a bundle previously written by ``VisualizationBundle.save(...)``."""
        in_path = pathlib.Path(path)
        with jnp.load(in_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            polarization_data: dict[str, IncidentVisualizationData] = {}
            for pol in metadata["polarizations"]:
                layer_profiles: list[LayerProfileData] = []
                for layer_index in range(metadata["num_layers"]):
                    prefix = f"{pol}__layer{layer_index}"
                    layer_profiles.append(
                        LayerProfileData(
                            x_nm=data[f"x_nm__{prefix}"],
                            z_nm=data[f"z_nm__{prefix}"],
                            field_xz=data[f"field_xz__{prefix}"],
                            left_coeffs=data[f"left_coeffs__{prefix}"],
                            right_coeffs=data[f"right_coeffs__{prefix}"],
                        )
                    )
                polarization_data[pol] = IncidentVisualizationData(
                    incident_pol=pol,
                    component=metadata["components"][pol],
                    reflected=data[f"reflected__{pol}"],
                    transmitted=data[f"transmitted__{pol}"],
                    layer_profiles=layer_profiles,
                )

            return VisualizationBundle(
                wavelength_nm=float(data["wavelength_nm"]),
                kappa_inv_nm=complex(data["kappa_inv_nm"]),
                N=int(data["N"]),
                num_points_rcwa=int(data["num_points_rcwa"]),
                num_points_x=int(data["num_points_x"]),
                num_points_z=int(data["num_points_z"]),
                polarization_data=polarization_data,
            )

    def incident_data(self, incident_pol: str) -> IncidentVisualizationData:
        """Return the stored data for one incident polarization."""
        pol = incident_pol.upper()
        try:
            return self.polarization_data[pol]
        except KeyError as exc:
            raise ValueError(
                f"Bundle does not contain incident polarization {incident_pol!r}."
            ) from exc


def _symmetric_slab_mode_kappa_inv_nm(
    wavelength_nm: float,
    thickness_nm: float,
    n_core: float,
    n_clad: float,
    pol: str,
    mode_order: int,
) -> float | None:
    """Return beta in 1/nm for one bound mode of a symmetric slab, or None if absent.

    This helper is purely an analytic slab-waveguide estimate. It does not act
    on any RCWA basis and does not perform a basis transformation. It simply
    solves the scalar TE/TM transcendental equation for a symmetric slab and
    returns

        beta = k0 * n_eff

    in the same units used by ``Stack.kappa_inv_nm``.
    """
    if thickness_nm <= 0.0:
        return None
    if n_core <= n_clad:
        return None
    if mode_order < 0:
        raise ValueError(f"mode_order must be nonnegative, got {mode_order}.")

    pol = pol.upper()
    if pol not in {"TE", "TM"}:
        raise ValueError(f"Unsupported polarization {pol!r}. Expected 'TE' or 'TM'.")

    V = jnp.pi * thickness_nm * jnp.sqrt(n_core**2 - n_clad**2) / wavelength_nm
    cutoff = 0.5 * mode_order * jnp.pi
    if V <= cutoff + 1e-12:
        return None

    lower = 0.5 * mode_order * jnp.pi + 1e-10
    upper = min(float(V) - 1e-10, 0.5 * (mode_order + 1) * jnp.pi - 1e-10)
    if lower >= upper:
        return None

    scale = 1.0 if pol == "TE" else (n_core**2 / n_clad**2)

    def dispersion(u: float) -> float:
        w = jnp.sqrt(max(float(V * V - u * u), 0.0))
        if mode_order % 2 == 0:
            return u * jnp.tan(u) - scale * w
        return -u / jnp.tan(u) - scale * w

    grid = jnp.linspace(lower, upper, 4097)
    values = jnp.array([float(dispersion(float(u))) for u in grid])
    bracket_index = None
    for i in range(len(grid) - 1):
        f_left = values[i]
        f_right = values[i + 1]
        if not (jnp.isfinite(f_left) and jnp.isfinite(f_right)):
            continue
        if f_left == 0.0:
            bracket_index = i
            break
        if f_left * f_right < 0.0:
            bracket_index = i
            break

    if bracket_index is None:
        return None

    a = float(grid[bracket_index])
    b = float(grid[bracket_index + 1])
    fa = float(dispersion(a))
    fb = float(dispersion(b))
    if fa == 0.0:
        u = a
    else:
        for _ in range(100):
            mid = 0.5 * (a + b)
            fm = float(dispersion(mid))
            if abs(fm) < 1e-20 or abs(b - a) < 1e-20:
                u = mid
                break
            if fa * fm <= 0.0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        else:
            u = 0.5 * (a + b)

    b_norm = 1.0 - (u / float(V)) ** 2
    n_eff = jnp.sqrt(n_clad**2 + b_norm * (n_core**2 - n_clad**2))
    return float(2 * jnp.pi * n_eff / wavelength_nm)


def _first_supported_symmetric_slab_mode_thickness_nm(
    wavelength_nm: float,
    start_thickness_nm: float,
    n_core: float,
    n_clad: float,
    pol: str,
    mode_order: int,
    thickness_step_nm: float = 10.0,
    max_thickness_nm: float = 2000.0,
) -> tuple[float, float]:
    """Return the first searched thickness that supports the requested slab mode.

    This is another scalar helper with no RCWA basis transformation. It just
    searches thickness until ``_symmetric_slab_mode_kappa_inv_nm(...)`` returns
    a bound propagation constant.
    """
    thickness_nm = start_thickness_nm
    while thickness_nm <= max_thickness_nm + 1e-12:
        kappa_inv_nm = _symmetric_slab_mode_kappa_inv_nm(
            wavelength_nm=wavelength_nm,
            thickness_nm=thickness_nm,
            n_core=n_core,
            n_clad=n_clad,
            pol=pol,
            mode_order=mode_order,
        )
        if kappa_inv_nm is not None:
            return thickness_nm, kappa_inv_nm
        thickness_nm += thickness_step_nm

    raise RuntimeError(
        f"Failed to find a supported {pol.upper()}{mode_order} mode up to {max_thickness_nm} nm."
    )


def _component_index(component: str) -> int:
    """Map a reduced-field component name onto the harmonic-major slot index.

    In the harmonic-major reduced-field basis, each retained harmonic has the
    local ordering

        [-H_y(n), H_x(n), E_y(n), D_x(n)].

    This function returns the local index inside that 4-vector. There is no
    matrix transform here; it is only the bookkeeping map used when slicing the
    harmonic-major field vector.
    """
    component_to_index = {
        "-H_y": 0,
        "H_x": 1,
        "E_y": 2,
        "D_x": 3,
    }
    try:
        return component_to_index[component]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported component {component!r}. Expected one of {SUPPORTED_COMPONENTS}."
        ) from exc


def _incident_coefficients(N: int, incident_pol: str) -> jnp.ndarray:
    """Return the unit incident coefficient vector in the isotropic port basis.

    Basis before and after:
    - input: scalar polarization label ``TE`` or ``TM``
    - output: isotropic substrate modal-coefficient vector in the solver's port
      ordering

          [forward_TE(-N..N), forward_TM(-N..N)]^T

      for the incident side alone.

    Mathematically this constructs

        a_sub^+ = e_j

    where ``j = Solver.zero_order_mode_index(N, incident_pol)`` is the forward
    zero-order TE or TM channel.
    """
    pol = incident_pol.upper()
    half = 2 * Stack.num_harmonics(N)
    inc = jnp.zeros(half, dtype=jnp.complex128)
    inc[Solver.zero_order_mode_index(N, pol)] = 1.0
    return inc


def _default_component_for_incident_pol(incident_pol: str) -> str:
    """Return the default plotted field component for TE or TM incidence."""
    pol = incident_pol.upper()
    if pol == "TE":
        return "E_y"
    if pol == "TM":
        return "-H_y"
    raise ValueError(f"Unknown incident_pol={incident_pol!r}")


def _incident_modal_rt_from_scattering_matrix(
    S11: jnp.ndarray,
    S21: jnp.ndarray,
    N: int,
    incident_pol: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return incident, reflected, and transmitted modal coefficients for one port excitation."""
    inc = _incident_coefficients(N, incident_pol)
    reflected = S11 @ inc
    transmitted = S21 @ inc
    return inc, reflected, transmitted


def _identity_scattering_matrix(num_modes: int) -> ScatteringMatrix:
    """Return the transparent two-port S-matrix on one modal basis."""
    identity = jnp.eye(num_modes, dtype=jnp.complex128)
    zero = jnp.zeros_like(identity)
    return zero, identity, identity, zero


def _layer_scattering_blocks(
    stack: Stack,
    modal_data: _StackModalData,
) -> list[ScatteringMatrix]:
    """Return the interface/propagation S-matrices used by the visualization march."""
    if not modal_data.layer_modes:
        return []

    blocks: list[ScatteringMatrix] = [
        Solver.basis_change_scattering_matrix(
            modal_data.substrate_continuity_fields,
            modal_data.layer_modes[0][2],
        )
    ]

    for i, (eigenvalues, _, layer_continuity_fields) in enumerate(modal_data.layer_modes):
        blocks.append(
            Solver.modal_propagation_scattering_matrix(
                eigenvalues,
                stack.thickness_normalized(i),
            )
        )
        right_fields = (
            modal_data.superstrate_continuity_fields
            if i == len(modal_data.layer_modes) - 1
            else modal_data.layer_modes[i + 1][2]
        )
        blocks.append(
            Solver.basis_change_scattering_matrix(
                layer_continuity_fields,
                right_fields,
            )
        )

    return blocks


def _interface_modal_coefficients(
    prefix_scattering: ScatteringMatrix,
    suffix_scattering: ScatteringMatrix,
    incident_coeffs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the modal coefficients incident on one internal interface from both sides.

    If ``P`` is the S-matrix from the external substrate port up to the
    interface and ``Q`` is the S-matrix from the interface to the external
    superstrate port, then with no illumination from the superstrate side:

        a_k^- = Q11 a_k^+
        a_k^+ = P21 a_0^+ + P22 a_k^-

    Solving this system yields the forward/right-going and backward/left-going
    modal coefficients on the common interface in a numerically stable way that
    avoids propagating the backward evanescent branch with ``exp(+lambda_b d)``.
    """
    _, _, P21, P22 = prefix_scattering
    Q11, _, _, _ = suffix_scattering
    identity = jnp.eye(P22.shape[0], dtype=P22.dtype)
    forward_coeffs = jnp.linalg.solve(identity - P22 @ Q11, P21 @ incident_coeffs)
    backward_coeffs = Q11 @ forward_coeffs
    return forward_coeffs, backward_coeffs


def _march_layer_face_coefficients(
    stack: Stack,
    modal_data: _StackModalData,
    incident_coeffs: jnp.ndarray,
    stop_after_layer_index: int | None,
    verbose: bool,
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Recover per-layer face coefficients using prefix/suffix scattering solves."""
    if not modal_data.layer_modes:
        return []

    total_start = perf_counter()
    blocks = _layer_scattering_blocks(stack, modal_data)
    num_modes = incident_coeffs.shape[0]

    step_start = perf_counter()
    prefix_scattering: list[ScatteringMatrix] = [_identity_scattering_matrix(num_modes)]
    for block in blocks:
        prefix_scattering.append(
            Solver.redheffer_star_product(prefix_scattering[-1], block)
        )

    suffix_scattering: list[ScatteringMatrix] = [
        _identity_scattering_matrix(num_modes) for _ in range(len(blocks) + 1)
    ]
    for i in range(len(blocks) - 1, -1, -1):
        suffix_scattering[i] = Solver.redheffer_star_product(
            blocks[i],
            suffix_scattering[i + 1],
        )
    _log_timing(verbose, "Built prefix/suffix scattering chains for visualization march", step_start)

    layer_face_coeffs: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    for i in range(len(modal_data.layer_modes)):
        left_interface_index = 2 * i + 1
        right_interface_index = left_interface_index + 1

        step_start = perf_counter()
        left_forward, left_backward = _interface_modal_coefficients(
            prefix_scattering[left_interface_index],
            suffix_scattering[left_interface_index],
            incident_coeffs,
        )
        left_elapsed = perf_counter() - step_start

        step_start = perf_counter()
        right_forward, right_backward = _interface_modal_coefficients(
            prefix_scattering[right_interface_index],
            suffix_scattering[right_interface_index],
            incident_coeffs,
        )
        right_elapsed = perf_counter() - step_start

        left_coeffs = jnp.concatenate([left_forward, left_backward])
        right_coeffs = jnp.concatenate([right_forward, right_backward])
        layer_face_coeffs.append((left_coeffs, right_coeffs))

        _log(
            verbose,
            (
                f"Layer march {i}: left-interface solve {_format_elapsed_seconds(left_elapsed)}, "
                f"right-interface solve {_format_elapsed_seconds(right_elapsed)}"
            ),
        )

        if stop_after_layer_index is not None and i >= stop_after_layer_index:
            break

    _log_timing(verbose, "Recovered layer face coefficients with scattering march", total_start)
    return layer_face_coeffs


def _layer_modal_data(
    stack: Stack,
    N: int,
    num_points_rcwa: int,
    verbose: bool,
) -> _StackModalData:
    """Build every modal map needed to reconstruct internal fields.

    This function performs two conceptually separate transformations:

    1. Reorder each layer operator from component-major to harmonic-major:

           Q_harmonic = P Q_component P^T

       where ``P = Solver.reorder_matrix(N)``.

    2. Diagonalize each half-space/layer operator to obtain a matrix whose
       columns are modal field vectors expressed in the harmonic-major
       reduced-field basis.

       For isotropic ports, the columns are ordered as

           [forward_TE(-N..N), forward_TM(-N..N),
            backward_TE(-N..N), backward_TM(-N..N)].

       For internal layers, the columns are ordered as

           [all forward layer modes, all backward layer modes].

    The returned reduced-field matrices are all maps of the form

        field_k = Fields @ modal_coeffs

    with ``field_k`` in harmonic-major reduced-field ordering.

    The cache also stores tangential-field versions of those modal maps for the
    zero-thickness interface basis changes. Those continuity maps act in

        [-H_y(n), H_x(n), E_y(n), E_x(n)]

    ordering, but only at interfaces; all interior reconstruction remains in
    the reduced ``D_x`` basis.
    """
    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Building modal cache for {len(stack.layers)} layer(s): "
            f"N={N}, num_points_rcwa={num_points_rcwa}"
        ),
    )

    step_start = perf_counter()
    q_matrices_component_major = stack.build_all_Q_matrices_normalized(
        N,
        num_points=num_points_rcwa,
    )
    _log_timing(verbose, "Built component-major layer Q matrices", step_start)

    step_start = perf_counter()
    layer_toeplitz_matrices = [
        layer.build_toeplitz_fourier_matrices(N, num_points=num_points_rcwa)
        for layer in stack.layers
    ]
    _log_timing(verbose, "Built layer Toeplitz matrices", step_start)

    step_start = perf_counter()
    q_matrices_harmonic_major = [
        Solver.component_to_harmonic_major(q_matrix)
        for q_matrix in q_matrices_component_major
    ]
    _log_timing(verbose, "Reordered layer Q matrices into harmonic-major basis", step_start)

    step_start = perf_counter()
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=num_points_rcwa),
        N,
    )
    substrate_continuity_fields = Solver.reduced_to_tangential_fields(
        substrate_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_substrate,
            N,
        ),
    )
    superstrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_superstrate_normalized(N, num_points=num_points_rcwa),
        N,
    )
    superstrate_continuity_fields = Solver.reduced_to_tangential_fields(
        superstrate_fields,
        Solver.isotropic_reduced_to_tangential_transform_component_major(
            stack.eps_superstrate,
            N,
        ),
    )
    _log_timing(verbose, "Diagonalized isotropic substrate/superstrate port modes", step_start)

    layer_modes: list[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]] = []
    reference_fields = substrate_fields
    for i, (q_matrix, toeplitz_matrices) in enumerate(
        zip(q_matrices_harmonic_major, layer_toeplitz_matrices)
    ):
        step_start = perf_counter()
        eigenvalues, layer_fields = Solver.layer_mode_fields(
            q_matrix,
            reference_fields=reference_fields,
            verbose=verbose,
        )
        eigensolve_elapsed = perf_counter() - step_start

        step_start = perf_counter()
        layer_continuity_fields = Solver.reduced_to_tangential_fields(
            layer_fields,
            Layer.build_reduced_to_tangential_field_transform_component_major(
                toeplitz_matrices,
                N,
            ),
        )
        continuity_elapsed = perf_counter() - step_start
        _log(
            verbose,
            (
                f"Layer {i}: mode solve {_format_elapsed_seconds(eigensolve_elapsed)}, "
                f"continuity map {_format_elapsed_seconds(continuity_elapsed)}"
            ),
        )
        layer_modes.append((eigenvalues, layer_fields, layer_continuity_fields))
        reference_fields = layer_fields

    modal_data = _StackModalData(
        q_matrices_component_major=q_matrices_component_major,
        q_matrices_harmonic_major=q_matrices_harmonic_major,
        substrate_fields=substrate_fields,
        superstrate_fields=superstrate_fields,
        substrate_continuity_fields=substrate_continuity_fields,
        superstrate_continuity_fields=superstrate_continuity_fields,
        layer_modes=layer_modes,
    )
    _log_timing(verbose, "Built modal cache", total_start)
    return modal_data


def _validate_layer_index(stack: Stack, layer_index: int) -> None:
    """Validate that the requested physical layer exists.

    No basis change happens here; this is only a geometry/index check before the
    code starts constructing modal transformations for that layer.
    """
    if not 0 <= layer_index < len(stack.layers):
        raise IndexError(
            f"layer_index={layer_index} is out of range for a stack with {len(stack.layers)} layers."
        )


def _layer_face_coefficients(
    stack: Stack,
    layer_index: int,
    incident_pol: str,
    N: int,
    num_points_rcwa: int,
    verbose: bool,
) -> tuple[_StackModalData, jnp.ndarray, jnp.ndarray]:
    """Recover the selected layer's modal coefficients on its left and right faces.

    The coefficient march is:

    1. Build the substrate-side incident coefficients

           a_sub^+

       in the isotropic substrate port basis.

    2. Build the per-block scattering matrices for:

       - substrate/layer basis changes,
       - modal propagation inside each layer,
       - layer/layer or layer/superstrate basis changes.

    3. Form prefix and suffix scattering chains up to every internal
       interface, then solve the local interface relation

           a_k^- = Q11 a_k^+
           a_k^+ = P21 a_0^+ + P22 a_k^-

       for the modal coefficients on that interface.

       This keeps the forward/right-going modes referenced from the left and
       the backward/left-going modes referenced from the right, avoiding the
       unstable ``exp(+lambda_b d)`` transfer step for strongly evanescent
       backward modes.

    Stored outputs:
    - ``left_coeffs`` is the selected layer modal coefficient vector at the left
      face in the layer basis

          [all forward layer modes, all backward layer modes].

    - ``right_coeffs`` is the same coefficient vector referenced at the right
      face of that same physical layer.
    """
    _validate_layer_index(stack, layer_index)

    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Recovering face coefficients for layer {layer_index}: "
            f"incident_pol={incident_pol.upper()}, N={N}, num_points_rcwa={num_points_rcwa}"
        ),
    )

    step_start = perf_counter()
    modal_data = _layer_modal_data(
        stack,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Built modal cache for face-coefficient recovery", step_start)

    step_start = perf_counter()
    inc = _incident_coefficients(N, incident_pol)
    _log_timing(verbose, "Built substrate incident coefficient vector", step_start)

    layer_face_coeffs = _march_layer_face_coefficients(
        stack,
        modal_data=modal_data,
        incident_coeffs=inc,
        stop_after_layer_index=layer_index,
        verbose=verbose,
    )
    if len(layer_face_coeffs) <= layer_index:
        raise RuntimeError(f"Failed to recover face coefficients for layer_index={layer_index}.")

    left_coeffs, right_coeffs = layer_face_coeffs[layer_index]
    _log_timing(
        verbose,
        f"Recovered face coefficients for layer {layer_index}",
        total_start,
    )
    return modal_data, left_coeffs, right_coeffs


def _field_k_at_local_depth(
    stack: Stack,
    modal_data: _StackModalData,
    layer_index: int,
    left_coeffs: jnp.ndarray,
    right_coeffs: jnp.ndarray,
    z_nm: float,
) -> jnp.ndarray:
    """Evaluate the harmonic-major reduced-field vector at one local depth.

    Inputs:
    - ``left_coeffs``: layer modal coefficients on the left face in the layer
      basis

          [all forward layer modes, all backward layer modes]

    - ``right_coeffs``: the same layer basis, but referenced at the right face.

    To avoid unnecessary exponential growth, the two halves are referenced from
    opposite faces:

        a^+(z) = exp(lambda_f z) a_L^+
        a^-(z) = exp(-lambda_b (d - z)) a_R^-

    Then the full modal vector inside the layer is

        a(z) = [a^+(z); a^-(z)].

    Finally, the layer modes-to-fields matrix converts from layer modal
    coefficients to harmonic-major reduced-field coefficients:

        field_k(z) = F_layer a(z).

    The returned vector therefore lives in the harmonic-major reduced-field
    basis

        [-H_y(n), H_x(n), E_y(n), D_x(n)]

    stacked over all retained harmonics.
    """
    layer = stack.layers[layer_index]
    if z_nm < -1e-12 or z_nm > layer.thickness_nm + 1e-12:
        raise ValueError(
            f"z_nm={z_nm} lies outside the layer thickness interval [0, {layer.thickness_nm}]."
        )

    z_nm = float(jnp.clip(z_nm, 0.0, layer.thickness_nm))
    z_normalized = 2 * jnp.pi * z_nm / stack.wavelength_nm
    thickness_normalized = stack.thickness_normalized(layer_index)

    eigenvalues, layer_fields, _ = modal_data.layer_modes[layer_index]
    half = eigenvalues.shape[0] // 2
    forward_at_z = jnp.exp(eigenvalues[:half] * z_normalized) * left_coeffs[:half]
    backward_at_z = jnp.exp(
        -eigenvalues[half:] * (thickness_normalized - z_normalized)
    ) * right_coeffs[half:]
    modal_coeffs_at_z = jnp.concatenate([forward_at_z, backward_at_z])
    return layer_fields @ modal_coeffs_at_z


def _field_quantity(field: jnp.ndarray, plot_quantity: str) -> jnp.ndarray:
    """Project a complex field onto the requested plotted scalar quantity.

    No basis transformation occurs here. This simply maps a complex array to its
    real part, imaginary part, or magnitude before plotting.
    """
    plot_quantity = plot_quantity.lower()
    if plot_quantity == "real":
        return jnp.real(field)
    if plot_quantity == "imag":
        return jnp.imag(field)
    if plot_quantity == "abs":
        return jnp.abs(field)
    raise ValueError("plot_quantity must be one of 'real', 'imag', or 'abs'.")


def evaluate_real_space_from_k(
    field_k: jnp.ndarray,
    component: str,
    x_nm: jnp.ndarray,
    x_domain_nm: tuple[float, float],
    kappa_inv_nm: float = 0.0,
) -> jnp.ndarray:
    """Evaluate one reduced-basis field component in real space from harmonic coefficients.

    Basis before and after:
    - input ``field_k`` is a harmonic-major reduced-field vector

          [-H_y(-N), H_x(-N), E_y(-N), D_x(-N),
           ...,
           -H_y(N), H_x(N), E_y(N), D_x(N)]^T

    - output is the chosen scalar field component sampled on the real-space
      ``x`` grid.

    The transform is:

    1. Extract the Fourier coefficients ``c_n`` of the requested component by
       striding through the harmonic-major vector.

    2. Reconstruct the full Bloch field

           f(x) = sum_n c_n exp(i (kappa + 2 pi n / period) (x - x_min)).

    The reconstructed field includes the full Bloch phase

        exp(i * kappa_inv_nm * (x - x_min))

    in addition to the periodic Fourier harmonics.
    """
    field_k = jnp.asarray(field_k, dtype=jnp.complex128)
    x_nm = jnp.asarray(x_nm, dtype=jnp.float64)

    if field_k.ndim != 1 or field_k.shape[0] % 4 != 0:
        raise ValueError(
            "Expected a 1D harmonic-major field vector with length divisible by 4, "
            f"got shape={field_k.shape}."
        )
    if x_nm.ndim != 1:
        raise ValueError(f"Expected x_nm to be 1D, got shape={x_nm.shape}.")

    component_idx = _component_index(component)
    num_h = field_k.shape[0] // 4
    N = (num_h - 1) // 2
    if 2 * N + 1 != num_h:
        raise ValueError(
            "Expected an odd number of retained harmonics in field_k, "
            f"got num_h={num_h}."
        )

    x_min_nm, x_max_nm = x_domain_nm
    period_nm = x_max_nm - x_min_nm
    harmonic_orders = jnp.arange(-N, N + 1)
    coeffs = field_k[component_idx::4]
    phase = 1j * (
        kappa_inv_nm + 2 * jnp.pi * harmonic_orders[None, :] / period_nm
    ) * (x_nm[:, None] - x_min_nm)
    return jnp.sum(coeffs[None, :] * jnp.exp(phase), axis=1)


def _profile_from_face_coefficients(
    stack: Stack,
    modal_data: _StackModalData,
    layer_index: int,
    left_coeffs: jnp.ndarray,
    right_coeffs: jnp.ndarray,
    component: str,
    num_points_x: int,
    num_points_z: int,
    verbose: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reconstruct one ``x-z`` field profile from precomputed layer face coefficients."""
    layer = stack.layers[layer_index]
    x_nm = layer.sample_points(num_points_x)
    z_nm = jnp.linspace(0.0, layer.thickness_nm, num_points_z)

    row_fields: list[jnp.ndarray] = []
    reconstruction_start = perf_counter()
    progress_stride = max(1, num_points_z // 8)
    for row, z_value_nm in enumerate(z_nm):
        row_start = perf_counter()
        row_fields.append(
            evaluate_real_space_from_k(
                _field_k_at_local_depth(
                    stack,
                    modal_data=modal_data,
                    layer_index=layer_index,
                    left_coeffs=left_coeffs,
                    right_coeffs=right_coeffs,
                    z_nm=float(z_value_nm),
                ),
                component=component,
                x_nm=x_nm,
                x_domain_nm=layer.x_domain_nm,
                kappa_inv_nm=stack.kappa_inv_nm,
            )
        )
        if verbose and (
            row == 0
            or row == num_points_z - 1
            or (row + 1) % progress_stride == 0
        ):
            _log(
                verbose,
                (
                    f"Reconstructed z slice {row + 1}/{num_points_z} at z={float(z_value_nm):.6g} nm: "
                    f"{_format_elapsed_seconds(perf_counter() - row_start)}"
                ),
            )

    field_xz = jnp.stack(row_fields, axis=0)
    reconstruction_elapsed = perf_counter() - reconstruction_start
    _log(
        verbose,
        (
            f"Reconstructed {num_points_z} z slices in "
            f"{_format_elapsed_seconds(reconstruction_elapsed)} "
            f"(avg {_format_elapsed_seconds(reconstruction_elapsed / max(num_points_z, 1))} per slice)"
        ),
    )
    return x_nm, z_nm, field_xz


def compute_visualization_bundle(
    stack: Stack,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    incident_polarizations: tuple[str, ...] = ("TE", "TM"),
    components_by_pol: dict[str, str] | None = None,
    cache_path: str | pathlib.Path | None = None,
    verbose: bool = False,
) -> VisualizationBundle:
    """Compute TE/TM reflection/transmission and ``x-z`` profiles in one batch.

    The expensive modal cache and total scattering matrix are each built once.
    For every requested incident polarization, the function stores

    - reflected modal amplitudes ``r``
    - transmitted modal amplitudes ``t``
    - left/right face coefficients for every layer
    - the reconstructed ``x-z`` field profile for one chosen component per polarization

    The resulting bundle can optionally be saved to ``cache_path``.
    """
    total_start = perf_counter()
    normalized_pols = tuple(pol.upper() for pol in incident_polarizations)
    if not normalized_pols:
        raise ValueError("incident_polarizations must contain at least one polarization.")
    if len(set(normalized_pols)) != len(normalized_pols):
        raise ValueError("incident_polarizations must not contain duplicates.")

    resolved_components = {
        pol: (
            components_by_pol[pol]
            if components_by_pol is not None and pol in components_by_pol
            else _default_component_for_incident_pol(pol)
        )
        for pol in normalized_pols
    }

    _log(
        verbose,
        (
            f"Computing visualization bundle: pols={normalized_pols}, N={N}, "
            f"num_points_x={num_points_x}, num_points_z={num_points_z}, "
            f"num_points_rcwa={num_points_rcwa}"
        ),
    )

    step_start = perf_counter()
    modal_data = _layer_modal_data(
        stack,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Built modal cache for visualization bundle", step_start)

    step_start = perf_counter()
    S11, _, S21, _ = Solver.total_scattering_matrix(
        stack,
        N,
        num_points=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Built total scattering matrix for visualization bundle", step_start)

    polarization_data: dict[str, IncidentVisualizationData] = {}
    for pol in normalized_pols:
        pol_start = perf_counter()
        component = resolved_components[pol]
        inc, reflected, transmitted = _incident_modal_rt_from_scattering_matrix(
            S11,
            S21,
            N,
            pol,
        )
        step_start = perf_counter()
        layer_face_coeffs = _march_layer_face_coefficients(
            stack,
            modal_data=modal_data,
            incident_coeffs=inc,
            stop_after_layer_index=None,
            verbose=verbose,
        )
        _log_timing(verbose, f"Marched face coefficients for {pol} incidence", step_start)

        layer_profiles: list[LayerProfileData] = []
        for layer_index, (left_coeffs, right_coeffs) in enumerate(layer_face_coeffs):
            step_start = perf_counter()
            x_nm, z_nm, field_xz = _profile_from_face_coefficients(
                stack,
                modal_data=modal_data,
                layer_index=layer_index,
                left_coeffs=left_coeffs,
                right_coeffs=right_coeffs,
                component=component,
                num_points_x=num_points_x,
                num_points_z=num_points_z,
                verbose=verbose,
            )
            _log_timing(
                verbose,
                f"Built {pol} layer {layer_index} x-z profile for component {component}",
                step_start,
            )
            layer_profiles.append(
                LayerProfileData(
                    x_nm=x_nm,
                    z_nm=z_nm,
                    field_xz=field_xz,
                    left_coeffs=left_coeffs,
                    right_coeffs=right_coeffs,
                )
            )

        polarization_data[pol] = IncidentVisualizationData(
            incident_pol=pol,
            component=component,
            reflected=reflected,
            transmitted=transmitted,
            layer_profiles=layer_profiles,
        )
        _log_timing(verbose, f"Computed {pol} batch data", pol_start)

    bundle = VisualizationBundle(
        wavelength_nm=stack.wavelength_nm,
        kappa_inv_nm=stack.kappa_inv_nm,
        N=N,
        num_points_rcwa=num_points_rcwa,
        num_points_x=num_points_x,
        num_points_z=num_points_z,
        polarization_data=polarization_data,
    )

    if cache_path is not None:
        step_start = perf_counter()
        out_path = bundle.save(cache_path)
        _log(verbose, f"Saved visualization bundle to {out_path}")
        _log_timing(verbose, "Wrote visualization bundle", step_start)

    _log_timing(verbose, "Computed visualization bundle", total_start)
    return bundle


def create_x_line_profile_at_fixed_z(
    stack: Stack,
    layer_index: int,
    incident_pol: str,
    component: str,
    z_nm: float,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return one x-line profile of a reduced-basis field component inside a chosen layer.

    Composition of maps:

    1. Recover the selected layer face coefficients in the layer modal basis.
    2. Propagate them to local depth ``z_nm`` and convert to the harmonic-major
       reduced-field vector

           field_k(z) = F_layer a(z).

    3. Inverse-Fourier transform that harmonic-major vector into the real-space
       scalar line profile of the selected component.

    The returned ``field_x`` is therefore a real-space slice of one component of
    the reduced field basis, sampled across one x-period of the selected layer.
    """
    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Creating x-line profile: layer={layer_index}, incident_pol={incident_pol.upper()}, "
            f"component={component}, z_nm={z_nm:.6g}, N={N}, "
            f"num_points_x={num_points_x}, num_points_rcwa={num_points_rcwa}"
        ),
    )

    step_start = perf_counter()
    modal_data, left_coeffs, right_coeffs = _layer_face_coefficients(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Recovered layer face coefficients", step_start)

    layer = stack.layers[layer_index]

    step_start = perf_counter()
    x_nm = layer.sample_points(num_points_x)
    _log_timing(verbose, "Built x grid", step_start)

    step_start = perf_counter()
    field_k = _field_k_at_local_depth(
        stack,
        modal_data=modal_data,
        layer_index=layer_index,
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        z_nm=z_nm,
    )
    _log_timing(verbose, "Computed harmonic field at requested depth", step_start)

    step_start = perf_counter()
    field_x = evaluate_real_space_from_k(
        field_k,
        component=component,
        x_nm=x_nm,
        x_domain_nm=layer.x_domain_nm,
        kappa_inv_nm=stack.kappa_inv_nm,
    )
    _log_timing(verbose, "Reconstructed real-space x profile", step_start)
    _log_timing(verbose, "Created x-line profile", total_start)
    return x_nm, field_x


def create_xz_profile(
    stack: Stack,
    layer_index: int,
    incident_pol: str,
    component: str,
    N: int,
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return an x-z profile of one reduced-basis field component inside a chosen layer.

    This repeatedly applies the fixed-z reconstruction:

        field_k(z_j) = F_layer a(z_j)
        field_x(x_i, z_j) = sum_n c_n(z_j) exp(i (kappa + G_n) (x_i - x_min)).

    Rows of the returned array correspond to the sampled ``z`` coordinates and
    columns correspond to the sampled ``x`` coordinates.
    """
    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Creating x-z profile: layer={layer_index}, incident_pol={incident_pol.upper()}, "
            f"component={component}, N={N}, num_points_x={num_points_x}, "
            f"num_points_z={num_points_z}, num_points_rcwa={num_points_rcwa}"
        ),
    )

    step_start = perf_counter()
    modal_data, left_coeffs, right_coeffs = _layer_face_coefficients(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Recovered layer face coefficients", step_start)

    step_start = perf_counter()
    x_nm, z_nm, field_xz = _profile_from_face_coefficients(
        stack,
        modal_data=modal_data,
        layer_index=layer_index,
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        component=component,
        num_points_x=num_points_x,
        num_points_z=num_points_z,
        verbose=verbose,
    )
    _log_timing(verbose, "Built x/z grids and reconstructed profile", step_start)
    _log_timing(verbose, "Created x-z profile", total_start)
    return x_nm, z_nm, field_xz


def create_xz_profile_from_bundle(
    bundle: VisualizationBundle,
    incident_pol: str,
    layer_index: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a precomputed ``x-z`` profile from a stored visualization bundle."""
    pol_data = bundle.incident_data(incident_pol)
    try:
        layer_profile = pol_data.layer_profiles[layer_index]
    except IndexError as exc:
        raise IndexError(
            f"layer_index={layer_index} is out of range for a bundle with "
            f"{len(pol_data.layer_profiles)} layers."
        ) from exc
    return layer_profile.x_nm, layer_profile.z_nm, layer_profile.field_xz


def create_x_line_profile_at_fixed_z_from_bundle(
    bundle: VisualizationBundle,
    incident_pol: str,
    layer_index: int,
    z_nm: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a line cut from the nearest stored ``z`` slice in a visualization bundle."""
    x_nm, z_grid_nm, field_xz = create_xz_profile_from_bundle(
        bundle,
        incident_pol=incident_pol,
        layer_index=layer_index,
    )
    z_idx = int(jnp.argmin(jnp.abs(z_grid_nm - z_nm)))
    return x_nm, field_xz[z_idx]


def plot_x_line_profile_at_fixed_z(
    stack: Stack,
    layer_index: int,
    incident_pol: str,
    component: str,
    z_nm: float,
    N: int,
    plot_quantity: str = "real",
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
):
    """Plot one x-line profile of a chosen reduced-basis field component.

    No additional basis transformation is introduced here. This is just the
    visualization wrapper around ``create_x_line_profile_at_fixed_z(...)``.
    """
    import matplotlib.pyplot as plt

    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Plotting x-line profile: layer={layer_index}, incident_pol={incident_pol.upper()}, "
            f"component={component}, z_nm={z_nm:.6g}, plot_quantity={plot_quantity}"
        ),
    )

    step_start = perf_counter()
    x_nm, field_x = create_x_line_profile_at_fixed_z(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        component=component,
        z_nm=z_nm,
        N=N,
        num_points_x=num_points_x,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Prepared x-line profile data", step_start)

    step_start = perf_counter()
    values = _field_quantity(field_x, plot_quantity)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_nm, values, lw=1.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(f"{plot_quantity}({component})")
    ax.set_title(
        f"{incident_pol.upper()} incidence, layer {layer_index}, z = {z_nm:.3f} nm"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _log_timing(verbose, "Built x-line plot", step_start)
    _log_timing(verbose, "Plotted x-line profile", total_start)
    return fig, ax


def plot_xz_profile(
    stack: Stack,
    layer_index: int,
    incident_pol: str,
    component: str,
    N: int,
    plot_quantity: str = "real",
    num_points_x: int = DEFAULT_NUM_POINTS_X,
    num_points_z: int = DEFAULT_NUM_POINTS_Z,
    num_points_rcwa: int = DEFAULT_NUM_POINTS_RCWA,
    verbose: bool = False,
):
    """Plot an x-z image of one reduced-basis field component inside a chosen layer.

    No additional basis transformation is introduced here. This is just the
    visualization wrapper around ``create_xz_profile(...)``.
    """
    import matplotlib.pyplot as plt

    total_start = perf_counter()
    _log(
        verbose,
        (
            f"Plotting x-z profile: layer={layer_index}, incident_pol={incident_pol.upper()}, "
            f"component={component}, plot_quantity={plot_quantity}"
        ),
    )

    step_start = perf_counter()
    x_nm, z_nm, field_xz = create_xz_profile(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        component=component,
        N=N,
        num_points_x=num_points_x,
        num_points_z=num_points_z,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    _log_timing(verbose, "Prepared x-z profile data", step_start)

    step_start = perf_counter()
    values = _field_quantity(field_xz, plot_quantity)
    layer = stack.layers[layer_index]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(
            layer.x_domain_nm[0],
            layer.x_domain_nm[1],
            z_nm[0],
            z_nm[-1],
        ),
    )
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Local z in layer (nm)")
    ax.set_title(
        f"{plot_quantity}({component}) for {incident_pol.upper()} incidence in layer {layer_index}"
    )
    fig.colorbar(image, ax=ax, label=f"{plot_quantity}({component})")
    fig.tight_layout()
    _log_timing(verbose, "Built x-z plot", step_start)
    _log_timing(verbose, "Plotted x-z profile", total_start)
    return fig, ax


def plot_x_line_profile_at_fixed_z_from_bundle(
    bundle: VisualizationBundle,
    incident_pol: str,
    layer_index: int,
    z_nm: float,
    plot_quantity: str = "real",
):
    """Plot a precomputed line cut from a visualization bundle."""
    import matplotlib.pyplot as plt

    pol_data = bundle.incident_data(incident_pol)
    x_nm, field_x = create_x_line_profile_at_fixed_z_from_bundle(
        bundle,
        incident_pol=incident_pol,
        layer_index=layer_index,
        z_nm=z_nm,
    )
    values = _field_quantity(field_x, plot_quantity)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x_nm, values, lw=1.5)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(f"{plot_quantity}({pol_data.component})")
    ax.set_title(
        f"{incident_pol.upper()} incidence, layer {layer_index}, nearest stored z to {z_nm:.3f} nm"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_xz_profile_from_bundle(
    bundle: VisualizationBundle,
    incident_pol: str,
    layer_index: int,
    plot_quantity: str = "real",
):
    """Plot a precomputed ``x-z`` profile from a visualization bundle."""
    import matplotlib.pyplot as plt

    pol_data = bundle.incident_data(incident_pol)
    x_nm, z_nm, field_xz = create_xz_profile_from_bundle(
        bundle,
        incident_pol=incident_pol,
        layer_index=layer_index,
    )
    values = _field_quantity(field_xz, plot_quantity)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    image = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(
            x_nm[0],
            x_nm[-1],
            z_nm[0],
            z_nm[-1],
        ),
    )
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("Local z in layer (nm)")
    ax.set_title(
        f"{plot_quantity}({pol_data.component}) for {incident_pol.upper()} incidence in layer {layer_index}"
    )
    fig.colorbar(image, ax=ax, label=f"{plot_quantity}({pol_data.component})")
    fig.tight_layout()
    return fig, ax
