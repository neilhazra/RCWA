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

from dataclasses import dataclass

import numpy as jnp

from rcwa import Layer, Solver, Stack


DEFAULT_NUM_POINTS_X = 1024
DEFAULT_NUM_POINTS_Z = 201
DEFAULT_NUM_POINTS_RCWA = 512
SUPPORTED_COMPONENTS = ("-H_y", "H_x", "E_y", "D_x")


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

    layer_modes:
        One ``(eigenvalues, layer_fields)`` tuple per physical layer. Each
        ``layer_fields`` matrix maps layer modal coefficients in the ordering

            [all forward layer modes, all backward layer modes]

        into the harmonic-major reduced-field basis.
    """
    q_matrices_component_major: list[jnp.ndarray]
    q_matrices_harmonic_major: list[jnp.ndarray]
    substrate_fields: jnp.ndarray
    superstrate_fields: jnp.ndarray
    layer_modes: list[tuple[jnp.ndarray, jnp.ndarray]]


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

    The returned matrices are therefore all maps of the form

        field_k = Fields @ modal_coeffs

    with ``field_k`` in harmonic-major reduced-field ordering.
    """
    q_matrices_component_major = stack.build_all_Q_matrices_normalized(
        N,
        num_points=num_points_rcwa,
    )
    q_matrices_harmonic_major = [
        Solver.component_to_harmonic_major(q_matrix)
        for q_matrix in q_matrices_component_major
    ]
    substrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_substrate_normalized(N, num_points=num_points_rcwa),
        N,
    )
    superstrate_fields = Solver.isotropic_mode_fields(
        stack.get_Q_superstrate_normalized(N, num_points=num_points_rcwa),
        N,
    )

    layer_modes: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    reference_fields = substrate_fields
    for q_matrix in q_matrices_harmonic_major:
        mode_data = Solver.layer_mode_fields(
            q_matrix,
            reference_fields=reference_fields,
            verbose=verbose,
        )
        layer_modes.append(mode_data)
        reference_fields = mode_data[1]

    return _StackModalData(
        q_matrices_component_major=q_matrices_component_major,
        q_matrices_harmonic_major=q_matrices_harmonic_major,
        substrate_fields=substrate_fields,
        superstrate_fields=superstrate_fields,
        layer_modes=layer_modes,
    )


def _modal_propagation_transfer_matrix(
    eigenvalues: jnp.ndarray,
    distance_normalized: float,
) -> jnp.ndarray:
    """Return the left-to-right propagation transfer matrix inside one layer.

    Basis before and after:
    - input coefficients: layer modal basis at the left face

          [a_L^+; a_L^-]

      where the first half are forward/right-going layer modes and the second
      half are backward/left-going layer modes.

    - output coefficients: the same layer modal basis, but referenced at the
      right face

          [a_R^+; a_R^-].

    The transfer matrix is diagonal because propagation does not mix modes
    inside a uniform layer once that layer has been diagonalized:

        [a_R^+]   [exp(lambda_f d)   0          ] [a_L^+]
        [a_R^-] = [0                 exp(lambda_b d)] [a_L^-]

    Here ``lambda_f`` are the forward eigenvalues and ``lambda_b`` are the
    backward eigenvalues in the sorted layer modal basis.
    """
    if eigenvalues.ndim != 1 or eigenvalues.shape[0] % 2 != 0:
        raise ValueError(
            "Expected a 1D even-length eigenvalue array, "
            f"got shape={eigenvalues.shape}."
        )

    half = eigenvalues.shape[0] // 2
    forward = jnp.exp(eigenvalues[:half] * distance_normalized)
    backward = jnp.exp(eigenvalues[half:] * distance_normalized)
    return jnp.diag(jnp.concatenate([forward, backward]))


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

    2. Use the stack S-matrix to recover the reflected coefficients

           a_sub^- = S11 a_sub^+.

    3. Concatenate them into the full substrate modal vector

           a_sub = [a_sub^+; a_sub^-].

    4. For each interface, change basis with

           a_right_basis = F_right^{-1} F_left a_left_basis

       where ``F_left`` and ``F_right`` are the modes-to-fields matrices whose
       columns are modal fields expressed in the common harmonic-major
       reduced-field basis.

    5. Inside each layer, propagate with the diagonal transfer matrix from
       ``_modal_propagation_transfer_matrix(...)``.

    Stored outputs:
    - ``left_coeffs`` is the selected layer modal coefficient vector at the left
      face in the layer basis

          [all forward layer modes, all backward layer modes].

    - ``right_coeffs`` is the same coefficient vector referenced at the right
      face of that same physical layer.
    """
    _validate_layer_index(stack, layer_index)

    modal_data = _layer_modal_data(
        stack,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    S11, _, _, _ = Solver.total_scattering_matrix(
        stack,
        N,
        num_points=num_points_rcwa,
        verbose=verbose,
    )
    inc = _incident_coefficients(N, incident_pol)
    reflected = S11 @ inc
    substrate_coeffs = jnp.concatenate([inc, reflected])

    current_right_coeffs: jnp.ndarray | None = None
    for i, (eigenvalues, layer_fields) in enumerate(modal_data.layer_modes):
        if i == 0:
            left_coeffs = Solver.basis_change_transfer_matrix(
                modal_data.substrate_fields,
                layer_fields,
            ) @ substrate_coeffs
        else:
            previous_fields = modal_data.layer_modes[i - 1][1]
            left_coeffs = Solver.basis_change_transfer_matrix(
                previous_fields,
                layer_fields,
            ) @ current_right_coeffs

        right_coeffs = (
            _modal_propagation_transfer_matrix(
                eigenvalues,
                stack.thickness_normalized(i),
            )
            @ left_coeffs
        )

        if i == layer_index:
            return modal_data, left_coeffs, right_coeffs

        current_right_coeffs = right_coeffs

    raise RuntimeError(f"Failed to recover face coefficients for layer_index={layer_index}.")


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

    eigenvalues, layer_fields = modal_data.layer_modes[layer_index]
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
    modal_data, left_coeffs, right_coeffs = _layer_face_coefficients(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    layer = stack.layers[layer_index]
    x_nm = layer.sample_points(num_points_x)
    field_k = _field_k_at_local_depth(
        stack,
        modal_data=modal_data,
        layer_index=layer_index,
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        z_nm=z_nm,
    )
    field_x = evaluate_real_space_from_k(
        field_k,
        component=component,
        x_nm=x_nm,
        x_domain_nm=layer.x_domain_nm,
        kappa_inv_nm=stack.kappa_inv_nm,
    )
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
    modal_data, left_coeffs, right_coeffs = _layer_face_coefficients(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=verbose,
    )
    layer = stack.layers[layer_index]
    x_nm = layer.sample_points(num_points_x)
    z_nm = jnp.linspace(0.0, layer.thickness_nm, num_points_z)

    field_xz = jnp.stack(
        [
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
            for z_value_nm in z_nm
        ],
        axis=0,
    )
    return x_nm, z_nm, field_xz


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
    return fig, ax


def main() -> None:
    """Run a homogeneous-slab demo using the helpers defined above.

    The demo itself does not introduce any new basis transformations. It simply
    builds a stack, chooses an incident polarization/component, and calls the
    plotting helpers, which perform the modal-to-field and Fourier
    reconstruction steps documented above.
    """
    import matplotlib.pyplot as plt

    wavelength_nm = 633.0
    n_core = 1.5
    n_clad = 1.0
    initial_thickness_nm = 120.0
    te1_thickness_nm, te1_kappa_inv_nm = _first_supported_symmetric_slab_mode_thickness_nm(
        wavelength_nm=wavelength_nm,
        start_thickness_nm=initial_thickness_nm,
        n_core=n_core,
        n_clad=n_clad,
        pol="TM",
        mode_order=0,
    )

    def make_stack() -> Stack:
        stack = Stack(
            wavelength_nm=wavelength_nm,
            kappa_inv_nm=te1_kappa_inv_nm,
            eps_substrate=n_clad**2,
            eps_superstrate=n_clad**2,
        )
        stack.add_layer(
            Layer.uniform(
                thickness_nm=te1_thickness_nm,
                eps_tensor=n_core**2 * jnp.eye(3, dtype=jnp.complex128),
                x_domain_nm=(0.0, 5000.0),
            )
        )
        return stack

    stack = make_stack()
    layer_index = 0
    incident_pol = "TM"
    component = "-H_y"
    N = 4
    z_nm = 0.25 * stack.layers[layer_index].thickness_nm

    print(
        "Homogeneous slab TE1 visualization\n"
        f"initial thickness = {initial_thickness_nm:.1f} nm, "
        f"chosen thickness = {te1_thickness_nm:.1f} nm, "
        f"kappa_inv_nm = {te1_kappa_inv_nm:.6f}"
    )

    plot_x_line_profile_at_fixed_z(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        component=component,
        z_nm=z_nm,
        N=N,
        plot_quantity="abs",
    )
    plot_xz_profile(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        component=component,
        N=N,
        plot_quantity="abs",
    )
    plt.show()


if __name__ == "__main__":
    main()
