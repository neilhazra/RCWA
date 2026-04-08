"""Finite grating coupler modeled with a 20 um RCWA supercell.

The local grating pitch stays the same as in ``grating_coupler_convergence.py``,
but only a 3 um section of the supercell is patterned. The rest of the top
layer is uniform superstrate, so the geometry behaves like a finite grating
written on top of a slab waveguide.

This is a supercell model. Because the feature pitch is much smaller than the
20 um computational period, it generally needs much larger Fourier truncations
than the infinite-periodic case to converge quantitatively.
"""

from __future__ import annotations

import numpy as jnp
from rcwa import Layer, Stack
from rcwa.solver import Solver

n_wg = 2.0
n_sub = 1.0
n_sup = 1.0
local_grating_period_nm = 399.8
supercell_period_nm = 30_000.0
grating_length_nm = 6_000.0
d_slab_nm = 120.0
d_grating_nm = 60.0
duty_cycle = 0.5
design_wl_nm = 633.0
sweep_half_span_nm = 40.0
sweep_num_samples = 81
geometry_fourier_order = 256
default_geometry_num_points = 8192
default_num_points = 1024
default_design_N = 16
default_sweep_N = 12
max_reasonable_dense_N = 96


def _merge_adjacent_segments(
    segments: list[tuple[float, float, jnp.ndarray]],
    tol_nm: float = 1e-9,
) -> list[tuple[float, float, jnp.ndarray]]:
    if not segments:
        return []

    merged = [segments[0]]
    for start_nm, end_nm, eps_tensor in segments[1:]:
        prev_start_nm, prev_end_nm, prev_eps_tensor = merged[-1]
        same_eps = jnp.allclose(prev_eps_tensor, eps_tensor, atol=1e-12, rtol=1e-12)
        touching = abs(start_nm - prev_end_nm) < tol_nm
        if same_eps and touching:
            merged[-1] = (prev_start_nm, end_nm, prev_eps_tensor)
        else:
            merged.append((start_nm, end_nm, eps_tensor))
    return merged


def build_finite_grating_segments() -> list[tuple[float, float, jnp.ndarray]]:
    if grating_length_nm > supercell_period_nm:
        raise ValueError("grating_length_nm must not exceed supercell_period_nm")

    eps_wg = n_wg**2 * jnp.eye(3, dtype=jnp.complex128)
    eps_sup = n_sup**2 * jnp.eye(3, dtype=jnp.complex128)

    grating_start_nm = 0.5 * (supercell_period_nm - grating_length_nm)
    grating_end_nm = grating_start_nm + grating_length_nm
    fill_width_nm = duty_cycle * local_grating_period_nm

    segments: list[tuple[float, float, jnp.ndarray]] = []
    if grating_start_nm > 0.0:
        segments.append((0.0, grating_start_nm, eps_sup))

    cell_start_nm = grating_start_nm
    while cell_start_nm < grating_end_nm - 1e-9:
        cell_end_nm = min(cell_start_nm + local_grating_period_nm, grating_end_nm)
        ridge_end_nm = min(cell_start_nm + fill_width_nm, cell_end_nm)

        if ridge_end_nm > cell_start_nm + 1e-9:
            segments.append((cell_start_nm, ridge_end_nm, eps_wg))
        if cell_end_nm > ridge_end_nm + 1e-9:
            segments.append((ridge_end_nm, cell_end_nm, eps_sup))

        cell_start_nm = cell_end_nm

    if grating_end_nm < supercell_period_nm:
        segments.append((grating_end_nm, supercell_period_nm, eps_sup))

    return _merge_adjacent_segments(segments)


def make_stack(wavelength_nm: float) -> Stack:
    x_domain_nm = (0.0, supercell_period_nm)
    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_sub**2,
        eps_superstrate=n_sup**2,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=d_slab_nm,
            eps_tensor=n_wg**2 * jnp.eye(3, dtype=jnp.complex128),
            x_domain_nm=x_domain_nm,
        )
    )
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=d_grating_nm,
            x_domain_nm=x_domain_nm,
            segments=build_finite_grating_segments(),
        )
    )
    return stack


def _centered_fft_coefficients(values: jnp.ndarray, max_order: int) -> jnp.ndarray:
    num_points = values.shape[0]
    fft_vals = jnp.fft.fft(values) / num_points
    positive_orders = fft_vals[: max_order + 1]
    negative_orders = fft_vals[-max_order:]
    return jnp.concatenate([negative_orders, positive_orders])




def _reconstruct_from_centered_coeffs(
    coeffs: jnp.ndarray,
    x_nm: jnp.ndarray,
    x_domain_nm: tuple[float, float],
) -> jnp.ndarray:
    x_min_nm, x_max_nm = x_domain_nm
    period_nm = x_max_nm - x_min_nm
    max_order = (coeffs.shape[0] - 1) // 2
    orders = jnp.arange(-max_order, max_order + 1)
    phase = 2j * jnp.pi * (x_nm[:, None] - x_min_nm) * orders[None, :] / period_nm
    return jnp.sum(coeffs[None, :] * jnp.exp(phase), axis=1)


def get_zero_order_field_rt(
    stack: Stack,
    N: int,
    pol: str = "TE",
    num_points: int = 4096,
) -> tuple[complex, complex]:
    pol = pol.upper()
    reflected, transmitted = Solver.reflection_transmission(
        stack,
        N,
        incident_pol=pol,
        num_points=num_points,
    )
    Q_sub = stack.get_Q_substrate_normalized(N, num_points=num_points)
    Q_sup = stack.get_Q_superstrate_normalized(N, num_points=num_points)
    _, evecs_sub = Solver.diagonalize_sort_isotropic_modes(Q_sub)
    _, evecs_sup = Solver.diagonalize_sort_isotropic_modes(Q_sup)
    h = Stack.zero_harmonic_index(N)

    if pol == "TE":
        E_sub_fwd = evecs_sub[h, 2, 0]
        E_sub_bwd = evecs_sub[h, 2, 2]
        E_sup_fwd = evecs_sup[h, 2, 0]
        zero_mode = Solver.zero_order_mode_index(N, "TE")
    elif pol == "TM":
        E_sub_fwd = evecs_sub[h, 3, 1]
        E_sub_bwd = evecs_sub[h, 3, 3]
        E_sup_fwd = evecs_sup[h, 3, 1]
        zero_mode = Solver.zero_order_mode_index(N, "TM")
    else:
        raise ValueError(f"Unknown polarization {pol!r}")

    r0 = reflected[zero_mode] * (E_sub_bwd / E_sub_fwd)
    t0 = transmitted[zero_mode] * (E_sup_fwd / E_sub_fwd)
    return r0, t0


def _validate_dense_truncation(N: int) -> None:
    if N > max_reasonable_dense_N:
        matrix_size = 4 * Stack.num_harmonics(N)
        raise ValueError(
            f"N={N} implies dense layer matrices of size {matrix_size}x{matrix_size}, "
            "which is not a reasonable default for this dense RCWA script. "
            f"Use N <= {max_reasonable_dense_N} unless you are deliberately running a very large solve."
        )


def solve_zero_order_field_rt(
    stack: Stack,
    N: int,
    num_points: int = default_num_points,
) -> dict[str, tuple[complex, complex]]:
    _validate_dense_truncation(N)

    S11, _, S21, _ = Solver.total_scattering_matrix(stack, N, num_points=num_points)
    half = 2 * Stack.num_harmonics(N)
    Q_sub = stack.get_Q_substrate_normalized(N, num_points=num_points)
    Q_sup = stack.get_Q_superstrate_normalized(N, num_points=num_points)
    _, evecs_sub = Solver.diagonalize_sort_isotropic_modes(Q_sub)
    _, evecs_sup = Solver.diagonalize_sort_isotropic_modes(Q_sup)
    h = Stack.zero_harmonic_index(N)

    results: dict[str, tuple[complex, complex]] = {}
    for pol, sub_cols, sup_col in [
        ("TE", (2, 0, 2), 2),
        ("TM", (3, 1, 3), 3),
    ]:
        inc = jnp.zeros(half, dtype=jnp.complex128)
        zero_mode = Solver.zero_order_mode_index(N, pol)
        inc[zero_mode] = 1.0
        reflected = S11 @ inc
        transmitted = S21 @ inc

        field_row, fwd_col, bwd_col = sub_cols
        E_sub_fwd = evecs_sub[h, field_row, fwd_col]
        E_sub_bwd = evecs_sub[h, field_row, bwd_col]
        E_sup_fwd = evecs_sup[h, field_row, fwd_col]

        r0 = reflected[zero_mode] * (E_sub_bwd / E_sub_fwd)
        t0 = transmitted[zero_mode] * (E_sup_fwd / E_sub_fwd)
        results[pol] = (r0, t0)

    return results


def transmission_power_scale(pol: str) -> float:
    pol = pol.upper()
    if pol == "TE":
        return n_sup / n_sub
    if pol == "TM":
        return n_sub / n_sup
    raise ValueError(f"Unknown polarization {pol!r}")


def get_zero_order_power_rt(
    stack: Stack,
    N: int,
    pol: str = "TE",
    num_points: int = default_num_points,
) -> tuple[float, float]:
    r0, t0 = get_zero_order_field_rt(stack, N=N, pol=pol, num_points=num_points)
    R0 = jnp.abs(r0) ** 2
    T0 = jnp.abs(t0) ** 2 * transmission_power_scale(pol)
    return R0, T0


def sweep_wavelength_response_all_pols(
    wavelengths_nm: jnp.ndarray,
    N: int,
    num_points: int = default_num_points,
) -> dict[str, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    _validate_dense_truncation(N)
    traces = {
        "TE": {"R": [], "T": [], "residual": []},
        "TM": {"R": [], "T": [], "residual": []},
    }

    for wavelength_nm in wavelengths_nm:
        stack = make_stack(float(wavelength_nm))
        rt_fields = solve_zero_order_field_rt(stack, N=N, num_points=num_points)
        for pol in ["TE", "TM"]:
            r0, t0 = rt_fields[pol]
            R0 = jnp.abs(r0) ** 2
            T0 = jnp.abs(t0) ** 2 * transmission_power_scale(pol)
            nonzero_order_power = jnp.maximum(0.0, 1.0 - R0 - T0)

            traces[pol]["R"].append(float(R0))
            traces[pol]["T"].append(float(T0))
            traces[pol]["residual"].append(float(nonzero_order_power))

    return {
        pol: (
            jnp.array(pol_traces["R"]),
            jnp.array(pol_traces["T"]),
            jnp.array(pol_traces["residual"]),
        )
        for pol, pol_traces in traces.items()
    }


def sweep_wavelength_response(
    wavelengths_nm: jnp.ndarray,
    N: int,
    pol: str = "TE",
    num_points: int = default_num_points,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return sweep_wavelength_response_all_pols(
        wavelengths_nm,
        N=N,
        num_points=num_points,
    )[pol.upper()]


def sample_grating_profile(
    num_points: int = default_geometry_num_points,
    fourier_order: int = geometry_fourier_order,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    stack = make_stack(design_wl_nm)
    grating_layer = stack.layers[1]
    x_nm = grating_layer.sample_points(num_points)
    eps_xx = grating_layer.sample_eps(num_points)[:, 0, 0]
    coeffs = _centered_fft_coefficients(eps_xx.real, fourier_order)
    eps_xx_reconstructed = _reconstruct_from_centered_coeffs(
        coeffs,
        x_nm,
        grating_layer.x_domain_nm,
    ).real
    return x_nm, eps_xx.real, eps_xx_reconstructed


def plot_geometry(
    num_points: int = default_geometry_num_points,
    fourier_order: int = geometry_fourier_order,
) -> None:
    import matplotlib.pyplot as plt

    x_nm, eps_xx, eps_xx_reconstructed = sample_grating_profile(
        num_points=num_points,
        fourier_order=fourier_order,
    )
    grating_start_nm = 0.5 * (supercell_period_nm - grating_length_nm)
    grating_end_nm = grating_start_nm + grating_length_nm

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x_nm / 1000.0, eps_xx, color="tab:blue", lw=1.5, label=r"Exact Re[$\epsilon_{xx}$]")
    ax.plot(
        x_nm / 1000.0,
        eps_xx_reconstructed,
        color="tab:red",
        lw=1.2,
        linestyle="--",
        label=f"Fourier reconstruction (|n| <= {fourier_order})",
    )
    ax.axvspan(
        grating_start_nm / 1000.0,
        grating_end_nm / 1000.0,
        color="tab:orange",
        alpha=0.15,
        label=f"Patterned {grating_length_nm / 1000.0:.1f} um region",
    )
    ax.set_xlabel("x (um)")
    ax.set_ylabel(r"Re[$\epsilon_{xx}$]")
    ax.set_title(
        "Finite Grating Supercell Profile\n"
        f"{grating_length_nm / 1000.0:.1f} um patterned region inside a "
        f"{supercell_period_nm / 1000.0:.1f} um RCWA supercell"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def run_design_point_demo(
    N: int = default_design_N,
    num_points: int = default_num_points,
) -> None:
    _validate_dense_truncation(N)
    stack = make_stack(design_wl_nm)
    print(
        "Finite supercell grating coupler\n"
        f"design wavelength = {design_wl_nm:.1f} nm, local period = {local_grating_period_nm:.1f} nm, "
        f"patterned length = {grating_length_nm / 1000.0:.1f} um, supercell = {supercell_period_nm / 1000.0:.1f} um"
    )
    print(
        "Zero-order amplitudes are reported for debugging only. In this large-period "
        "supercell, power can scatter into many open diffraction orders, so zero-order "
        "R/T does not represent the full coupling budget."
    )

    for pol, (r0, t0) in solve_zero_order_field_rt(stack, N=N, num_points=num_points).items():
        print(
            f"{pol}: r0 = {complex(r0):.6g}, t0 = {complex(t0):.6g}, "
            f"|r0|^2 = {float(jnp.abs(r0) ** 2):.6f}, |t0|^2 = {float(jnp.abs(t0) ** 2):.6f}"
        )


def plot_wavelength_sweep(
    center_wavelength_nm: float = design_wl_nm,
    half_span_nm: float = sweep_half_span_nm,
    num_samples: int = sweep_num_samples,
    N: int = default_sweep_N,
    num_points: int = default_num_points,
) -> None:
    import matplotlib.pyplot as plt
    _validate_dense_truncation(N)

    wavelengths_nm = jnp.linspace(
        center_wavelength_nm - half_span_nm,
        center_wavelength_nm + half_span_nm,
        num_samples,
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    traces = sweep_wavelength_response_all_pols(
        wavelengths_nm,
        N=N,
        num_points=num_points,
    )

    for ax, pol in zip(axes, ["TE", "TM"]):
        reflected, transmitted, residual = traces[pol]
        peak_index = int(jnp.argmax(residual))
        peak_wavelength_nm = float(wavelengths_nm[peak_index])
        peak_residual = float(residual[peak_index])

        print(
            f"{pol}: max non-zero-order power = {peak_residual:.4f} at "
            f"{peak_wavelength_nm:.1f} nm"
        )

        ax.plot(wavelengths_nm, reflected, label="R0")
        ax.plot(wavelengths_nm, transmitted, label="T0")
        ax.plot(wavelengths_nm, residual, label="1 - R0 - T0")
        ax.axvline(center_wavelength_nm, color="k", linestyle=":", alpha=0.4, label="Design")
        ax.set_ylabel("Power")
        ax.set_title(f"{pol} incidence")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Finite Grating Supercell Wavelength Sweep\n"
        "Here 1 - R0 - T0 is the power leaving the zero-order channels, not a pure guided-mode metric."
    )
    fig.tight_layout()


def main() -> None:
    import matplotlib.pyplot as plt

    plot_geometry()
    run_design_point_demo()
    #plot_wavelength_sweep()
    plt.show()


if __name__ == "__main__":
    main()
