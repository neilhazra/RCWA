"""Visualize design-point fields for the finite grating coupler supercell."""

from __future__ import annotations
import pathlib
import sys
import numpy as jnp

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcwa import Layer, Solver, Stack
from rcwa.visualize import (
    VisualizationBundle,
    compute_visualization_bundle,
    plot_xz_profile_from_bundle,
)


N = 512
VERBOSE = True
NUM_POINTS_RCWA = 4096*2
NUM_POINTS_X = 1400*4
NUM_POINTS_Z = 161
PLOT_QUANTITY = "abs"
INCIDENT_POLS = ("TE", "TM")
LOAD_BUNDLE_FROM_CACHE_IF_PRESENT = True
CACHE_PATH = pathlib.Path(__file__).with_name("grating_coupler_visualize_cache.npz")

n_wg = 2.0
n_sub = 1.0
n_sup = 1.0
local_grating_period_nm = 399.8
supercell_period_nm = 25_000.0
grating_length_nm = 2_000.0
d_slab_nm = 120.0
d_grating_nm = 60.0
duty_cycle = 0.5
design_wl_nm = 633.0
geometry_fourier_order = N
geometry_num_points = 4096*2


def _layer_name(layer_index: int) -> str:
    names = ("slab", "grating")
    if 0 <= layer_index < len(names):
        return names[layer_index]
    return f"layer {layer_index}"


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


def sample_grating_profile(
    num_points: int = geometry_num_points,
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
    num_points: int = geometry_num_points,
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
def main() -> None:
    import matplotlib.pyplot as plt

    stack = make_stack(design_wl_nm)
    plot_geometry()
    plt.show()
    if LOAD_BUNDLE_FROM_CACHE_IF_PRESENT and CACHE_PATH.exists():
        bundle = VisualizationBundle.load(CACHE_PATH)
        print(f"Loaded visualization bundle from {CACHE_PATH}")
    else:
        bundle = compute_visualization_bundle(
            stack,
            N=N,
            num_points_x=NUM_POINTS_X,
            num_points_z=NUM_POINTS_Z,
            num_points_rcwa=NUM_POINTS_RCWA,
            incident_polarizations=INCIDENT_POLS,
            cache_path=CACHE_PATH,
            verbose=VERBOSE,
        )

    print(
        "Finite grating coupler field visualization\n"
        f"design wavelength = {design_wl_nm:.1f} nm, N = {N}, "
        f"num_points_rcwa = {NUM_POINTS_RCWA}, cache = {CACHE_PATH.name}"
    )
    print(
        "Zero-order modal amplitudes are printed for orientation only. In this large-period "
        "supercell, power can scatter into many open diffraction orders."
    )

    for pol in INCIDENT_POLS:
        pol_data = bundle.incident_data(pol)
        zero_mode = Solver.zero_order_mode_index(N, pol)
        r0 = pol_data.reflected[zero_mode]
        t0 = pol_data.transmitted[zero_mode]
        print(
            f"{pol}: component = {pol_data.component}, "
            f"r0 = {complex(r0):.6g}, t0 = {complex(t0):.6g}, "
            f"|r0|^2 = {float(abs(r0) ** 2):.6f}, |t0|^2 = {float(abs(t0) ** 2):.6f}"
        )


    for pol in INCIDENT_POLS:
        component = bundle.incident_data(pol).component
        for layer_index in range(len(stack.layers)):
            fig, ax = plot_xz_profile_from_bundle(
                bundle,
                incident_pol=pol,
                layer_index=layer_index,
                plot_quantity=PLOT_QUANTITY,
            )
            ax.set_title(
                f"{_layer_name(layer_index).capitalize()} layer\n"
                f"{PLOT_QUANTITY}({component}) for {pol} incidence at "
                f"{design_wl_nm:.1f} nm"
            )
            fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
