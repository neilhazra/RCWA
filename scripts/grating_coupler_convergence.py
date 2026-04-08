"""Grating coupler convergence study."""

import numpy as jnp

from rcwa import Layer, Stack
from rcwa.solver import Solver

n_wg = 2.0
n_sub = 1.0
n_sup = 1.0
period_nm = 399.8
d_slab_nm = 120.0
d_grating_nm = 10.0
duty_cycle = 0.5
design_wl = 633.0
C_NM_THz = 299792.458


def make_stack(wavelength_nm: float, coupler_period_nm: float = period_nm) -> Stack:
    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_sub**2,
        eps_superstrate=n_sup**2,
    )
    fill_width = duty_cycle * coupler_period_nm
    # Uniform slab waveguide that supports the guided mode underneath the grating.
    stack.add_layer(
        Layer.uniform(
            thickness_nm=d_slab_nm,
            eps_tensor=n_wg**2 * jnp.eye(3),
            x_domain_nm=(0.0, coupler_period_nm),
        )
    )
    # Patterned grating layer etched into the top of the waveguide stack.
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=d_grating_nm,
            x_domain_nm=(0.0, coupler_period_nm),
            segments=[
                (0.0, fill_width, n_wg**2 * jnp.eye(3)),
                (fill_width, coupler_period_nm, n_sup**2 * jnp.eye(3)),
            ],
        )
    )
    return stack


def get_field_rt(stack: Stack, N: int, pol: str = "TE", num_points: int = 512):
    pol = pol.upper()
    r, t = Solver.reflection_transmission(stack, N, incident_pol=pol, num_points=num_points)
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
        r0 = r[zero_mode] * (E_sub_bwd / E_sub_fwd)
        t0 = t[zero_mode] * (E_sup_fwd / E_sub_fwd)
        return r0, t0

    if pol == "TM":
        E_sub_fwd = evecs_sub[h, 3, 1]
        E_sub_bwd = evecs_sub[h, 3, 3]
        E_sup_fwd = evecs_sup[h, 3, 1]
        zero_mode = Solver.zero_order_mode_index(N, "TM")
        r0 = r[zero_mode] * (E_sub_bwd / E_sub_fwd)
        t0 = t[zero_mode] * (E_sup_fwd / E_sub_fwd)
        return r0, t0

    raise ValueError(f"Unknown polarization {pol!r}")


def transmission_power_scale(pol: str) -> float:
    pol = pol.upper()
    if pol == "TE":
        return n_sup / n_sub
    if pol == "TM":
        return n_sub / n_sup
    raise ValueError(f"Unknown polarization {pol!r}")


def get_power_rt(stack: Stack, N: int, pol: str = "TE", num_points: int = 512):
    r0, t0 = get_field_rt(stack, N, pol=pol, num_points=num_points)
    R0 = jnp.abs(r0) ** 2
    T0 = jnp.abs(t0) ** 2 * transmission_power_scale(pol)
    return R0, T0


def sweep_frequency_response(
    frequencies_thz: jnp.ndarray,
    N: int,
    pol: str = "TE",
    num_points: int = 512,
):
    reflected = []
    transmitted = []
    coupled = []

    for frequency_thz in frequencies_thz:
        wavelength_nm = C_NM_THz / float(frequency_thz)
        stack = make_stack(wavelength_nm)
        R0, T0 = get_power_rt(stack, N, pol=pol, num_points=num_points)
        coupled_power = jnp.maximum(0.0, 1.0 - R0 - T0)

        reflected.append(float(R0))
        transmitted.append(float(T0))
        coupled.append(float(coupled_power))

    return (
        jnp.array(reflected),
        jnp.array(transmitted),
        jnp.array(coupled),
    )


def sweep_periodicity_response(
    periods_nm: jnp.ndarray,
    wavelength_nm: float,
    N: int,
    pol: str = "TE",
    num_points: int = 512,
):
    reflected = []
    transmitted = []
    coupled = []

    for coupler_period_nm in periods_nm:
        stack = make_stack(wavelength_nm, coupler_period_nm=float(coupler_period_nm))
        R0, T0 = get_power_rt(stack, N, pol=pol, num_points=num_points)
        coupled_power = jnp.maximum(0.0, 1.0 - R0 - T0)

        reflected.append(float(R0))
        transmitted.append(float(T0))
        coupled.append(float(coupled_power))

    return (
        jnp.array(reflected),
        jnp.array(transmitted),
        jnp.array(coupled),
    )


def plot_frequency_sweep(N_sweep: int = 55, num_points: int = 1024):
    import matplotlib.pyplot as plt

    center_frequency_thz = C_NM_THz / design_wl
    frequencies_thz = jnp.linspace(center_frequency_thz - 80.0, center_frequency_thz + 80.0, 161)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for ax, pol in zip(axes, ["TE", "TM"]):
        reflected, transmitted, coupled = sweep_frequency_response(
            frequencies_thz,
            N=N_sweep,
            pol=pol,
            num_points=num_points,
        )
        peak_index = int(jnp.argmax(coupled))
        peak_frequency_thz = float(frequencies_thz[peak_index])
        peak_wavelength_nm = C_NM_THz / peak_frequency_thz
        peak_coupling = float(coupled[peak_index])

        print(
            f"{pol}: peak coupled power = {peak_coupling:.4f} at "
            f"{peak_frequency_thz:.2f} THz ({peak_wavelength_nm:.1f} nm)"
        )

        ax.plot(frequencies_thz, reflected, label="R0")
        ax.plot(frequencies_thz, transmitted, label="T0")
        ax.plot(frequencies_thz, coupled, label="Coupled = 1 - R0 - T0")
        ax.axvline(center_frequency_thz, color="k", linestyle=":", alpha=0.4, label="Design")
        ax.set_ylabel("Power")
        ax.set_title(f"{pol} incidence")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def thz_to_nm(freq_thz):
        return C_NM_THz / freq_thz

    def nm_to_thz(wavelength_nm):
        return C_NM_THz / wavelength_nm

    axes[-1].set_xlabel("Frequency (THz)")
    top_axis = axes[0].secondary_xaxis("top", functions=(thz_to_nm, nm_to_thz))
    top_axis.set_xlabel("Wavelength (nm)")
    fig.suptitle(
        "Grating Coupler Frequency Sweep\n"
        "For this subwavelength geometry, 1 - R0 - T0 tracks power coupled out of the zero-order radiation channels."
    )
    fig.tight_layout()
    return fig


def plot_periodicity_sweep(
    wavelength_nm: float = design_wl,
    N_sweep: int = 55,
    num_points: int = 1024,
):
    import matplotlib.pyplot as plt

    periods_nm = jnp.linspace(period_nm - 120.0, period_nm + 120.0, 161)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for ax, pol in zip(axes, ["TE", "TM"]):
        reflected, transmitted, coupled = sweep_periodicity_response(
            periods_nm,
            wavelength_nm=wavelength_nm,
            N=N_sweep,
            pol=pol,
            num_points=num_points,
        )
        peak_index = int(jnp.argmax(coupled))
        peak_period_nm = float(periods_nm[peak_index])
        peak_coupling = float(coupled[peak_index])

        print(
            f"{pol}: peak coupled power = {peak_coupling:.4f} at "
            f"period = {peak_period_nm:.2f} nm for wavelength {wavelength_nm:.1f} nm"
        )

        ax.plot(periods_nm, reflected, label="R0")
        ax.plot(periods_nm, transmitted, label="T0")
        ax.plot(periods_nm, coupled, label="Coupled = 1 - R0 - T0")
        ax.axvline(period_nm, color="k", linestyle=":", alpha=0.4, label="Design")
        ax.set_ylabel("Power")
        ax.set_title(f"{pol} incidence")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Grating Period (nm)")
    fig.suptitle(
        f"Grating Coupler Period Sweep at {wavelength_nm:.1f} nm\n"
        "Duty cycle and thicknesses are fixed while the grating period is varied."
    )
    fig.tight_layout()
    return fig


def run_all_sweeps() -> None:
    import matplotlib.pyplot as plt

    plot_frequency_sweep()
    plot_periodicity_sweep()
    plt.show()


if __name__ == "__main__":
    run_all_sweeps()
