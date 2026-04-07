"""Grating coupler convergence study."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from rcwa import Layer, Stack
from rcwa.solver import Solver

n_wg = 2.0
n_sub = 1.5
n_sup = 1.0
period_nm = 370.0
d_wg_nm = 200.0
duty_cycle = 0.5
design_wl = 633.0


def make_stack(wavelength_nm: float) -> Stack:
    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=n_sub**2,
        eps_superstrate=n_sup**2,
    )
    fill_width = duty_cycle * period_nm
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=d_wg_nm,
            x_domain_nm=(0.0, period_nm),
            segments=[
                (0.0, fill_width, n_wg**2 * jnp.eye(3)),
                (fill_width, period_nm, n_sup**2 * jnp.eye(3)),
            ],
        )
    )
    return stack


def get_field_rt(stack, N, pol="TE"):
    r, t = Solver.reflection_transmission(stack, N, incident_pol=pol)
    Q_sub = stack.get_Q_substrate_normalized(N)
    Q_sup = stack.get_Q_superstrate_normalized(N)
    _, evecs_sub = Stack.diagonalize_sort_isotropic_modes(Q_sub)
    _, evecs_sup = Stack.diagonalize_sort_isotropic_modes(Q_sup)
    h = Stack.zero_harmonic_index(N)
    if pol == "TE":
        E_sub_fwd = evecs_sub[h, 2, 0]
        E_sub_bwd = evecs_sub[h, 2, 2]
        E_sup_fwd = evecs_sup[h, 2, 0]
        zero_mode = Solver.zero_order_mode_index(N, "TE")
        r0 = r[zero_mode] * (E_sub_bwd / E_sub_fwd)
        t0 = t[zero_mode] * (E_sup_fwd / E_sub_fwd)
    else:
        E_sub_fwd = evecs_sub[h, 3, 1]
        E_sub_bwd = evecs_sub[h, 3, 3]
        E_sup_fwd = evecs_sup[h, 3, 1]
        zero_mode = Solver.zero_order_mode_index(N, "TM")
        r0 = r[zero_mode] * (E_sub_bwd / E_sub_fwd)
        t0 = t[zero_mode] * (E_sup_fwd / E_sub_fwd)
    return r0, t0


if __name__ == "__main__":
    G = design_wl / period_nm
    d_norm = 2 * jnp.pi * d_wg_nm / design_wl
    N_max = 600
    print(f"G={G:.3f}, d_norm={d_norm:.3f}, scanning to N={N_max}")

    N_values = list(range(300, N_max + 1,21))
    R_te = []
    T_te = []
    R_tm = []
    T_tm = []

    stack = make_stack(design_wl)
    for N in N_values:
        print(f"  N={N}")
        for pol, R_list, T_list in [
            ("TE", R_te, T_te),
            ("TM", R_tm, T_tm),
        ]:
            r0, t0 = get_field_rt(stack, N, pol)
            R_list.append(float(jnp.abs(r0) ** 2))
            T_list.append(float(jnp.abs(t0) ** 2 * n_sup / n_sub))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_values, R_te, "o-", label="R0 TE")
    ax.plot(N_values, T_te, "s--", label="T0 TE")
    ax.plot(N_values, R_tm, "o-", label="R0 TM")
    ax.plot(N_values, T_tm, "s--", label="T0 TM")
    ax.set_xlabel("Truncation order N")
    ax.set_ylabel("Coefficient")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
