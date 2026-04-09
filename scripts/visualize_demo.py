"""Demo script for the field-visualization helpers in :mod:`rcwa.visualize`."""

from __future__ import annotations

import pathlib
import sys

import numpy as jnp


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcwa import Layer, Stack
from rcwa.visualize import (
    _first_supported_symmetric_slab_mode_thickness_nm,
    plot_x_line_profile_at_fixed_z,
    plot_xz_profile,
)


def main() -> None:
    """Run the homogeneous-slab visualization demo."""
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
        mode_order=2,
    )

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
        plot_quantity="real",
    )
    plot_xz_profile(
        stack,
        layer_index=layer_index,
        incident_pol=incident_pol,
        component=component,
        N=N,
        plot_quantity="real",
    )
    plt.show()


if __name__ == "__main__":
    main()
