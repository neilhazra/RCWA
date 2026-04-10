from __future__ import annotations

import numpy as jnp
import pytest

from rcwa import Layer, Solver, Stack
from rcwa.visualize import (
    _field_k_at_local_depth,
    _layer_face_coefficients,
    _first_supported_symmetric_slab_mode_thickness_nm,
    _symmetric_slab_mode_kappa_inv_nm,
    VisualizationBundle,
    compute_visualization_bundle,
    create_x_line_profile_at_fixed_z_from_bundle,
    evaluate_real_space_from_k,
    create_x_line_profile_at_fixed_z,
    create_xz_profile_from_bundle,
    create_xz_profile,
)


def _uniform_single_layer_stack() -> Stack:
    stack = Stack(
        wavelength_nm=633.0,
        kappa_inv_nm=0.0,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=120.0,
            eps_tensor=2.25 * jnp.eye(3, dtype=jnp.complex128),
            x_domain_nm=(0.0, 500.0),
        )
    )
    return stack


def _oblique_anisotropic_single_layer_stack() -> Stack:
    wavelength_nm = 780.0
    zeta = 0.75
    stack = Stack(
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=zeta * 2 * jnp.pi / wavelength_nm,
        eps_substrate=1.0,
        eps_superstrate=2.25,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=122.0,
            eps_tensor=jnp.diag(
                jnp.array([4.0, 6.0, 1.6**2], dtype=jnp.complex128)
            ),
            x_domain_nm=(0.0, 500.0),
        )
    )
    return stack


def _manual_inverse_fourier(
    coeffs: jnp.ndarray,
    x_nm: jnp.ndarray,
    x_domain_nm: tuple[float, float],
    kappa_inv_nm: float = 0.0,
) -> jnp.ndarray:
    x_min_nm, x_max_nm = x_domain_nm
    period_nm = x_max_nm - x_min_nm
    max_order = (coeffs.shape[0] - 1) // 2
    harmonic_orders = jnp.arange(-max_order, max_order + 1)
    phase = 1j * (
        kappa_inv_nm + 2 * jnp.pi * harmonic_orders[None, :] / period_nm
    ) * (x_nm[:, None] - x_min_nm)
    return jnp.sum(coeffs[None, :] * jnp.exp(phase), axis=1)


@pytest.mark.parametrize(
    ("component", "component_index", "kappa_inv_nm"),
    [
        ("-H_y", 0, 0.0),
        ("H_x", 1, 0.0031),
        ("E_y", 2, 0.0),
        ("D_x", 3, 0.0075),
    ],
)
def test_evaluate_real_space_from_k_reconstructs_known_fourier_series(
    component: str,
    component_index: int,
    kappa_inv_nm: float,
) -> None:
    coeffs = jnp.array(
        [0.3 - 0.1j, -0.25 + 0.4j, 1.2 + 0.05j, 0.15 - 0.2j, -0.1 + 0.3j],
        dtype=jnp.complex128,
    )
    field_k = jnp.zeros(20, dtype=jnp.complex128)
    field_k[component_index::4] = coeffs
    x_domain_nm = (0.0, 400.0)
    x_nm = jnp.linspace(0.0, 400.0, 64, endpoint=False)

    expected = _manual_inverse_fourier(
        coeffs,
        x_nm,
        x_domain_nm,
        kappa_inv_nm=kappa_inv_nm,
    )
    actual = evaluate_real_space_from_k(
        field_k,
        component,
        x_nm,
        x_domain_nm,
        kappa_inv_nm=kappa_inv_nm,
    )

    assert jnp.allclose(actual, expected, atol=1e-12)


def test_uniform_layer_line_profile_is_constant_across_x_for_zero_order_excitation() -> None:
    stack = _uniform_single_layer_stack()
    x_nm, field_x = create_x_line_profile_at_fixed_z(
        stack,
        layer_index=0,
        incident_pol="TE",
        component="E_y",
        z_nm=0.5 * stack.layers[0].thickness_nm,
        N=1,
        num_points_x=128,
        num_points_rcwa=128,
    )

    assert x_nm.shape == (128,)
    assert field_x.shape == (128,)
    assert jnp.max(jnp.abs(field_x - field_x[0])) < 1e-10


def test_symmetric_slab_mode_solver_reports_no_te1_for_original_demo_thickness() -> None:
    assert (
        _symmetric_slab_mode_kappa_inv_nm(
            wavelength_nm=633.0,
            thickness_nm=120.0,
            n_core=1.5,
            n_clad=1.0,
            pol="TE",
            mode_order=1,
        )
        is None
    )


def test_first_supported_te1_thickness_returns_bound_kappa() -> None:
    thickness_nm, kappa_inv_nm = _first_supported_symmetric_slab_mode_thickness_nm(
        wavelength_nm=633.0,
        start_thickness_nm=120.0,
        n_core=1.5,
        n_clad=1.0,
        pol="TE",
        mode_order=1,
        thickness_step_nm=10.0,
    )
    k0 = 2 * jnp.pi / 633.0

    assert thickness_nm > 120.0
    assert k0 * 1.0 < kappa_inv_nm < k0 * 1.5


def test_create_x_line_profile_matches_layer_face_fields_at_left_and_right_faces() -> None:
    stack = _uniform_single_layer_stack()
    modal_data, left_coeffs, right_coeffs = _layer_face_coefficients(
        stack,
        layer_index=0,
        incident_pol="TE",
        N=1,
        num_points_rcwa=128,
        verbose=False,
    )
    layer = stack.layers[0]
    x_nm = layer.sample_points(96)

    left_field_k = _field_k_at_local_depth(
        stack,
        modal_data=modal_data,
        layer_index=0,
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        z_nm=0.0,
    )
    right_field_k = _field_k_at_local_depth(
        stack,
        modal_data=modal_data,
        layer_index=0,
        left_coeffs=left_coeffs,
        right_coeffs=right_coeffs,
        z_nm=layer.thickness_nm,
    )
    expected_left = evaluate_real_space_from_k(
        left_field_k,
        "E_y",
        x_nm,
        layer.x_domain_nm,
        kappa_inv_nm=stack.kappa_inv_nm,
    )
    expected_right = evaluate_real_space_from_k(
        right_field_k,
        "E_y",
        x_nm,
        layer.x_domain_nm,
        kappa_inv_nm=stack.kappa_inv_nm,
    )

    actual_x_left, actual_left = create_x_line_profile_at_fixed_z(
        stack,
        layer_index=0,
        incident_pol="TE",
        component="E_y",
        z_nm=0.0,
        N=1,
        num_points_x=96,
        num_points_rcwa=128,
    )
    actual_x_right, actual_right = create_x_line_profile_at_fixed_z(
        stack,
        layer_index=0,
        incident_pol="TE",
        component="E_y",
        z_nm=layer.thickness_nm,
        N=1,
        num_points_x=96,
        num_points_rcwa=128,
    )

    assert jnp.array_equal(actual_x_left, x_nm)
    assert jnp.array_equal(actual_x_right, x_nm)
    assert jnp.allclose(actual_left, expected_left, atol=1e-10)
    assert jnp.allclose(actual_right, expected_right, atol=1e-10)


def test_create_xz_profile_matches_fixed_z_line_profiles_row_by_row() -> None:
    stack = _uniform_single_layer_stack()
    x_nm, z_nm, field_xz = create_xz_profile(
        stack,
        layer_index=0,
        incident_pol="TE",
        component="E_y",
        N=1,
        num_points_x=80,
        num_points_z=7,
        num_points_rcwa=128,
    )

    assert x_nm.shape == (80,)
    assert z_nm.shape == (7,)
    assert field_xz.shape == (7, 80)

    for row, z_value in enumerate(z_nm):
        row_x, row_field = create_x_line_profile_at_fixed_z(
            stack,
            layer_index=0,
            incident_pol="TE",
            component="E_y",
            z_nm=float(z_value),
            N=1,
            num_points_x=80,
            num_points_rcwa=128,
        )
        assert jnp.array_equal(row_x, x_nm)
        assert jnp.allclose(field_xz[row], row_field, atol=1e-10)


def test_visualization_bundle_matches_individual_profiles_and_roundtrips(
    tmp_path,
) -> None:
    stack = _uniform_single_layer_stack()
    cache_path = tmp_path / "visualization_bundle.npz"
    bundle = compute_visualization_bundle(
        stack,
        N=1,
        num_points_x=48,
        num_points_z=5,
        num_points_rcwa=128,
        cache_path=cache_path,
        verbose=False,
    )

    assert cache_path.exists()
    assert set(bundle.polarization_data) == {"TE", "TM"}

    for pol, component in [("TE", "E_y"), ("TM", "-H_y")]:
        expected_reflected, expected_transmitted = Solver.reflection_transmission(
            stack,
            1,
            incident_pol=pol,
            num_points=128,
            verbose=False,
        )
        actual_pol_data = bundle.incident_data(pol)
        assert actual_pol_data.component == component
        assert jnp.allclose(actual_pol_data.reflected, expected_reflected, atol=1e-12)
        assert jnp.allclose(actual_pol_data.transmitted, expected_transmitted, atol=1e-12)

        expected_x, expected_z, expected_field_xz = create_xz_profile(
            stack,
            layer_index=0,
            incident_pol=pol,
            component=component,
            N=1,
            num_points_x=48,
            num_points_z=5,
            num_points_rcwa=128,
            verbose=False,
        )
        actual_x, actual_z, actual_field_xz = create_xz_profile_from_bundle(
            bundle,
            incident_pol=pol,
            layer_index=0,
        )
        assert jnp.array_equal(actual_x, expected_x)
        assert jnp.array_equal(actual_z, expected_z)
        assert jnp.allclose(actual_field_xz, expected_field_xz, atol=1e-10)

        line_x, line_field = create_x_line_profile_at_fixed_z_from_bundle(
            bundle,
            incident_pol=pol,
            layer_index=0,
            z_nm=float(expected_z[2]),
        )
        assert jnp.array_equal(line_x, expected_x)
        assert jnp.allclose(line_field, expected_field_xz[2], atol=1e-10)

    loaded = VisualizationBundle.load(cache_path)
    assert loaded.N == bundle.N
    assert loaded.num_points_rcwa == bundle.num_points_rcwa
    assert loaded.num_points_x == bundle.num_points_x
    assert loaded.num_points_z == bundle.num_points_z
    assert complex(loaded.kappa_inv_nm) == complex(bundle.kappa_inv_nm)
    for pol in ["TE", "TM"]:
        loaded_pol = loaded.incident_data(pol)
        bundle_pol = bundle.incident_data(pol)
        assert loaded_pol.component == bundle_pol.component
        assert jnp.allclose(loaded_pol.reflected, bundle_pol.reflected, atol=1e-12)
        assert jnp.allclose(loaded_pol.transmitted, bundle_pol.transmitted, atol=1e-12)
        assert jnp.allclose(
            loaded_pol.layer_profiles[0].field_xz,
            bundle_pol.layer_profiles[0].field_xz,
            atol=1e-12,
        )


def test_layer_face_coefficients_use_tangential_continuity_for_oblique_anisotropic_layer() -> None:
    stack = _oblique_anisotropic_single_layer_stack()
    N = 0
    num_points_rcwa = 128
    modal_data, left_coeffs, _ = _layer_face_coefficients(
        stack,
        layer_index=0,
        incident_pol="TM",
        N=N,
        num_points_rcwa=num_points_rcwa,
        verbose=False,
    )
    S11, _, _, _ = Solver.total_scattering_matrix(
        stack,
        N,
        num_points=num_points_rcwa,
        verbose=False,
    )
    inc = jnp.zeros(2 * Stack.num_harmonics(N), dtype=jnp.complex128)
    inc[Solver.zero_order_mode_index(N, "TM")] = 1.0
    reflected = S11 @ inc
    substrate_coeffs = jnp.concatenate([inc, reflected])

    substrate_tangential_field = modal_data.substrate_continuity_fields @ substrate_coeffs
    layer_tangential_field = modal_data.layer_modes[0][2] @ left_coeffs

    assert jnp.allclose(substrate_tangential_field, layer_tangential_field, atol=1e-10)


def test_create_xz_profile_supports_arbitrary_layer_index_and_returns_finite_arrays(
    sample_problem: dict,
) -> None:
    stack = sample_problem["stack"]
    x_nm, z_nm, field_xz = create_xz_profile(
        stack,
        layer_index=2,
        incident_pol="TE",
        component="E_y",
        N=1,
        num_points_x=48,
        num_points_z=5,
        num_points_rcwa=128,
    )

    assert x_nm.shape == (48,)
    assert z_nm.shape == (5,)
    assert field_xz.shape == (5, 48)
    assert jnp.all(jnp.isfinite(field_xz))


def test_create_xz_profile_remains_finite_for_high_order_uniform_slab_visualization() -> None:
    stack = Stack(
        wavelength_nm=633.0,
        kappa_inv_nm=0.0,
        eps_substrate=1.0,
        eps_superstrate=1.0,
    )
    stack.add_layer(
        Layer.uniform(
            thickness_nm=120.0,
            eps_tensor=2.25 * jnp.eye(3, dtype=jnp.complex128),
            x_domain_nm=(0.0, 100.0),
        )
    )

    _, _, field_xz = create_xz_profile(
        stack,
        layer_index=0,
        incident_pol="TE",
        component="E_y",
        N=96,
        num_points_x=16,
        num_points_z=5,
        num_points_rcwa=max(1024, 32 * (2 * 96 + 1)),
    )

    assert jnp.all(jnp.isfinite(field_xz))


def test_visualize_helpers_raise_for_invalid_inputs() -> None:
    stack = _uniform_single_layer_stack()

    with pytest.raises(ValueError, match="Unsupported component"):
        create_x_line_profile_at_fixed_z(
            stack,
            layer_index=0,
            incident_pol="TE",
            component="E_x",
            z_nm=10.0,
            N=0,
        )

    with pytest.raises(ValueError, match="Unknown incident_pol"):
        create_x_line_profile_at_fixed_z(
            stack,
            layer_index=0,
            incident_pol="bad",
            component="E_y",
            z_nm=10.0,
            N=0,
        )

    with pytest.raises(IndexError, match="layer_index=3"):
        create_x_line_profile_at_fixed_z(
            stack,
            layer_index=3,
            incident_pol="TE",
            component="E_y",
            z_nm=10.0,
            N=0,
        )

    with pytest.raises(ValueError, match="outside the layer thickness interval"):
        create_x_line_profile_at_fixed_z(
            stack,
            layer_index=0,
            incident_pol="TE",
            component="E_y",
            z_nm=-1.0,
            N=0,
        )

    with pytest.raises(ValueError, match="outside the layer thickness interval"):
        create_x_line_profile_at_fixed_z(
            stack,
            layer_index=0,
            incident_pol="TE",
            component="E_y",
            z_nm=stack.layers[0].thickness_nm + 1.0,
            N=0,
        )
