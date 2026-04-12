"""Test total_scattering_matrix against 2x2 transfer-matrix / Fresnel and pyGTM 4x4.

The modal S-matrix operates on eigenvector amplitudes, not directly on E-field
amplitudes.  To extract physical reflection/transmission coefficients we project
the modal amplitudes through the mode-to-tangential-field matrices and take
ratios of the relevant E-field components (Ey for TE, Ex for TM).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as jnp
import pytest

from rcwa import Layer, Solver, Stack

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "pyGTM"))
GTM = pytest.importorskip("GTM.GTMcore")


# ---------------------------------------------------------------------------
# Helpers: extract physical r, t from modal S-matrix
# ---------------------------------------------------------------------------

def __mode_to_tangential_field(stack: Stack, eps: complex, N: int, num_points: int = 512):
    """Tangential-field matrix F mapping modal amplitudes to [-Hy, Hx, Ey, Ex]."""
    Q = stack._build_uniform_medium_Q_normalized(eps, N, num_points)
    mode_to_reduced = Solver.harmonic_to_component_major_rows(
        Solver._isotropic_halfspace_mode_to_field(Q, N)
    )
    tang_xform = stack._uniform_medium_reduced_to_tangential_field_transform_component_major(eps, N)
    return tang_xform @ mode_to_reduced


def __extract_rt(stack: Stack, N: int, pol: str, num_points: int = 512):
    """Extract physical E-field r and t for the zero-order harmonic.

    Returns (r, t) as complex scalars comparable to Fresnel / characteristic-matrix values.
    """
    S11, S12, S21, S22 = Solver.total_scattering_matrix(
        stack, N, num_points=num_points, verbose=False,
    )
    F_sub = __mode_to_tangential_field(stack, stack.eps_substrate, N, num_points)
    F_sup = __mode_to_tangential_field(stack, stack.eps_superstrate, N, num_points)

    half = S11.shape[0]
    num_h = Stack.num_harmonics(N)

    # Incident mode: unit amplitude on the zero-order forward mode
    mode_idx = Solver.zero_order_mode_index(N, pol)
    inc = jnp.zeros(half, dtype=jnp.complex128)
    inc[mode_idx] = 1.0

    # Tangential fields: [-Hy, Hx, Ey, Ex] in component-major order
    # For num_h harmonics, row layout is:
    #   [-Hy(-N..N), Hx(-N..N), Ey(-N..N), Ex(-N..N)]
    # The E-field component we want is at harmonic-zero within its block.
    zero_h = Stack.zero_harmonic_index(N)
    if pol.upper() == "TE":
        field_row = 2 * num_h + zero_h   # Ey block, zero harmonic
    else:
        field_row = 3 * num_h + zero_h   # Ex block, zero harmonic

    inc_field = (F_sub[:, :half] @ inc)[field_row]
    refl_field = (F_sub[:, half:] @ (S11 @ inc))[field_row]
    trans_field = (F_sup[:, :half] @ (S21 @ inc))[field_row]

    r = refl_field / inc_field
    t = trans_field / inc_field
    return r, t

def _extract_rt(stack: Stack, N: int, pol: str, num_points: int = 512):
    """Extract physical E-field r and t for the zero-order harmonic.

    Returns (r, t) as complex scalars comparable to Fresnel / characteristic-matrix values.
    r/t matrix rows: [Ey(-N..N), Ex(-N..N)], columns: [TE(-N..N), TM(-N..N)].
    """
    r, t = Solver.reflection_transmission(
        stack, N, num_points=num_points, verbose=False,
    )
    num_h = Stack.num_harmonics(N)
    zero_h = Stack.zero_harmonic_index(N)

    if pol.upper() == "TE":
        row = zero_h              # Ey at zero order
        col = zero_h              # TE input at zero order
    else:
        row = num_h + zero_h      # Ex at zero order
        col = num_h + zero_h      # TM input at zero order

    return r[row, col], t[row, col]



# ---------------------------------------------------------------------------
# Fresnel / characteristic-matrix references
# ---------------------------------------------------------------------------

def _fresnel_normal(n1: complex, n2: complex):
    """Fresnel r, t at normal incidence (same for TE and TM)."""
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def _characteristic_matrix_rt(
    n_incident, n_exit, layer_ns, layer_thicknesses_nm, wavelength_nm,
):
    """2x2 characteristic matrix method at normal incidence (exp(-iwt) convention).

    The RCWA solver uses the exp(-iwt) time convention, so forward propagation
    accumulates phase exp(+ikd).  The standard textbook characteristic matrix
    uses exp(+iwt), so we negate the phase to match.
    """
    matrix = jnp.eye(2, dtype=jnp.complex128)
    for n_layer, thickness_nm in zip(layer_ns, layer_thicknesses_nm):
        phase = -2 * jnp.pi * n_layer * thickness_nm / wavelength_nm
        layer_matrix = jnp.array(
            [
                [jnp.cos(phase), 1j * jnp.sin(phase) / n_layer],
                [1j * n_layer * jnp.sin(phase), jnp.cos(phase)],
            ],
            dtype=jnp.complex128,
        )
        matrix = matrix @ layer_matrix
    B = matrix[0, 0] + matrix[0, 1] * n_exit
    C = matrix[1, 0] + matrix[1, 1] * n_exit
    r = (n_incident * B - C) / (n_incident * B + C)
    t = 2 * n_incident / (n_incident * B + C)
    return r, t


# ---------------------------------------------------------------------------
# Stack builder
# ---------------------------------------------------------------------------

def _make_stack(wavelength_nm, period_nm, eps_sub, eps_sup, layer_eps_list, layer_thicknesses_nm):
    layers = [
        Layer.uniform(
            thickness_nm=thick,
            eps_tensor=jnp.asarray(eps, dtype=jnp.complex128) * jnp.eye(3, dtype=jnp.complex128),
            x_domain_nm=(0.0, period_nm),
        )
        for eps, thick in zip(layer_eps_list, layer_thicknesses_nm)
    ]
    return Stack(
        layers=layers,
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=eps_sub,
        eps_superstrate=eps_sup,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFresnelInterface:
    """Bare interface (zero-thickness layer) between different half-spaces."""

    @pytest.mark.parametrize("pol", ["TE", "TM"])
    def test_glass_air_interface(self, pol):
        n_sub, n_sup = 1.5, 1.0
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_sub**2], [0.0])
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol=pol)
        r_ref, t_ref = _fresnel_normal(n_sub, n_sup)
        assert abs(r_rcwa - r_ref) < 1e-10, f"{pol} r mismatch: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-10, f"{pol} t mismatch: {t_rcwa} vs {t_ref}"

    @pytest.mark.parametrize("pol", ["TE", "TM"])
    def test_same_medium_identity(self, pol):
        n = 1.5
        stack = _make_stack(550.0, 200.0, n**2, n**2, [n**2], [0.0])
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol=pol)
        assert abs(r_rcwa) < 1e-10, f"{pol} reflection should vanish, got {r_rcwa}"
        assert abs(t_rcwa - 1.0) < 1e-10, f"{pol} transmission should be 1, got {t_rcwa}"

    @pytest.mark.parametrize("pol", ["TE", "TM"])
    def test_energy_conservation_interface(self, pol):
        n_sub, n_sup = 1.5, 1.0
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_sub**2], [0.0])
        r, t = _extract_rt(stack, N=0, pol=pol)
        R = abs(r) ** 2
        T = abs(t) ** 2 * (n_sup / n_sub)
        assert abs(R + T - 1.0) < 1e-10, f"R+T={R+T}"

    @pytest.mark.parametrize("n_sub, n_sup", [
        (1.0, 1.5),
        (2.0, 1.0),
        (1.0, 2.5),
        (1.52, 1.0),
    ])
    def test_various_index_contrasts(self, n_sub, n_sup):
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_sub**2], [0.0])
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol="TE")
        r_ref, t_ref = _fresnel_normal(n_sub, n_sup)
        assert abs(r_rcwa - r_ref) < 1e-10, f"r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-10, f"t: {t_rcwa} vs {t_ref}"


class TestSingleFilm:
    """Single uniform film vs 2x2 characteristic matrix."""

    @pytest.mark.parametrize(
        "n_sub, n_film, n_sup, d_nm",
        [
            (1.0, 1.5, 1.0, 100.0),
            (1.5, 2.0, 1.0, 137.5),
            (1.0, 1.5 + 0.01j, 1.0, 200.0),
            (1.52, 1.38, 1.0, 99.6),
        ],
    )
    @pytest.mark.parametrize("pol", ["TE", "TM"])
    def test_single_film(self, n_sub, n_film, n_sup, d_nm, pol):
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_film**2], [d_nm])
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol=pol)
        r_ref, t_ref = _characteristic_matrix_rt(n_sub, n_sup, [n_film], [d_nm], 550.0)
        assert abs(r_rcwa - r_ref) < 1e-6, f"{pol} r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-6, f"{pol} t: {t_rcwa} vs {t_ref}"

    @pytest.mark.parametrize("pol", ["TE", "TM"])
    def test_single_film_energy_conservation(self, pol):
        n_sub, n_film, n_sup = 1.0, 1.5, 1.0
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_film**2], [100.0])
        r, t = _extract_rt(stack, N=0, pol=pol)
        R = abs(r) ** 2
        T = abs(t) ** 2 * (jnp.real(n_sup) / jnp.real(n_sub))
        assert abs(R + T - 1.0) < 1e-10, f"R+T={R+T}"


class TestMultiLayerFilm:
    """Multi-layer stacks vs 2x2 characteristic matrix."""

    def test_two_layer_ar_coating(self):
        n_sub, n_sup = 1.52, 1.0
        n1, d1 = 2.1, 17.0
        n2, d2 = 1.38, 99.6
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n1**2, n2**2], [d1, d2])
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol="TE")
        r_ref, t_ref = _characteristic_matrix_rt(n_sub, n_sup, [n1, n2], [d1, d2], 550.0)
        assert abs(r_rcwa - r_ref) < 1e-6, f"r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-6, f"t: {t_rcwa} vs {t_ref}"

    def test_bragg_mirror_5_pairs(self):
        wavelength_nm = 550.0
        n_sub, n_sup = 1.52, 1.0
        nH, nL = 2.3, 1.38
        dH = wavelength_nm / (4 * nH)
        dL = wavelength_nm / (4 * nL)

        layer_ns = [nH, nL] * 5
        layer_ds = [dH, dL] * 5

        stack = _make_stack(
            wavelength_nm, 200.0,
            n_sub**2, n_sup**2,
            [n**2 for n in layer_ns], layer_ds,
        )
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol="TE")
        r_ref, t_ref = _characteristic_matrix_rt(n_sub, n_sup, layer_ns, layer_ds, wavelength_nm)
        assert abs(r_rcwa - r_ref) < 1e-5, f"r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-5, f"t: {t_rcwa} vs {t_ref}"

    def test_bragg_mirror_energy_conservation(self):
        wavelength_nm = 550.0
        n_sub, n_sup = 1.52, 1.0
        nH, nL = 2.3, 1.38
        layer_ns = [nH, nL] * 5
        layer_ds = [wavelength_nm / (4 * nH), wavelength_nm / (4 * nL)] * 5

        stack = _make_stack(
            wavelength_nm, 200.0,
            n_sub**2, n_sup**2,
            [n**2 for n in layer_ns], layer_ds,
        )
        r, t = _extract_rt(stack, N=0, pol="TE")
        R = abs(r) ** 2
        T = abs(t) ** 2 * (n_sup / n_sub)
        assert abs(R + T - 1.0) < 1e-8, f"R+T={R+T}"


class TestHigherHarmonics:
    """With N>0, zero-order should still match for sub-wavelength period uniform layers."""

    @pytest.mark.parametrize("N", [1, 2])
    def test_uniform_film_higher_N(self, N):
        n_sub, n_film, n_sup = 1.5, 2.0, 1.0
        d_nm = 100.0
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_film**2], [d_nm])
        r_rcwa, t_rcwa = _extract_rt(stack, N=N, pol="TE")
        r_ref, t_ref = _characteristic_matrix_rt(n_sub, n_sup, [n_film], [d_nm], 550.0)
        assert abs(r_rcwa - r_ref) < 1e-5, f"N={N} r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-5, f"N={N} t: {t_rcwa} vs {t_ref}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_interface_higher_N(self, N):
        n_sub, n_sup = 1.5, 1.0
        stack = _make_stack(550.0, 200.0, n_sub**2, n_sup**2, [n_sub**2], [0.0])
        r_rcwa, t_rcwa = _extract_rt(stack, N=N, pol="TE")
        r_ref, t_ref = _fresnel_normal(n_sub, n_sup)
        assert abs(r_rcwa - r_ref) < 1e-8, f"N={N} r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-8, f"N={N} t: {t_rcwa} vs {t_ref}"


# ---------------------------------------------------------------------------
# Helpers for pyGTM 4x4 comparison
# ---------------------------------------------------------------------------

def _const_eps_fn(value: complex):
    return lambda frequency_hz, _value=complex(value): _value


def _rotate_in_plane_eps(eps_local: jnp.ndarray, theta_rad: float) -> jnp.ndarray:
    c = jnp.cos(theta_rad)
    s = jnp.sin(theta_rad)
    R = jnp.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.complex128)
    return R @ eps_local @ R.T


def _pygtm_reflectances(
    wavelength_nm: float,
    eps_incident: complex,
    eps_exit: complex,
    layers: list[tuple[float, jnp.ndarray, float]],
) -> dict[str, float]:
    """Get reflectance coefficients from pyGTM at normal incidence.

    layers: list of (thickness_nm, eps_diag_3, angle_deg) where eps_diag_3 is
    the diagonal of the permittivity tensor in the local frame.

    pyGTM convention: superstrate=incident side, substrate=exit side.
    Returns dict with keys Rpp, Rss, Rps, Rsp.
    """
    system = GTM.System()
    system.set_superstrate(GTM.Layer(epsilon1=_const_eps_fn(eps_incident)))
    system.set_substrate(GTM.Layer(epsilon1=_const_eps_fn(eps_exit)))

    for thickness_nm, eps_diag, angle_deg in layers:
        system.add_layer(GTM.Layer(
            thickness=thickness_nm * 1e-9,
            epsilon1=_const_eps_fn(eps_diag[0]),
            epsilon2=_const_eps_fn(eps_diag[1]),
            epsilon3=_const_eps_fn(eps_diag[2]),
            phi=-jnp.deg2rad(angle_deg),
        ))

    frequency_hz = GTM.c_const / (wavelength_nm * 1e-9)
    system.initialize_sys(frequency_hz)
    zeta = 0.0
    system.calculate_GammaStar(frequency_hz, zeta)
    with jnp.errstate(divide="ignore", invalid="ignore"):
        r_out, _, _, _ = system.calculate_r_t(zeta)
    # r_out = [rpp, rps, rss, rsp]
    return {
        "Rpp": abs(r_out[0]) ** 2,
        "Rss": abs(r_out[2]) ** 2,
        "Rps": abs(r_out[1]) ** 2,
        "Rsp": abs(r_out[3]) ** 2,
    }


def _extract_reflectances(stack: Stack, N: int = 0, num_points: int = 512) -> dict[str, float]:
    """Extract zero-order reflectances from the r/t matrices.

    r matrix rows: [Ey(-N..N), Ex(-N..N)], columns: [TE(-N..N), TM(-N..N)].
    Returns Rpp, Rss, Rps, Rsp as |E_reflected / E_incident|^2.
    """
    r, _ = Solver.reflection_transmission(
        stack, N, num_points=num_points, verbose=False,
    )
    num_h = Stack.num_harmonics(N)
    zero_h = Stack.zero_harmonic_index(N)

    ey_row = zero_h              # Ey at zero order
    ex_row = num_h + zero_h      # Ex at zero order
    te_col = zero_h              # TE input at zero order
    tm_col = num_h + zero_h      # TM input at zero order

    return {
        "Rss": abs(r[ey_row, te_col]) ** 2,   # TE → Ey
        "Rpp": abs(r[ex_row, tm_col]) ** 2,   # TM → Ex
        "Rps": abs(r[ex_row, te_col]) ** 2,   # TE → Ex
        "Rsp": abs(r[ey_row, tm_col]) ** 2,   # TM → Ey
    }


def _make_anisotropic_stack(
    wavelength_nm: float,
    period_nm: float,
    eps_substrate: complex,
    eps_superstrate: complex,
    layers: list[tuple[float, jnp.ndarray, float]],
) -> Stack:
    """Build a Stack from layers specified as (thickness_nm, eps_diag_3, angle_deg)."""
    rcwa_layers = []
    for thickness_nm, eps_diag, angle_deg in layers:
        eps_local = jnp.diag(jnp.asarray(eps_diag, dtype=jnp.complex128))
        eps_tensor = _rotate_in_plane_eps(eps_local, jnp.deg2rad(angle_deg))
        rcwa_layers.append(
            Layer.uniform(thickness_nm, eps_tensor, (0.0, period_nm))
        )
    return Stack(
        layers=rcwa_layers,
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=eps_substrate,
        eps_superstrate=eps_superstrate,
    )


# ---------------------------------------------------------------------------
# Tests: lossy isotropic media vs pyGTM
# ---------------------------------------------------------------------------

class TestLossyIsotropicVsPyGTM:
    """Lossy isotropic films — RCWA reflectance vs pyGTM 4x4 transfer matrix."""

    @pytest.mark.parametrize(
        "n_sub, n_film, n_sup, d_nm",
        [
            (1.5, 2.0 + 0.5j, 1.0, 150.0),      # moderately lossy
            (1.0, 0.5 + 3.0j, 1.52, 50.0),        # metallic
            (1.0, 1.5 + 0.001j, 1.0, 300.0),       # weakly lossy
            (2.0, 3.0 + 1.0j, 1.0, 80.0),          # high-index lossy
        ],
    )
    def test_lossy_film_reflectance(self, n_sub, n_film, n_sup, d_nm):
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_diag = [n_film**2, n_film**2, n_film**2]

        gtm = _pygtm_reflectances(
            wavelength_nm, n_sub**2, n_sup**2,
            [(d_nm, eps_diag, 0.0)],
        )
        stack = _make_anisotropic_stack(
            wavelength_nm, period_nm, n_sub**2, n_sup**2,
            [(d_nm, eps_diag, 0.0)],
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"
        # No cross-polarization for isotropic
        assert rcwa["Rps"] < 1e-15
        assert rcwa["Rsp"] < 1e-15

    def test_lossy_two_layer(self):
        wavelength_nm = 633.0
        period_nm = 500.0
        n_sub, n_sup = 1.5, 1.0
        n1, n2 = 2.0 + 0.1j, 1.3 + 0.5j
        d1, d2 = 100.0, 60.0
        eps1 = [n1**2] * 3
        eps2 = [n2**2] * 3

        gtm = _pygtm_reflectances(
            wavelength_nm, n_sub**2, n_sup**2,
            [(d1, eps1, 0.0), (d2, eps2, 0.0)],
        )
        stack = _make_anisotropic_stack(
            wavelength_nm, period_nm, n_sub**2, n_sup**2,
            [(d1, eps1, 0.0), (d2, eps2, 0.0)],
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8


# ---------------------------------------------------------------------------
# Tests: anisotropic media vs pyGTM
# ---------------------------------------------------------------------------

class TestAnisotropicVsPyGTM:
    """Anisotropic (birefringent) layers — RCWA reflectance vs pyGTM 4x4."""

    @pytest.mark.parametrize(
        "no, ne, angle_deg, d_nm",
        [
            (1.5, 1.7, 45.0, 200.0),     # in-plane birefringent, 45°
            (1.5, 1.7, 30.0, 200.0),     # 30° rotation
            (1.5, 1.7, 0.0, 200.0),      # aligned — no cross-polarization
            (1.5, 1.7, 90.0, 200.0),     # 90° — no cross-polarization
            (1.4, 2.0, 22.5, 150.0),     # large birefringence
        ],
    )
    def test_in_plane_birefringent(self, no, ne, angle_deg, d_nm):
        wavelength_nm = 633.0
        period_nm = 500.0
        # z-axis uses no (ordinary index)
        eps_diag = [no**2, ne**2, no**2]

        gtm = _pygtm_reflectances(
            wavelength_nm, 1.0, 1.0,
            [(d_nm, eps_diag, angle_deg)],
        )
        stack = _make_anisotropic_stack(
            wavelength_nm, period_nm, 1.0, 1.0,
            [(d_nm, eps_diag, angle_deg)],
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-8, f"Rps: {rcwa['Rps']} vs {gtm['Rps']}"
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-8, f"Rsp: {rcwa['Rsp']} vs {gtm['Rsp']}"

    def test_birefringent_on_glass_substrate(self):
        """Birefringent film on glass substrate, air superstrate."""
        wavelength_nm = 550.0
        period_nm = 400.0
        no, ne = 1.54, 1.75
        d_nm = 250.0
        angle_deg = 35.0
        eps_diag = [no**2, ne**2, no**2]

        gtm = _pygtm_reflectances(
            wavelength_nm, 1.52**2, 1.0,
            [(d_nm, eps_diag, angle_deg)],
        )
        stack = _make_anisotropic_stack(
            wavelength_nm, period_nm, 1.52**2, 1.0,
            [(d_nm, eps_diag, angle_deg)],
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-8
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-8

    def test_two_birefringent_layers(self):
        """Two birefringent layers with different orientations."""
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_diag1 = [1.5**2, 1.7**2, 1.5**2]
        eps_diag2 = [1.6**2, 1.8**2, 1.6**2]
        layers = [
            (100.0, eps_diag1, 30.0),
            (150.0, eps_diag2, 60.0),
        ]

        gtm = _pygtm_reflectances(633.0, 1.0, 1.0, layers)
        stack = _make_anisotropic_stack(633.0, period_nm, 1.0, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7

    def test_biaxial_layer(self):
        """Fully biaxial (three distinct principal values)."""
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_diag = [1.5**2, 1.7**2, 1.9**2]
        layers = [(200.0, eps_diag, 40.0)]

        gtm = _pygtm_reflectances(wavelength_nm, 1.0, 1.0, layers)
        stack = _make_anisotropic_stack(wavelength_nm, period_nm, 1.0, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-8
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-8


def _euler_matrix(theta: float, phi: float, psi: float) -> jnp.ndarray:
    """Build the Euler rotation matrix using pyGTM's convention."""
    euler = jnp.zeros((3, 3), dtype=jnp.complex128)
    euler[0, 0] = jnp.cos(psi) * jnp.cos(phi) - jnp.cos(theta) * jnp.sin(phi) * jnp.sin(psi)
    euler[0, 1] = -jnp.sin(psi) * jnp.cos(phi) - jnp.cos(theta) * jnp.sin(phi) * jnp.cos(psi)
    euler[0, 2] = jnp.sin(theta) * jnp.sin(phi)
    euler[1, 0] = jnp.cos(psi) * jnp.sin(phi) + jnp.cos(theta) * jnp.cos(phi) * jnp.sin(psi)
    euler[1, 1] = -jnp.sin(psi) * jnp.sin(phi) + jnp.cos(theta) * jnp.cos(phi) * jnp.cos(psi)
    euler[1, 2] = -jnp.sin(theta) * jnp.cos(phi)
    euler[2, 0] = jnp.sin(theta) * jnp.sin(psi)
    euler[2, 1] = jnp.sin(theta) * jnp.cos(psi)
    euler[2, 2] = jnp.cos(theta)
    return euler


def _rotate_eps_euler(eps_diag: list, theta: float, phi: float, psi: float) -> jnp.ndarray:
    """Rotate a diagonal permittivity tensor by Euler angles (pyGTM convention).

    pyGTM computes: eps_lab = euler^{-1} @ eps_crystal @ euler
    """
    euler = _euler_matrix(theta, phi, psi)
    eps_crystal = jnp.diag(jnp.asarray(eps_diag, dtype=jnp.complex128))
    return jnp.linalg.inv(euler) @ eps_crystal @ euler


def _pygtm_reflectances_euler(
    wavelength_nm: float,
    eps_incident: complex,
    eps_exit: complex,
    layers: list[tuple[float, list, float, float, float]],
    zeta: complex = 0.0,
) -> dict[str, float]:
    """pyGTM reflectances with full 3D Euler angles.

    layers: list of (thickness_nm, eps_diag_3, theta, phi, psi) in radians.
    zeta: in-plane wavevector kx/k0 (can be complex).
    """
    system = GTM.System()
    system.set_superstrate(GTM.Layer(epsilon1=_const_eps_fn(eps_incident)))
    system.set_substrate(GTM.Layer(epsilon1=_const_eps_fn(eps_exit)))

    for thickness_nm, eps_diag, theta, phi, psi in layers:
        system.add_layer(GTM.Layer(
            thickness=thickness_nm * 1e-9,
            epsilon1=_const_eps_fn(eps_diag[0]),
            epsilon2=_const_eps_fn(eps_diag[1]),
            epsilon3=_const_eps_fn(eps_diag[2]),
            theta=theta,
            phi=phi,
            psi=psi,
        ))

    frequency_hz = GTM.c_const / (wavelength_nm * 1e-9)
    system.initialize_sys(frequency_hz)
    system.calculate_GammaStar(frequency_hz, zeta)
    with jnp.errstate(divide="ignore", invalid="ignore"):
        r_out, _, _, _ = system.calculate_r_t(zeta)
    return {
        "Rpp": abs(r_out[0]) ** 2,
        "Rss": abs(r_out[2]) ** 2,
        "Rps": abs(r_out[1]) ** 2,
        "Rsp": abs(r_out[3]) ** 2,
    }


def _make_stack_euler(
    wavelength_nm: float,
    period_nm: float,
    eps_substrate: complex,
    eps_superstrate: complex,
    layers: list[tuple[float, list, float, float, float]],
    kappa_inv_nm: complex = 0.0,
) -> Stack:
    """Build a Stack from layers with full Euler angles."""
    rcwa_layers = []
    for thickness_nm, eps_diag, theta, phi, psi in layers:
        eps_tensor = _rotate_eps_euler(eps_diag, theta, phi, psi)
        rcwa_layers.append(
            Layer.uniform(thickness_nm, eps_tensor, (0.0, period_nm))
        )
    return Stack(
        layers=rcwa_layers,
        wavelength_nm=wavelength_nm,
        kappa_inv_nm=kappa_inv_nm,
        eps_substrate=eps_substrate,
        eps_superstrate=eps_superstrate,
    )


class TestFullyBiaxial3DRotationVsPyGTM:
    """Fully biaxial layers with arbitrary 3D Euler rotations vs pyGTM 4x4."""

    @pytest.mark.parametrize("seed", [42, 123, 7, 2024, 9999])
    def test_random_3d_rotation_single_layer(self, seed):
        """Single biaxial layer with random 3D orientation."""
        rng = jnp.random.default_rng(seed)
        wavelength_nm = 633.0
        period_nm = 500.0
        d_nm = 200.0

        # Random biaxial permittivities (all distinct real parts)
        n_vals = 1.3 + 0.5 * rng.random(3)
        eps_diag = [complex(n**2) for n in n_vals]

        # Random Euler angles
        theta = rng.uniform(0, jnp.pi)
        phi = rng.uniform(0, 2 * jnp.pi)
        psi = rng.uniform(0, 2 * jnp.pi)

        layers = [(d_nm, eps_diag, theta, phi, psi)]

        gtm = _pygtm_reflectances_euler(wavelength_nm, 1.0, 1.0, layers)
        stack = _make_stack_euler(wavelength_nm, period_nm, 1.0, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-8, f"Rps: {rcwa['Rps']} vs {gtm['Rps']}"
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-8, f"Rsp: {rcwa['Rsp']} vs {gtm['Rsp']}"

    @pytest.mark.parametrize("seed", [11, 77, 256])
    def test_random_3d_rotation_lossy_biaxial(self, seed):
        """Lossy biaxial layer with random 3D orientation."""
        rng = jnp.random.default_rng(seed)
        wavelength_nm = 550.0
        period_nm = 400.0
        d_nm = 150.0

        n_vals = (1.3 + 0.5 * rng.random(3)) + 1j * (0.01 + 0.05 * rng.random(3))
        eps_diag = [complex(n**2) for n in n_vals]

        theta = rng.uniform(0, jnp.pi)
        phi = rng.uniform(0, 2 * jnp.pi)
        psi = rng.uniform(0, 2 * jnp.pi)

        layers = [(d_nm, eps_diag, theta, phi, psi)]

        gtm = _pygtm_reflectances_euler(wavelength_nm, 1.52**2, 1.0, layers)
        stack = _make_stack_euler(wavelength_nm, period_nm, 1.52**2, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7, f"Rps: {rcwa['Rps']} vs {gtm['Rps']}"
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7, f"Rsp: {rcwa['Rsp']} vs {gtm['Rsp']}"

    @pytest.mark.parametrize("seed", [3, 42])
    def test_random_3d_two_layer_stack(self, seed):
        """Two biaxial layers with independent random 3D orientations."""
        rng = jnp.random.default_rng(seed)
        wavelength_nm = 633.0
        period_nm = 500.0

        layers = []
        for d_nm in [120.0, 180.0]:
            n_vals = 1.3 + 0.5 * rng.random(3)
            eps_diag = [complex(n**2) for n in n_vals]
            theta = rng.uniform(0, jnp.pi)
            phi = rng.uniform(0, 2 * jnp.pi)
            psi = rng.uniform(0, 2 * jnp.pi)
            layers.append((d_nm, eps_diag, theta, phi, psi))

        gtm = _pygtm_reflectances_euler(wavelength_nm, 1.0, 1.52**2, layers)
        stack = _make_stack_euler(wavelength_nm, period_nm, 1.0, 1.52**2, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7

    def test_random_3d_thick_layer(self):
        """Thick biaxial layer (many wavelengths) with random 3D rotation."""
        rng = jnp.random.default_rng(314)
        wavelength_nm = 633.0
        period_nm = 500.0
        d_nm = 2000.0  # ~3 wavelengths

        eps_diag = [1.5**2, 1.7**2, 1.9**2]
        theta = rng.uniform(0, jnp.pi)
        phi = rng.uniform(0, 2 * jnp.pi)
        psi = rng.uniform(0, 2 * jnp.pi)

        layers = [(d_nm, eps_diag, theta, phi, psi)]

        gtm = _pygtm_reflectances_euler(wavelength_nm, 1.0, 1.0, layers)
        stack = _make_stack_euler(wavelength_nm, period_nm, 1.0, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7


class TestAnisotropicLossyVsPyGTM:
    """Anisotropic + lossy media vs pyGTM."""

    def test_lossy_birefringent(self):
        """Lossy birefringent layer (e.g. dichroic crystal)."""
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_diag = [(1.5 + 0.02j)**2, (1.7 + 0.05j)**2, (1.5 + 0.02j)**2]
        layers = [(200.0, eps_diag, 45.0)]

        gtm = _pygtm_reflectances(wavelength_nm, 1.0, 1.0, layers)
        stack = _make_anisotropic_stack(wavelength_nm, period_nm, 1.0, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7

    def test_metal_film_on_birefringent(self):
        """Metal film on top of birefringent layer."""
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_metal = (0.3 + 3.5j)**2
        eps_diag_metal = [eps_metal, eps_metal, eps_metal]
        eps_diag_bire = [1.5**2, 1.7**2, 1.5**2]
        layers = [
            (30.0, eps_diag_metal, 0.0),
            (200.0, eps_diag_bire, 45.0),
        ]

        gtm = _pygtm_reflectances(wavelength_nm, 1.52**2, 1.0, layers)
        stack = _make_anisotropic_stack(wavelength_nm, period_nm, 1.52**2, 1.0, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7

    def test_three_layer_mixed_stack(self):
        """3-layer: lossy isotropic / birefringent / lossy isotropic."""
        wavelength_nm = 550.0
        period_nm = 400.0
        n_lossy = 2.0 + 0.3j
        eps_iso = [n_lossy**2] * 3
        eps_bire = [1.4**2, 1.8**2, 1.4**2]
        layers = [
            (50.0, eps_iso, 0.0),
            (150.0, eps_bire, 55.0),
            (50.0, eps_iso, 0.0),
        ]

        gtm = _pygtm_reflectances(wavelength_nm, 1.0, 1.52**2, layers)
        stack = _make_anisotropic_stack(wavelength_nm, period_nm, 1.0, 1.52**2, layers)
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7
        assert abs(rcwa["Rps"] - gtm["Rps"]) < 1e-7
        assert abs(rcwa["Rsp"] - gtm["Rsp"]) < 1e-7


# ---------------------------------------------------------------------------
# Tests: oblique / complex kx
# ---------------------------------------------------------------------------

def _zeta_to_kappa_inv_nm(zeta: complex, wavelength_nm: float) -> complex:
    """Convert normalized in-plane wavevector to kappa_inv_nm."""
    return zeta * 2 * jnp.pi / wavelength_nm


class TestComplexKxIsotropicVsFresnel:
    """Oblique and complex kx for isotropic media vs analytic Fresnel."""

    @staticmethod
    def _fresnel_oblique_te(n1, n2, zeta):
        q1 = jnp.sqrt(complex(n1**2 - zeta**2))
        q2 = jnp.sqrt(complex(n2**2 - zeta**2))
        r = (q1 - q2) / (q1 + q2)
        t = 2 * q1 / (q1 + q2)
        return r, t

    @staticmethod
    def _fresnel_oblique_tm(n1, n2, zeta):
        eps1, eps2 = n1**2, n2**2
        q1 = jnp.sqrt(complex(eps1 - zeta**2))
        q2 = jnp.sqrt(complex(eps2 - zeta**2))
        r = (eps2 * q1 - eps1 * q2) / (eps2 * q1 + eps1 * q2)
        t = 2 * eps2 * q1 / (eps2 * q1 + eps1 * q2)
        return r, t

    @pytest.mark.parametrize(
        "n_sub, n_sup, zeta",
        [
            (1.5, 1.0, 1.5 * jnp.sin(jnp.deg2rad(20.0))),    # oblique 20°
            (1.5, 1.0, 1.5 * jnp.sin(jnp.deg2rad(40.0))),    # oblique 40°
            (1.0, 1.5, 1.0 * jnp.sin(jnp.deg2rad(30.0))),    # air→glass
            (2.0, 1.0, 2.0 * jnp.sin(jnp.deg2rad(15.0))),    # high-index
            (1.5, 1.0, 1.2),                                   # TIR regime
        ],
    )
    def test_bare_interface_oblique_real_kx(self, n_sub, n_sup, zeta):
        """Bare interface at real oblique incidence."""
        wavelength_nm = 633.0
        period_nm = 500.0
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)
        stack = Stack(
            layers=[Layer.uniform(
                0.0, n_sub**2 * jnp.eye(3, dtype=jnp.complex128), (0.0, period_nm),
            )],
            wavelength_nm=wavelength_nm, kappa_inv_nm=kappa,
            eps_substrate=n_sub**2, eps_superstrate=n_sup**2,
        )
        r_te, t_te = _extract_rt(stack, N=0, pol="TE")
        r_tm, t_tm = _extract_rt(stack, N=0, pol="TM")
        r_te_ref, _ = self._fresnel_oblique_te(n_sub, n_sup, zeta)
        r_tm_ref, _ = self._fresnel_oblique_tm(n_sub, n_sup, zeta)
        assert abs(abs(r_te) - abs(r_te_ref)) < 1e-10, f"|rTE|: {abs(r_te)} vs {abs(r_te_ref)}"
        assert abs(abs(r_tm) - abs(r_tm_ref)) < 1e-10, f"|rTM|: {abs(r_tm)} vs {abs(r_tm_ref)}"

    @pytest.mark.parametrize(
        "zeta",
        [
            0.3 + 0.1j,
            0.3 - 0.1j,
            0.5 + 0.2j,
            0.1 - 0.3j,
            -0.2 + 0.15j,
        ],
    )
    def test_bare_interface_complex_kx(self, zeta):
        """Bare interface with complex kx."""
        wavelength_nm = 633.0
        period_nm = 500.0
        n_sub, n_sup = 1.5, 1.0
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)
        stack = Stack(
            layers=[Layer.uniform(
                0.0, n_sub**2 * jnp.eye(3, dtype=jnp.complex128), (0.0, period_nm),
            )],
            wavelength_nm=wavelength_nm, kappa_inv_nm=kappa,
            eps_substrate=n_sub**2, eps_superstrate=n_sup**2,
        )
        r_te, _ = _extract_rt(stack, N=0, pol="TE")
        r_tm, _ = _extract_rt(stack, N=0, pol="TM")
        r_te_ref, _ = self._fresnel_oblique_te(n_sub, n_sup, zeta)
        r_tm_ref, _ = self._fresnel_oblique_tm(n_sub, n_sup, zeta)
        assert abs(abs(r_te) - abs(r_te_ref)) < 1e-10, f"|rTE|: {abs(r_te)} vs {abs(r_te_ref)}"
        assert abs(abs(r_tm) - abs(r_tm_ref)) < 1e-10, f"|rTM|: {abs(r_tm)} vs {abs(r_tm_ref)}"


class TestComplexKxIsotropicFilmVs2x2TM:
    """Isotropic thin film at oblique / complex kx.

    Real oblique kx: compared against 2x2 TE characteristic matrix (unambiguous).
    Complex kx: compared against pyGTM 4x4 (avoids sqrt branch-cut ambiguity in 2x2).
    """

    @staticmethod
    def _tm_2x2_oblique_te(n_inc, n_exit, n_film, d_nm, wavelength_nm, zeta):
        q_inc = jnp.sqrt(complex(n_inc**2 - zeta**2))
        q_exit = jnp.sqrt(complex(n_exit**2 - zeta**2))
        q_film = jnp.sqrt(complex(n_film**2 - zeta**2))
        phase = -2 * jnp.pi * q_film * d_nm / wavelength_nm
        M = jnp.array([
            [jnp.cos(phase), 1j * jnp.sin(phase) / q_film],
            [1j * q_film * jnp.sin(phase), jnp.cos(phase)],
        ], dtype=jnp.complex128)
        B = M[0, 0] + M[0, 1] * q_exit
        C = M[1, 0] + M[1, 1] * q_exit
        r = (q_inc * B - C) / (q_inc * B + C)
        t = 2 * q_inc / (q_inc * B + C)
        return r, t

    @pytest.mark.parametrize(
        "n_sub, n_film, n_sup, d_nm, zeta",
        [
            (1.5, 2.0, 1.0, 150.0, 1.5 * jnp.sin(jnp.deg2rad(20.0))),
            (1.0, 1.5, 1.0, 100.0, jnp.sin(jnp.deg2rad(30.0))),
            (1.52, 1.38, 1.0, 99.6, 1.52 * jnp.sin(jnp.deg2rad(25.0))),
            (1.0, 1.5 + 0.01j, 1.0, 200.0, jnp.sin(jnp.deg2rad(35.0))),
        ],
    )
    def test_te_film_real_oblique_vs_2x2(self, n_sub, n_film, n_sup, d_nm, zeta):
        """Real oblique kx — compare complex r,t against 2x2 characteristic matrix."""
        wavelength_nm = 633.0
        period_nm = 500.0
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)
        stack = Stack(
            layers=[Layer.uniform(
                d_nm,
                jnp.asarray(n_film**2, dtype=jnp.complex128) * jnp.eye(3, dtype=jnp.complex128),
                (0.0, period_nm),
            )],
            wavelength_nm=wavelength_nm, kappa_inv_nm=kappa,
            eps_substrate=n_sub**2, eps_superstrate=n_sup**2,
        )
        r_rcwa, t_rcwa = _extract_rt(stack, N=0, pol="TE")
        r_ref, t_ref = self._tm_2x2_oblique_te(n_sub, n_sup, n_film, d_nm, wavelength_nm, zeta)
        assert abs(r_rcwa - r_ref) < 1e-6, f"TE r: {r_rcwa} vs {r_ref}"
        assert abs(t_rcwa - t_ref) < 1e-6, f"TE t: {t_rcwa} vs {t_ref}"

    @pytest.mark.parametrize(
        "n_sub, n_film, n_sup, d_nm, zeta",
        [
            (1.0, 1.5, 1.0, 200.0, 0.3 + 0.1j),
            (1.0, 1.5, 1.0, 200.0, 0.3 - 0.1j),
            (1.0, 2.0, 1.0, 150.0, 0.5 + 0.2j),
            (1.0, 1.5 + 0.01j, 1.0, 200.0, 0.2 - 0.15j),
            (1.5, 2.0, 1.0, 150.0, 0.4 + 0.2j),
            (1.52, 1.38, 1.0, 99.6, -0.2 + 0.15j),
        ],
    )
    def test_isotropic_film_complex_kx_vs_pygtm(self, n_sub, n_film, n_sup, d_nm, zeta):
        """Complex kx isotropic film — Rss and Rpp vs pyGTM."""
        wavelength_nm = 633.0
        period_nm = 500.0
        eps_film = complex(n_film**2)
        eps_sub = complex(n_sub**2)
        eps_sup = complex(n_sup**2)
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)

        layers_euler = [(d_nm, [eps_film, eps_film, eps_film], 0.0, 0.0, 0.0)]
        gtm = _pygtm_reflectances_euler(wavelength_nm, eps_sub, eps_sup, layers_euler, zeta=zeta)

        stack = _make_stack_euler(
            wavelength_nm, period_nm, eps_sub, eps_sup, layers_euler, kappa_inv_nm=kappa,
        )
        rcwa = _extract_reflectances(stack, N=0)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-10, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-10, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"


class TestComplexKxAnisotropicVsPyGTM:
    """Anisotropic layers with oblique / complex kx — Rpp, Rss vs pyGTM."""

    @pytest.mark.parametrize(
        "zeta_desc, zeta",
        [
            ("oblique_20", 1.5 * jnp.sin(jnp.deg2rad(20.0))),
            ("oblique_40", 1.5 * jnp.sin(jnp.deg2rad(40.0))),
            ("complex_pos_im", 0.3 + 0.1j),
            ("complex_neg_im", 0.3 - 0.1j),
            ("complex_large", 0.5 + 0.2j),
            ("complex_neg_both", -0.2 + 0.15j),
        ],
    )
    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_biaxial_3d_rotation_complex_kx(self, zeta_desc, zeta, seed):
        """Fully biaxial layer with random 3D rotation at oblique/complex kx."""
        rng = jnp.random.default_rng(seed)
        wavelength_nm = 633.0
        period_nm = 500.0
        d_nm = 200.0

        n_vals = 1.3 + 0.5 * rng.random(3)
        eps_diag = [complex(n**2) for n in n_vals]

        theta = rng.uniform(0, jnp.pi)
        phi = rng.uniform(0, 2 * jnp.pi)
        psi = rng.uniform(0, 2 * jnp.pi)

        layers = [(d_nm, eps_diag, theta, phi, psi)]
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)

        gtm = _pygtm_reflectances_euler(
            wavelength_nm, 1.5**2, 1.0, layers, zeta=zeta,
        )
        stack = _make_stack_euler(
            wavelength_nm, period_nm, 1.5**2, 1.0, layers, kappa_inv_nm=kappa,
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-8, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-8, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"

    @pytest.mark.parametrize(
        "zeta",
        [
            1.5 * jnp.sin(jnp.deg2rad(25.0)),
            0.4 + 0.15j,
            0.4 - 0.15j,
        ],
    )
    def test_lossy_biaxial_complex_kx(self, zeta):
        """Lossy biaxial with random 3D rotation at complex kx."""
        rng = jnp.random.default_rng(99)
        wavelength_nm = 550.0
        period_nm = 400.0
        d_nm = 150.0

        n_vals = (1.3 + 0.5 * rng.random(3)) + 1j * (0.01 + 0.05 * rng.random(3))
        eps_diag = [complex(n**2) for n in n_vals]
        theta = rng.uniform(0, jnp.pi)
        phi = rng.uniform(0, 2 * jnp.pi)
        psi = rng.uniform(0, 2 * jnp.pi)

        layers = [(d_nm, eps_diag, theta, phi, psi)]
        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)

        gtm = _pygtm_reflectances_euler(
            wavelength_nm, 1.52**2, 1.0, layers, zeta=zeta,
        )
        stack = _make_stack_euler(
            wavelength_nm, period_nm, 1.52**2, 1.0, layers, kappa_inv_nm=kappa,
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7, f"Rss: {rcwa['Rss']} vs {gtm['Rss']}"
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7, f"Rpp: {rcwa['Rpp']} vs {gtm['Rpp']}"

    @pytest.mark.parametrize(
        "zeta",
        [
            1.0 * jnp.sin(jnp.deg2rad(30.0)),
            1.52 * jnp.sin(jnp.deg2rad(15.0)),
        ],
    )
    def test_two_layer_stack_complex_kx(self, zeta):
        """Two-layer anisotropic stack at oblique real kx.

        pyGTM's mode classifier crashes on complex kx with anisotropic multi-layer
        stacks, so we test only real oblique incidence here.  Complex kx is already
        covered by the single-layer tests above.
        """
        rng = jnp.random.default_rng(55)
        wavelength_nm = 633.0
        period_nm = 500.0

        layers = []
        for d_nm in [120.0, 180.0]:
            n_vals = 1.3 + 0.5 * rng.random(3)
            eps_diag = [complex(n**2) for n in n_vals]
            theta = rng.uniform(0, jnp.pi)
            phi = rng.uniform(0, 2 * jnp.pi)
            psi = rng.uniform(0, 2 * jnp.pi)
            layers.append((d_nm, eps_diag, theta, phi, psi))

        kappa = _zeta_to_kappa_inv_nm(zeta, wavelength_nm)

        gtm = _pygtm_reflectances_euler(
            wavelength_nm, 1.0, 1.52**2, layers, zeta=zeta,
        )
        stack = _make_stack_euler(
            wavelength_nm, period_nm, 1.0, 1.52**2, layers, kappa_inv_nm=kappa,
        )
        rcwa = _extract_reflectances(stack)

        assert abs(rcwa["Rss"] - gtm["Rss"]) < 1e-7
        assert abs(rcwa["Rpp"] - gtm["Rpp"]) < 1e-7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
