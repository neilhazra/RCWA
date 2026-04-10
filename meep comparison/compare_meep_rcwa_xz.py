"""Compare normalized x-z cross sections from RCWAFFF and PyMeep.

This script is intended to be run with a Python interpreter that already has
``meep`` installed. The current setup on this machine is ``/tmp/meep-env``.

The comparison is restricted to devices that are periodic in x, stratified in z,
and homogeneous in y. Only the finite device region is compared.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import meep as mp
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rcwa import Layer, Stack
from rcwa.visualize import VisualizationBundle, compute_visualization_bundle


OUTPUT_DIR = pathlib.Path(__file__).with_name("output")
INCIDENT_POLS = ("TE", "TM")
RCWA_COMPONENTS_BY_POL = {"TE": "E_y", "TM": "-H_y"}
MEEP_COMPONENT_LABELS = {"TE": "E_y", "TM": "H_y"}
MEEP_SOURCE_COMPONENTS = {"TE": mp.Ez, "TM": mp.Hz}

RCWA_ORDER = 512
RCWA_NUM_POINTS_X = 320
RCWA_NUM_POINTS_Z = 181
RCWA_NUM_POINTS_FOURIER = 4096

MEEP_RESOLUTION = 256
MEEP_GAUSSIAN_FWIDTH_FACTOR = 0.15
MEEP_DECAY_STEPS = 256
MEEP_DECAY_THRESHOLD = 1e-8

SUBSTRATE_BUFFER_NM = 600.0
SUPERSTRATE_BUFFER_NM = 600.0
PML_THICKNESS_NM = 400.0
SOURCE_OFFSET_FROM_PML_NM = 150.0


def _rcwa_cache_tag() -> str:
    return (
        f"rcwa_N{RCWA_ORDER}"
        f"_nx{RCWA_NUM_POINTS_X}"
        f"_nz{RCWA_NUM_POINTS_Z}"
        f"_nfft{RCWA_NUM_POINTS_FOURIER}"
    )


def _meep_cache_tag() -> str:
    return (
        f"meep_res{MEEP_RESOLUTION}"
        f"_fw{MEEP_GAUSSIAN_FWIDTH_FACTOR:g}"
        f"_decay{MEEP_DECAY_STEPS}"
        f"_thr{MEEP_DECAY_THRESHOLD:g}"
    )


@dataclass(frozen=True)
class LayerSpec:
    thickness_nm: float
    eps_high: float
    eps_low: float | None = None
    fill_fraction: float = 1.0

    @property
    def is_uniform(self) -> bool:
        return self.eps_low is None or abs(self.fill_fraction - 1.0) < 1e-12

    def rcwa_layer(self, x_domain_nm: tuple[float, float], period_nm: float) -> Layer:
        high_tensor = self.eps_high * np.eye(3, dtype=np.complex128)
        if self.is_uniform:
            return Layer.uniform(
                thickness_nm=self.thickness_nm,
                eps_tensor=high_tensor,
                x_domain_nm=x_domain_nm,
            )

        assert self.eps_low is not None
        low_tensor = self.eps_low * np.eye(3, dtype=np.complex128)
        ridge_width_nm = self.fill_fraction * period_nm
        ridge_left_nm = -0.5 * ridge_width_nm
        ridge_right_nm = 0.5 * ridge_width_nm
        segments = [
            (x_domain_nm[0], ridge_left_nm, low_tensor),
            (ridge_left_nm, ridge_right_nm, high_tensor),
            (ridge_right_nm, x_domain_nm[1], low_tensor),
        ]
        return Layer.piecewise(
            thickness_nm=self.thickness_nm,
            x_domain_nm=x_domain_nm,
            segments=segments,
        )


@dataclass(frozen=True)
class DeviceSpec:
    name: str
    title: str
    wavelength_nm: float
    period_nm: float
    eps_substrate: float
    eps_superstrate: float
    layers: tuple[LayerSpec, ...]

    @property
    def total_thickness_nm(self) -> float:
        return float(sum(layer.thickness_nm for layer in self.layers))

    @property
    def x_domain_nm(self) -> tuple[float, float]:
        return (-0.5 * self.period_nm, 0.5 * self.period_nm)


DEVICE_SPECS = (
    DeviceSpec(
        name="uniform_slab",
        title="Uniform Slab",
        wavelength_nm=633.0,
        period_nm=400.0,
        eps_substrate=1.45**2,
        eps_superstrate=1.0**2,
        layers=(
            LayerSpec(
                thickness_nm=180.0,
                eps_high=2.0**2,
            ),
        ),
    ),
    DeviceSpec(
        name="binary_grating_on_slab",
        title="Binary Grating on Slab",
        wavelength_nm=633.0,
        period_nm=400.0,
        eps_substrate=1.45**2,
        eps_superstrate=1.0**2,
        layers=(
            LayerSpec(
                thickness_nm=120.0,
                eps_high=2.0**2,
            ),
            LayerSpec(
                thickness_nm=60.0,
                eps_high=2.0**2,
                eps_low=1.0**2,
                fill_fraction=0.5,
            ),
        ),
    ),
)


def _build_rcwa_stack(spec: DeviceSpec) -> Stack:
    stack = Stack(
        wavelength_nm=spec.wavelength_nm,
        kappa_inv_nm=0.0,
        eps_substrate=spec.eps_substrate,
        eps_superstrate=spec.eps_superstrate,
    )
    for layer_spec in spec.layers:
        stack.add_layer(layer_spec.rcwa_layer(spec.x_domain_nm, spec.period_nm))
    return stack


def _stitch_rcwa_bundle(
    bundle: VisualizationBundle,
    incident_pol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pol_data = bundle.incident_data(incident_pol)

    x_nm: np.ndarray | None = None
    z_segments_nm: list[np.ndarray] = []
    field_segments: list[np.ndarray] = []
    z_offset_nm = 0.0

    for layer_index, layer_profile in enumerate(pol_data.layer_profiles):
        layer_x_nm = np.asarray(layer_profile.x_nm, dtype=np.float64)
        layer_z_nm = np.asarray(layer_profile.z_nm, dtype=np.float64) + z_offset_nm
        layer_field = np.asarray(layer_profile.field_xz, dtype=np.complex128)

        if x_nm is None:
            x_nm = layer_x_nm
        elif not np.allclose(x_nm, layer_x_nm, atol=1e-12, rtol=1e-12):
            raise ValueError("All stitched RCWA layers must share the same x grid.")

        if layer_index > 0:
            layer_z_nm = layer_z_nm[1:]
            layer_field = layer_field[1:, :]

        z_segments_nm.append(layer_z_nm)
        field_segments.append(layer_field)
        z_offset_nm += float(layer_profile.z_nm[-1])

    if x_nm is None:
        raise ValueError("Visualization bundle does not contain any layer profiles.")

    return (
        x_nm,
        np.concatenate(z_segments_nm, axis=0),
        np.concatenate(field_segments, axis=0),
    )


def _compute_or_load_rcwa_bundle(
    spec: DeviceSpec,
    output_dir: pathlib.Path,
    force: bool,
) -> VisualizationBundle:
    cache_path = output_dir / f"{spec.name}_{_rcwa_cache_tag()}_rcwa_bundle.npz"
    if cache_path.exists() and not force:
        return VisualizationBundle.load(cache_path)

    stack = _build_rcwa_stack(spec)
    return compute_visualization_bundle(
        stack,
        N=RCWA_ORDER,
        num_points_x=RCWA_NUM_POINTS_X,
        num_points_z=RCWA_NUM_POINTS_Z,
        num_points_rcwa=RCWA_NUM_POINTS_FOURIER,
        incident_polarizations=INCIDENT_POLS,
        components_by_pol=RCWA_COMPONENTS_BY_POL,
        cache_path=cache_path,
        verbose=True,
    )


def _medium_from_eps(eps_scalar: float) -> mp.Medium:
    return mp.Medium(index=float(np.sqrt(np.real(eps_scalar))))


def _build_meep_geometry(
    spec: DeviceSpec,
    cell_height_um: float,
    device_bottom_um: float,
) -> list[mp.GeometricObject]:
    geometry: list[mp.GeometricObject] = []

    substrate_height_um = SUBSTRATE_BUFFER_NM / 1000.0
    substrate_center_um = -0.5 * cell_height_um + (PML_THICKNESS_NM / 1000.0) + 0.5 * substrate_height_um
    geometry.append(
        mp.Block(
            center=mp.Vector3(0.0, substrate_center_um, 0.0),
            size=mp.Vector3(mp.inf, substrate_height_um, mp.inf),
            material=_medium_from_eps(spec.eps_substrate),
        )
    )

    z_cursor_um = device_bottom_um
    period_um = spec.period_nm / 1000.0
    for layer_spec in spec.layers:
        thickness_um = layer_spec.thickness_nm / 1000.0
        center_y_um = z_cursor_um + 0.5 * thickness_um

        if layer_spec.is_uniform:
            geometry.append(
                mp.Block(
                    center=mp.Vector3(0.0, center_y_um, 0.0),
                    size=mp.Vector3(mp.inf, thickness_um, mp.inf),
                    material=_medium_from_eps(layer_spec.eps_high),
                )
            )
        else:
            assert layer_spec.eps_low is not None
            geometry.append(
                mp.Block(
                    center=mp.Vector3(0.0, center_y_um, 0.0),
                    size=mp.Vector3(mp.inf, thickness_um, mp.inf),
                    material=_medium_from_eps(layer_spec.eps_low),
                )
            )
            geometry.append(
                mp.Block(
                    center=mp.Vector3(0.0, center_y_um, 0.0),
                    size=mp.Vector3(period_um * layer_spec.fill_fraction, thickness_um, mp.inf),
                    material=_medium_from_eps(layer_spec.eps_high),
                )
            )

        z_cursor_um += thickness_um

    return geometry


def _compute_meep_fields(
    spec: DeviceSpec,
    incident_pol: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wavelength_um = spec.wavelength_nm / 1000.0
    period_um = spec.period_nm / 1000.0
    total_thickness_um = spec.total_thickness_nm / 1000.0
    pml_um = PML_THICKNESS_NM / 1000.0
    substrate_buffer_um = SUBSTRATE_BUFFER_NM / 1000.0
    superstrate_buffer_um = SUPERSTRATE_BUFFER_NM / 1000.0
    source_offset_um = SOURCE_OFFSET_FROM_PML_NM / 1000.0

    cell_height_um = total_thickness_um + substrate_buffer_um + superstrate_buffer_um + 2.0 * pml_um
    device_bottom_um = -0.5 * cell_height_um + pml_um + substrate_buffer_um
    device_center_um = device_bottom_um + 0.5 * total_thickness_um
    source_y_um = 0.5 * cell_height_um - pml_um - source_offset_um

    freq = 1.0 / wavelength_um
    source_component = MEEP_SOURCE_COMPONENTS[incident_pol]

    sim = mp.Simulation(
        cell_size=mp.Vector3(period_um, cell_height_um, 0.0),
        resolution=MEEP_RESOLUTION,
        k_point=mp.Vector3(),
        default_material=_medium_from_eps(spec.eps_superstrate),
        geometry=_build_meep_geometry(spec, cell_height_um, device_bottom_um),
        boundary_layers=[mp.PML(pml_um, direction=mp.Y)],
        sources=[
            mp.Source(
                src=mp.GaussianSource(
                    frequency=freq,
                    fwidth=MEEP_GAUSSIAN_FWIDTH_FACTOR * freq,
                ),
                component=source_component,
                center=mp.Vector3(0.0, source_y_um, 0.0),
                size=mp.Vector3(period_um, 0.0, 0.0),
            )
        ],
    )

    dft_fields = sim.add_dft_fields(
        [source_component],
        freq,
        0.0,
        1,
        center=mp.Vector3(0.0, device_center_um, 0.0),
        size=mp.Vector3(period_um, total_thickness_um, 0.0),
    )
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            MEEP_DECAY_STEPS,
            source_component,
            mp.Vector3(0.0, device_center_um, 0.0),
            MEEP_DECAY_THRESHOLD,
        )
    )

    field_xz_native = np.asarray(sim.get_dft_array(dft_fields, source_component, 0), dtype=np.complex128)
    x_um, y_um, _, _ = sim.get_array_metadata(dft_cell=dft_fields)
    sim.reset_meep()

    x_nm = np.asarray(x_um, dtype=np.float64) * 1000.0
    z_nm = (np.asarray(y_um, dtype=np.float64) - device_bottom_um) * 1000.0
    field_zx = field_xz_native.T
    return x_nm, z_nm, field_zx


def _compute_or_load_meep_fields(
    spec: DeviceSpec,
    incident_pol: str,
    output_dir: pathlib.Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = output_dir / (
        f"{spec.name}_{incident_pol.lower()}_{_meep_cache_tag()}_meep_fields.npz"
    )
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as data:
            return (
                np.asarray(data["x_nm"], dtype=np.float64),
                np.asarray(data["z_nm"], dtype=np.float64),
                np.asarray(data["field_zx"], dtype=np.complex128),
            )

    x_nm, z_nm, field_zx = _compute_meep_fields(spec, incident_pol)
    np.savez_compressed(
        cache_path,
        x_nm=x_nm,
        z_nm=z_nm,
        field_zx=field_zx,
        incident_pol=np.asarray(incident_pol),
        component_label=np.asarray(MEEP_COMPONENT_LABELS[incident_pol]),
        meep_version=np.asarray(mp.__version__),
        meep_resolution=np.asarray(MEEP_RESOLUTION),
    )
    return x_nm, z_nm, field_zx


def _normalized_magnitude(
    field: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, float]:
    magnitude = np.abs(field)
    finite_mask = np.isfinite(magnitude)
    finite_values = magnitude[finite_mask]
    if finite_values.size == 0:
        raise ValueError(
            f"{label} contains no finite values. "
            "If this is the RCWA field, the current Fourier order is likely too high for "
            "the present visualization propagation path. Try lowering RCWA_ORDER or "
            "rerun with a smaller --rcwa-order."
        )
    if finite_values.size != magnitude.size:
        raise ValueError(
            f"{label} contains non-finite values "
            f"({magnitude.size - finite_values.size} / {magnitude.size} entries). "
            "If this is the RCWA field, try a smaller --rcwa-order."
        )
    scale = float(np.max(finite_values))
    if scale <= 0.0:
        return magnitude, 1.0
    return magnitude / scale, scale


def _interp_meep_to_rcwa_grid(
    meep_x_nm: np.ndarray,
    meep_z_nm: np.ndarray,
    meep_values_zx: np.ndarray,
    rcwa_x_nm: np.ndarray,
    rcwa_z_nm: np.ndarray,
) -> np.ndarray:
    interpolator = RegularGridInterpolator(
        (meep_z_nm, meep_x_nm),
        meep_values_zx,
        bounds_error=False,
        fill_value=None,
    )
    z_mesh_nm, x_mesh_nm = np.meshgrid(rcwa_z_nm, rcwa_x_nm, indexing="ij")
    points = np.column_stack([z_mesh_nm.ravel(), x_mesh_nm.ravel()])
    return interpolator(points).reshape(z_mesh_nm.shape)


def _plot_overlay_polyline(
    ax,
    x_nm: np.ndarray,
    z_nm: np.ndarray,
) -> None:
    """Draw one high-contrast geometry line on top of a field image."""
    ax.plot(x_nm, z_nm, color="black", lw=2.8, alpha=0.9, zorder=10)
    ax.plot(x_nm, z_nm, color="white", lw=1.2, alpha=0.95, zorder=11)


def _layer_geometry_rectangles(
    spec: DeviceSpec,
) -> list[tuple[float, float, float, float]]:
    """Return rectangular geometry features as ``(x0, x1, z0, z1)`` in nm."""
    rectangles: list[tuple[float, float, float, float]] = []
    x0_nm, x1_nm = spec.x_domain_nm
    z0_nm = 0.0
    for layer_spec in spec.layers:
        z1_nm = z0_nm + layer_spec.thickness_nm
        rectangles.append((x0_nm, x1_nm, z0_nm, z1_nm))
        if not layer_spec.is_uniform:
            ridge_width_nm = layer_spec.fill_fraction * spec.period_nm
            ridge_x0_nm = -0.5 * ridge_width_nm
            ridge_x1_nm = 0.5 * ridge_width_nm
            rectangles.append((ridge_x0_nm, ridge_x1_nm, z0_nm, z1_nm))
        z0_nm = z1_nm
    return rectangles


def _overlay_device_geometry(ax, spec: DeviceSpec) -> None:
    """Overlay the layer geometry outline on one comparison panel."""
    x0_nm, x1_nm = spec.x_domain_nm
    for rect_x0_nm, rect_x1_nm, rect_z0_nm, rect_z1_nm in _layer_geometry_rectangles(spec):
        _plot_overlay_polyline(
            ax,
            np.array([rect_x0_nm, rect_x1_nm], dtype=np.float64),
            np.array([rect_z0_nm, rect_z0_nm], dtype=np.float64),
        )
        _plot_overlay_polyline(
            ax,
            np.array([rect_x0_nm, rect_x1_nm], dtype=np.float64),
            np.array([rect_z1_nm, rect_z1_nm], dtype=np.float64),
        )
        _plot_overlay_polyline(
            ax,
            np.array([rect_x0_nm, rect_x0_nm], dtype=np.float64),
            np.array([rect_z0_nm, rect_z1_nm], dtype=np.float64),
        )
        _plot_overlay_polyline(
            ax,
            np.array([rect_x1_nm, rect_x1_nm], dtype=np.float64),
            np.array([rect_z0_nm, rect_z1_nm], dtype=np.float64),
        )

    ax.set_xlim(x0_nm, x1_nm)
    ax.set_ylim(0.0, spec.total_thickness_nm)


def _plot_comparison(
    spec: DeviceSpec,
    incident_pol: str,
    rcwa_x_nm: np.ndarray,
    rcwa_z_nm: np.ndarray,
    rcwa_norm_zx: np.ndarray,
    meep_norm_on_rcwa_zx: np.ndarray,
    diff_zx: np.ndarray,
    metrics: dict[str, float],
    output_dir: pathlib.Path,
    show: bool,
) -> pathlib.Path:
    extent = (float(rcwa_x_nm[0]), float(rcwa_x_nm[-1]), float(rcwa_z_nm[0]), float(rcwa_z_nm[-1]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    panels = (
        (rcwa_norm_zx, "RCWA normalized"),
        (meep_norm_on_rcwa_zx, f"Meep normalized ({mp.__version__})"),
        (diff_zx, "Absolute difference"),
    )
    cmaps = ("magma", "magma", "viridis")
    vmaxes = (1.0, 1.0, None)

    for ax, (values, title), cmap, vmax in zip(axes, panels, cmaps, vmaxes, strict=True):
        image = ax.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_xlabel("x (nm)")
        ax.set_ylabel("z from substrate interface (nm)")
        ax.set_title(title)
        _overlay_device_geometry(ax, spec)
        fig.colorbar(image, ax=ax, shrink=0.9)

    component_label = RCWA_COMPONENTS_BY_POL[incident_pol]
    fig.suptitle(
        f"{spec.title} | {incident_pol} incidence | normalized |{component_label}|"
        f"\nRMS diff = {metrics['rms_difference']:.4e}, max diff = {metrics['max_difference']:.4e}"
    )

    out_path = output_dir / f"{spec.name}_{incident_pol.lower()}_normalized_compare.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _save_comparison_npz(
    spec: DeviceSpec,
    incident_pol: str,
    output_dir: pathlib.Path,
    rcwa_x_nm: np.ndarray,
    rcwa_z_nm: np.ndarray,
    rcwa_norm_zx: np.ndarray,
    meep_x_nm: np.ndarray,
    meep_z_nm: np.ndarray,
    meep_norm_native_zx: np.ndarray,
    meep_norm_on_rcwa_zx: np.ndarray,
    diff_zx: np.ndarray,
    metrics: dict[str, float],
) -> pathlib.Path:
    out_path = output_dir / f"{spec.name}_{incident_pol.lower()}_comparison_data.npz"
    np.savez_compressed(
        out_path,
        rcwa_x_nm=rcwa_x_nm,
        rcwa_z_nm=rcwa_z_nm,
        rcwa_normalized_zx=rcwa_norm_zx,
        meep_x_nm=meep_x_nm,
        meep_z_nm=meep_z_nm,
        meep_normalized_native_zx=meep_norm_native_zx,
        meep_normalized_on_rcwa_grid_zx=meep_norm_on_rcwa_zx,
        absolute_difference_zx=diff_zx,
        metrics_json=np.asarray(json.dumps(metrics)),
    )
    return out_path


def _run_case(
    spec: DeviceSpec,
    incident_pol: str,
    output_dir: pathlib.Path,
    force: bool,
    show: bool,
) -> dict[str, object]:
    print(f"[compare] {spec.name} | {incident_pol}")

    bundle = _compute_or_load_rcwa_bundle(spec, output_dir=output_dir, force=force)
    rcwa_x_nm, rcwa_z_nm, rcwa_field_zx = _stitch_rcwa_bundle(bundle, incident_pol)
    meep_x_nm, meep_z_nm, meep_field_zx = _compute_or_load_meep_fields(
        spec,
        incident_pol=incident_pol,
        output_dir=output_dir,
        force=force,
    )

    rcwa_norm_zx, rcwa_scale = _normalized_magnitude(
        rcwa_field_zx,
        label=f"RCWA field for {spec.name} {incident_pol}",
    )
    meep_norm_native_zx, meep_scale = _normalized_magnitude(
        meep_field_zx,
        label=f"Meep field for {spec.name} {incident_pol}",
    )
    meep_norm_on_rcwa_zx = _interp_meep_to_rcwa_grid(
        meep_x_nm,
        meep_z_nm,
        meep_norm_native_zx,
        rcwa_x_nm,
        rcwa_z_nm,
    )
    diff_zx = np.abs(rcwa_norm_zx - meep_norm_on_rcwa_zx)

    metrics = {
        "rcwa_native_max_abs": rcwa_scale,
        "meep_native_max_abs": meep_scale,
        "rms_difference": float(np.sqrt(np.mean(diff_zx**2))),
        "max_difference": float(np.max(diff_zx)),
        "mean_difference": float(np.mean(diff_zx)),
        "meep_version": mp.__version__,
    }

    comparison_npz = _save_comparison_npz(
        spec,
        incident_pol=incident_pol,
        output_dir=output_dir,
        rcwa_x_nm=rcwa_x_nm,
        rcwa_z_nm=rcwa_z_nm,
        rcwa_norm_zx=rcwa_norm_zx,
        meep_x_nm=meep_x_nm,
        meep_z_nm=meep_z_nm,
        meep_norm_native_zx=meep_norm_native_zx,
        meep_norm_on_rcwa_zx=meep_norm_on_rcwa_zx,
        diff_zx=diff_zx,
        metrics=metrics,
    )
    figure_path = _plot_comparison(
        spec,
        incident_pol=incident_pol,
        rcwa_x_nm=rcwa_x_nm,
        rcwa_z_nm=rcwa_z_nm,
        rcwa_norm_zx=rcwa_norm_zx,
        meep_norm_on_rcwa_zx=meep_norm_on_rcwa_zx,
        diff_zx=diff_zx,
        metrics=metrics,
        output_dir=output_dir,
        show=show,
    )

    print(
        f"[compare] saved {figure_path.name} | rms={metrics['rms_difference']:.4e} "
        f"max={metrics['max_difference']:.4e}"
    )
    return {
        "metrics": metrics,
        "comparison_npz": str(comparison_npz),
        "figure": str(figure_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="all",
        choices=["all", *[spec.name for spec in DEVICE_SPECS]],
        help="Run one device or all configured devices.",
    )
    parser.add_argument(
        "--polarization",
        default="all",
        choices=["all", *INCIDENT_POLS],
        help="Run one incident polarization or both.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=OUTPUT_DIR,
        help="Directory for caches and plots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute RCWA and Meep caches even if cached files already exist.",
    )
    parser.add_argument(
        "--rcwa-order",
        type=int,
        default=None,
        help="Override the RCWA Fourier order N for this run.",
    )
    parser.add_argument(
        "--rcwa-fourier-samples",
        type=int,
        default=None,
        help="Override the real-space sample count used to build RCWA Fourier coefficients.",
    )
    parser.add_argument(
        "--meep-resolution",
        type=int,
        default=None,
        help="Override the Meep spatial resolution in pixels per micron.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display matplotlib windows in addition to saving figures.",
    )
    return parser.parse_args()


def main() -> None:
    global RCWA_ORDER, RCWA_NUM_POINTS_FOURIER, MEEP_RESOLUTION

    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rcwa_order is not None:
        RCWA_ORDER = args.rcwa_order
    if args.rcwa_fourier_samples is not None:
        RCWA_NUM_POINTS_FOURIER = args.rcwa_fourier_samples
    if args.meep_resolution is not None:
        MEEP_RESOLUTION = args.meep_resolution

    selected_devices = (
        DEVICE_SPECS
        if args.device == "all"
        else tuple(spec for spec in DEVICE_SPECS if spec.name == args.device)
    )
    selected_pols = INCIDENT_POLS if args.polarization == "all" else (args.polarization,)

    summary: dict[str, dict[str, object]] = {
        "_meta": {
            "meep_version": mp.__version__,
            "rcwa_order": RCWA_ORDER,
            "rcwa_num_points_x": RCWA_NUM_POINTS_X,
            "rcwa_num_points_z": RCWA_NUM_POINTS_Z,
            "rcwa_num_points_fourier": RCWA_NUM_POINTS_FOURIER,
            "meep_resolution": MEEP_RESOLUTION,
        }
    }

    for spec in selected_devices:
        summary[spec.name] = {}
        for incident_pol in selected_pols:
            summary[spec.name][incident_pol] = _run_case(
                spec,
                incident_pol=incident_pol,
                output_dir=output_dir,
                force=args.force,
                show=args.show,
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[compare] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
