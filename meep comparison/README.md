# Meep Comparison

This folder contains a direct `PyMeep` vs `RCWAFFF` x-z cross-section comparison for structures that are:

- periodic in `x`
- stratified in `z`
- homogeneous in `y`

The current comparison cases are:

- `uniform_slab`
- `binary_grating_on_slab`

For each device and each incident polarization:

- `TE` compares normalized `|E_y|`
- `TM` compares normalized `|H_y|`

The script samples only the finite device region, not the semi-infinite substrate/superstrate buffers.

## Environment

Per the official Meep installation docs, `PyMeep` was installed with Conda on April 9, 2026 into:

- Miniconda prefix: `/tmp/miniconda-meep`
- Meep env: `/tmp/meep-env`

The installed Meep version is `1.33.0-beta`.

## Run

```bash
/tmp/meep-env/bin/python "RCWAFFF/meep comparison/compare_meep_rcwa_xz.py"
```

Outputs are written to:

- `RCWAFFF/meep comparison/output/*.npz`
- `RCWAFFF/meep comparison/output/*.png`
- `RCWAFFF/meep comparison/output/summary.json`

The `.npz` files include the raw stitched RCWA fields, the raw Meep DFT fields, and the normalized comparison grids used for the plots.
