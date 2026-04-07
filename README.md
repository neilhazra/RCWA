# Organized RCWA Package

This is a sibling package to `refactored_rcwa`, with the same public API:

```python
from rcwa import Layer, Stack, Solver
```

The code is split more strictly by responsibility:

- `rcwa/layer.py`
  Local layer physics only: dielectric sampling, Fourier coefficients, and per-layer Q assembly.
- `rcwa/stack.py`
  Stack-wide geometry, half-space Q tensors, harmonic indexing, and isotropic mode utilities.
- `rcwa/solver.py`
  Basis changes, transfer/scattering conversions, Redheffer chaining, and reflection/transmission solves.

Run tests with the existing virtual environment:

```bash
cd organized_rcwa
PYTHONPATH=. ../myenv/bin/pytest -q
```
