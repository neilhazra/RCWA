from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from rcwa import Layer, Stack


@pytest.fixture
def sample_problem() -> dict:
    period_nm = 500.0
    x_domain_nm = (0.0, period_nm)

    eps_sio2_405 = 1.4696**2
    eps_nbocl2_405 = np.diag([5.406, 10.465 - 1.923j, 1.6**2])
    n_si_405 = 5.57 - 0.387j
    eps_si_405 = n_si_405**2
    eps_air = 1.0

    stack = Stack(
        wavelength_nm=405.0,
        kappa_inv_nm=0.0,
        eps_substrate=eps_sio2_405,
        eps_superstrate=eps_air,
    )
    stack.add_layer(Layer.uniform(90.0, eps_sio2_405 * jnp.eye(3), x_domain_nm=x_domain_nm))
    stack.add_layer(Layer.uniform(100.0, jnp.array(eps_nbocl2_405), x_domain_nm=x_domain_nm))
    stack.add_layer(
        Layer.piecewise(
            thickness_nm=100.0,
            x_domain_nm=x_domain_nm,
            segments=[
                (0.0, 100.0, eps_si_405 * jnp.eye(3)),
                (100.0, 500.0, eps_air * jnp.eye(3)),
            ],
        )
    )

    return {
        "period_nm": period_nm,
        "x_domain_nm": x_domain_nm,
        "eps_sio2_405": eps_sio2_405,
        "eps_nbocl2_405": eps_nbocl2_405,
        "eps_si_405": eps_si_405,
        "eps_air": eps_air,
        "stack": stack,
    }
