import jax

# Match the numerical behavior used by the original codebase.
jax.config.update("jax_enable_x64", True)
