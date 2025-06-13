import jax.numpy as jnp
from jaxtyping import Array


def normalize_0_to_1(x: Array) -> Array:
    min_val = jnp.min(x)
    max_val = jnp.max(x)
    return (x - min_val) / (max_val - min_val)
