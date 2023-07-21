from typing import Any
from flax.struct import dataclass as pytree
import jax.numpy as jnp


@pytree
class GaussianProcess:
  """Gaussian process.

  Attributes:
    mean: mean of the Gaussian process.
    inv_covariance: inverse of the covariance matrix of the Gaussian process.
  """
  mean: Any
  inv_covariance: Any


class GPConstraints:
  """Constraints of Gaussian processes."""

  def find_max_mode():
    raise NotImplementedError


class NoConstraints(GPConstraints):
  """No constraints on the Gaussian process."""

  def find_max_mode(self, gp: GaussianProcess):
    return gp.mean
