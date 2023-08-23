from typing import List
from abc import abstractmethod
from flax.struct import dataclass as pytree
import jax.numpy as jnp


class UnivariateKernel:
  """Abstract class for univariate kernels."""

  @abstractmethod
  def __call__(self, x):
    """Compute the covariance matrix of the kernel.
    
    Args:
      x: points at which to evaluate the kernel of shape (n_points,)
    
    Returns:
      covariance matrix of the kernel of shape (n_points, n_points)
    """
    raise NotImplementedError


class MultivariateKernel:
  """Abstract class for univariate kernels."""

  @abstractmethod
  def __call__(self, x):
    """Compute the covariance matrix of the kernel.
    
    Args:
      x: points at which to evaluate the kernel of shape (n_points,)
    
    Returns:
      covariance matrix of the kernel of shape (n_points, n_points)
    """
    raise NotImplementedError


@pytree
class TensorProductKernel(MultivariateKernel):
  """Abstract class for multivariate kernels."""
  kernels: List[UnivariateKernel]

  def __call__(self, x):
    """Default implementation of the multivariate kernel.
    
    Args:
      x: points at which to evaluate the kernel of shape (n_points, n_variables)
    
    Returns:
      covariance matrix of the kernel of shape (n_points, n_points)
    """
    return jnp.prod([kernel(x[:, i]) for i, kernel in enumerate(self.kernels)], axis=0)


@pytree
class GaussianKernel(MultivariateKernel):
  """Gaussian kernel.
  
  Attributes:
    length_scale: length scale of the kernel.
    
  Remark: the kernel is defined as exp(-0.5 * ||x - y||^2 / length_scale^2)

  length_scale = length_scale_multiplier * gamma
  """
  variance: float = 1.
  length_scale: float = 1.

  def __call__(self, x):
    """Compute the covariance matrix of the kernel.
    
    Args:
      x: points at which to evaluate the kernel, of shape (n_points, n_variables)
    
    Returns:
      covariance matrix of the kernel of shape (n_points, n_points)
    """
    default_bandwidth = jnp.sqrt(x.shape[-1] * jnp.var(x))  # based on scikit's RBF kernel
    bandwidth = self.length_scale * default_bandwidth
    x = x / bandwidth
    hilbert_norm = jnp.sum(jnp.square(x[:, None, :] - x[None, :, :]), axis=-1)
    return self.variance * jnp.exp(-0.5 * hilbert_norm)
