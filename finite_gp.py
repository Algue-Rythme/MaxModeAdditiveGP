from abc import ABC, abstractmethod
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as onp
from jax.tree_util import tree_map
from flax.struct import dataclass as pytree


@pytree
class BlockInterpolator:
  """Function defined on a block of variables, used for interpolation.
  
  Attributes:
    K: kernel on spatial positions, of size (L_b, L_b)
    Phi: values of the hat functions at x_train, of size (L_b, n_points)
  """
  K: jnp.array
  Phi: jnp.array


@pytree
class PartitionInterpolator:
  """Function defined on disjoints blocks of variables, used for interpolation.

  Note: L = sum_b L_b is the total number of hat functions.
  
  Attributes:
    K: kernel on spatial positions, block diagonal with blocks of size (L_b, L_b)
    Phi: values of the hat functions at x_train, list of arrays of shape (L_b, n_points)
  """
  K: List[jnp.array]
  Phi: List[jnp.array]


class FiniteDimensionalGP(ABC):
  """Finite dimensional Gaussian process evaluated at a set of points.

  Based on Woodbury identity: https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Cov^{-1} = K^{-1} + (1 / lbda) \Phi_i^T \Phi_i

  Attributes:
    mean: mean of the Gaussian process of shape (L,).
  """
  @staticmethod
  @abstractmethod
  def from_partition_interpolator(partition_interpolator: PartitionInterpolator,
                                  y_train: jnp.array,
                                  regul: float):
    """Create a finite dimensional Gaussian process from the kernel, the interpolant Phi and the inverse regularization parameter."""
    raise NotImplementedError
  
  @staticmethod
  def _compute_mean(partition_interpolator: PartitionInterpolator,
                    y_train: jnp.array,
                    regul: float) -> List[jnp.array]:
    """Compute the mean of the additive GP.
    
    Args:
      K: covariance matrix of the kernel, list of arrays of shape (L_b, L_b)
      Phi: values of the hat functions at x_train, list of arrays of shape (L_b, n_points)
      y_train: values of the function at x_train of shape (n_points,)
      covariance: whether to return the full covariance matrix (default False)

    Returns:
      mean: mean of the additive GP, list of arrays of shape (L_b,)
    """
    K, Phi = partition_interpolator.K, partition_interpolator.Phi

    K_Phi = tree_map(jnp.matmul, K, Phi)  # list of arrays of shape (L_b, n_points)

    K_Phi = jnp.concatenate(K_Phi, axis=0)  # shape (L, n_points)
    Phi = jnp.concatenate(Phi, axis=0)  # shape (L, n_points)

    Phi_K_Phi = Phi.T @ K_Phi  # array of shape (n_points, n_points)
    n_points = Phi_K_Phi.shape[0]
    A = Phi_K_Phi + regul * jnp.eye(n_points)  # shape (n_points, n_points)

    cho_factor = jax.scipy.linalg.cho_factor(A)
    inv_A_y_train = jax.scipy.linalg.cho_solve(cho_factor, y_train)  # array of shape (n_points,)

    # mean = K @ Phi @ ( Phi.T @ K @ Phi + regul * I )^-1 @ y_train
    mean = K_Phi @ inv_A_y_train  # arrays of shape (L,)
    
    return mean

  @abstractmethod
  def precision_matvec(self, x: jnp.array) -> jnp.array:
    """Compute the matrix-vector product Cov^{-1} x."""
    raise NotImplementedError


@pytree
class DenseFiniteDimensionalGP(FiniteDimensionalGP):
  """Finite dimensional Gaussian process evaluated at a set of points. Dense representation.

  Attributes:
    mean: mean of the Gaussian process of shape (L,).
    precision: inverse of the covariance matrix of the Gaussian process, of shape (L, L).
  """
  mean: jnp.array
  precision: jnp.array

  @staticmethod
  def from_partition_interpolator(partition_interpolator: PartitionInterpolator,
                                  y_train: jnp.array,
                                  regul: float) -> FiniteDimensionalGP:
    """Create a finite dimensional Gaussian process from the kernel, the interpolant Phi and the inverse regularization parameter."""
    mean = FiniteDimensionalGP._compute_mean(partition_interpolator, y_train, regul)
    K, Phi = partition_interpolator.K, partition_interpolator.Phi
    # computing the inverse of the covariance matrix can be done more efficiently
    # than inverting the covariance matrix, since K is block diagonal.
    K_inv = tree_map(jnp.linalg.inv, K)  # list of arrays of shape (L_b, L_b)
    # from list of blocks to single big matrix with blocks on diagonal
    K_inv = jax.scipy.linalg.block_diag(*K_inv)  # shape (L, L)
    Phi = jnp.concatenate(Phi, axis=0)  # shape (L, n_points)
    GramPhi = Phi @ Phi.T  # matrix of shape (L, L)
    precision = K_inv + (1. / regul) * GramPhi  # equation (30)
    return DenseFiniteDimensionalGP(mean, precision)

  def precision_matvec(self, x: jnp.array) -> jnp.array:
    """Compute the matrix-vector product Cov^{-1} x."""
    return jnp.dot(self.precision, x)


def _split_as(x: jnp.array, y: List[jnp.array]):
  """Split x into a list of arrays of the same shape as y."""
  indices = onp.cumsum([len(y_i) for y_i in y[:-1]])  # onp.array = compile-time constant
  return jnp.split(x, indices)


@pytree
class SparseFiniteDimensionalGP(FiniteDimensionalGP):
  """Finite dimensional Gaussian process evaluated at a set of points. Sparse representation.

  Attributes:
    mean: mean of the Gaussian process of shape (L,).
    K_cholesky_factors: list of tuples of arrays of shape (L_b, L_b),
                        factors of the Cholesky decomposition of the block-diagonal kernel.
    Phi: values of the hat functions at x_train, list of arrays of shape (L_b, n_points)
    inv_regul: inverse of the regularization parameter.
  """
  mean: jnp.array
  K_cholesky_factors: List[Tuple[jnp.array]]
  Phi: List[jnp.array]
  inv_regul: float

  @staticmethod
  def from_partition_interpolator(partition_interpolator: PartitionInterpolator,
                                  y_train: jnp.array,
                                  regul: float) -> FiniteDimensionalGP:
    """Create a finite dimensional Gaussian process from the kernel, the interpolant Phi and the inverse regularization parameter."""
    mean = FiniteDimensionalGP._compute_mean(partition_interpolator, y_train, regul)
    K, Phi = partition_interpolator.K, partition_interpolator.Phi
    def cho_factor(K_block):
      fac, _ = jax.scipy.linalg.cho_factor(K_block, lower=False)
      return fac
    K_cholesky_factors = tree_map(cho_factor, K)  # list of arrays of shape (L_b, L_b)
    return SparseFiniteDimensionalGP(mean, K_cholesky_factors, Phi, 1. / regul)

  def precision_matvec(self, x: jnp.array) -> jnp.array:
    """Compute the matrix-vector product Cov^{-1} x."""
    # split x so that it has the same shape as Phi
    splitted_x = _split_as(x, self.Phi)

    # list of arrays of shape (L_b,)
    def cho_solve(cho_factor, rhs):
      # boolean is a compile-time constant: not jittable
      return jax.scipy.linalg.cho_solve((cho_factor, False), rhs)
    
    inv_K_x = tree_map(cho_solve, self.K_cholesky_factors, splitted_x)
    # shape array of (L,)
    inv_K_x = jnp.concatenate(inv_K_x, axis=0)
    # shape array of (L,)
    Phi_dense = jnp.concatenate(self.Phi, axis=0)  # shape (L, n_points)
    Gram_Phi_x = jnp.dot(Phi_dense, jnp.dot(Phi_dense.T, x))

    return inv_K_x + self.inv_regul * Gram_Phi_x