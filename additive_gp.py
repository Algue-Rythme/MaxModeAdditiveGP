from dataclasses import dataclass as python_dataclass
from kernels import MultivariateKernel
from typing import List, Union

from flax.struct import dataclass as pytree
import jax.numpy as jnp
from jaxopt.tree_util import tree_add_scalar_mul, tree_sum, tree_map

from constraints import GPConstraints, NoConstraints, GaussianProcess
from function_basis import FunctionBasis
from variables import VariablePartition, VariableBlock, Variable1D


@pytree
class BlockInterpolator:
  """Function defined on a block of variables, used for interpolation.
  
  Attributes:
    K: kernel on spatial positions, of size (L_b, L_b)
    Phi: values of the hat functions at x_train, of size (n_points, L_b)
  """
  K: jnp.array
  Phi: jnp.array


@pytree
class PartitionInterpolator:
  """Function defined on blocks of variables, used for interpolation.

  Note: L = sum_b L_b is the total number of hat functions.
  
  Attributes:
    K: kernel on spatial positions, of size (L, L)
    Phi: values of the hat functions at x_train, of size (n_points, L)
  """
  K: List[jnp.array]
  Phi: List[jnp.array]


@pytree
class StructuredPrediction:
  """Prediction of an additive function over blocks.
  
  Attributes:
    y_pred: prediction of the function over the whole domain.
    y_pred_per_block: prediction of the function over each block independently. Useful for interpertation.
  """
  y_pred: jnp.array
  y_pred_per_block: List[jnp.array]


@pytree
class AdditiveFunction:
  """Additive function.

  Note: L = sum_b L_b is the total number of hat functions.

  Attributes:
    ksi: max mode of the GP, list of arrays of shape (L_b,)
    variable_partition: partition of the variables
    function_basis: basis of the function
  """
  ksi: List[jnp.array]
  variable_partition: VariablePartition
  function_basis: FunctionBasis

  def predict_block(self,
                    block_idx: int,
                    x_test: jnp.array) -> jnp.array:
    """Predict the function at x_test for a single variable block."""
    block = self.variable_partition.blocks[block_idx]
    ksi_block = self.ksi[block_idx]  # shape (L_b,)
    x_proj = block.project(x_test)  # shape (n_points, n_variables)
    Phi = self.function_basis.evaluate_nD(block, x_proj)  # shape (n_points, L_b)
    # Phi is of shape (n_points, L_b)
    # ksi is of shape (L_b,)
    y_pred = Phi @ ksi_block
    return y_pred  # y_pred is of shape (n_points,)

  def predict(self, x_test: jnp.array) -> jnp.array:
    """Predict the function at x_test.
    
    Args:
      ksi: coefficients of the additive GP of shape (L,)
      x_test: points at which to predict the GP of shape (n_points, n_variables)
    
    Returns:
      y_pred: values of the function at x_test of shape (n_points,)
    """
    y_pred_per_block = []
    for block_idx in range(len(self.variable_partition.blocks)):
      y_pred_block = self.predict_block(block_idx, x_test)
      y_pred_per_block.append(y_pred_block)
    y_pred = sum(y_pred_per_block)
    return StructuredPrediction(y_pred, y_pred_per_block)


@python_dataclass
class MaxModeAdditiveGaussianProcess:
  """Additive Gaussian process.
  
  Attributes:
    variable_partition: partition of the input variables.
    function_basis: basis of the functions.
    kernel: kernel of the GP.
    constraints: constraints on the GP (e.g none, monotonicity, convexity...).
    regul: float, regularization parameter (default 1e-3).
    debug: bool, whether to perform debug checks (default False).
  """
  variable_partition: VariablePartition
  function_basis: FunctionBasis
  kernel: List[MultivariateKernel]
  constraints: GPConstraints
  regul: Union[float, List[float]] = 1e-3
  debug: bool = False

  def fit_variable_block(self,
                         variable_block: VariableBlock,
                         x_train: jnp.array) -> BlockInterpolator:
    """Fit the GP on a single variable block.

    Note: L_b = prod_i m_i is the number of hat functions in the block.
    
    Args:
      variable_block: block of variables of shape (n_variables,)
      x_train: points at which to evaluate the GP of shape (n_points, n_variables)
    
    Returns:
      K: covariance matrix of the kernel of shape (L_b, L_b)
      Phi: values of the hat functions at x_train of shape (n_points, L_b)
    """
    x_proj = variable_block.project(x_train)
    # m is the number of hat functions in the block
    Phi = self.function_basis.evaluate_nD(variable_block, x_proj)  # Phi is of shape (n_points, L_b)
    multi_indices = variable_block.compute_subdivision_nd()  # shape (m_1 * ... * m_d, d)
    K = self.kernel(multi_indices)  # K is of shape (L_b, L_b)
    return BlockInterpolator(K, Phi)

  def fit_blocks(self, x_train) -> PartitionInterpolator:
    """Fit the GP on all variable blocks.
    
    Args:
      x_train: points at which to evaluate the GP of shape (n_points, n_variables)
      
    Returns:
      K: covariance matrix of the kernel of shape (L, L)
      Phi: values of the hat functions at x_train of shape (n_points, L)
    """
    K, Phi = [], []
    for block in self.variable_partition.blocks:
      block_interpolator = self.fit_variable_block(block, x_train)
      K.append(block_interpolator.K)
      Phi.append(block_interpolator.Phi)
    return PartitionInterpolator(K, Phi)

  def build_additive_gp(self, partition_interpolator, y_train):
    """Fit the GP on all variable blocks.
    
    Args:
      partition_interpolator: pair (K, Phi) of shape (L, L) and (n_points, L).
      y_train: values of the function at x_train of shape (n_points,)
    
    Returns:
      Gaussian process with mean of size (L,) and covariance matrix of size (L,L)
      In practice mean is a list of arrays of shape (L_b,) and covariance matrix is a list of arrays of shape (L_b, L_b)
      since it is block diagonal.
    """
    K, Phi = partition_interpolator.K, partition_interpolator.Phi

    def compute_B(K_b, Phi_b):
      return K_b @ Phi_b.T  # shape (L_b, n_points)
    B = tree_map(compute_B, K, Phi)  # list of arrays of shape (L_b, n_points)

    def compute_Phi_B(Phi_b, B_b):
      return Phi_b @ B_b
    Phi_B = tree_map(compute_Phi_B, Phi, B)  # list of arrays of shape (n_points, n_points)

    n_points = Phi_B[0].shape[0]
    A = sum(Phi_B) + self.regul * jnp.eye(n_points)  # shape (n_points, n_points)

    # TODO: A is PSD so we can use Cholesky decomposition (maybe faster?)
    inv_A = jnp.linalg.inv(A)  # shape (n_points, n_points)
    B_inv_A = tree_map(lambda B_b: B_b @ inv_A, B)  # list of arrays of shape (L_b, n_points)

    mean = tree_map(lambda B_b: B_b @ y_train, B_inv_A)  # list of arrays of shape (L_b,)

    # TODO: K is PSD so we can use Cholesky decomposition (maybe faster?)
    K_inv = tree_map(jnp.linalg.inv, K)  # list of arrays of shape (L_b, L_b)
    inv_regul = 1 / self.regul
    Gram_Phi = tree_map(lambda Phi_b: Phi_b.T @ Phi_b, Phi)  # list of arrays of shape (L_b, L_b)
    inv_covariance = tree_add_scalar_mul(K_inv, inv_regul, Gram_Phi)  # equation (30)

    if self.debug:
      covariance = K - B_inv_A @ B.T  # work only for a single block.
      assert jnp.allclose(covariance @ inv_covariance, jnp.eye(covariance.shape[0]))

    return GaussianProcess(mean, inv_covariance)

  def fit(self, x_train, y_train):
    """Fit the additive GP.
    
    Args:
      x_train: points at which to fit the GP of shape (n_points, n_variables)
      y_train: values of the function at x_train of shape (n_points,)
    
    Returns:
      ksi: coefficients of the additive GP, list of arrays of shape (L_b,)
    """
    partition_interpolator = self.fit_blocks(x_train)
    gp = self.build_additive_gp(partition_interpolator, y_train)
    ksi = self.constraints.find_max_mode(gp)
    # ksi is of shape (m,)
    return AdditiveFunction(ksi, self.variable_partition, self.function_basis)
