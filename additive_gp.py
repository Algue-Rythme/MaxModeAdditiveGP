from dataclasses import dataclass as python_dataclass
from kernels import MultivariateKernel
from typing import List, Union, Any, Optional

import jax
from flax.struct import dataclass as pytree
import jax.numpy as jnp
from jaxopt.tree_util import tree_map

from constraints import FiniteDimensionalGP, BlockConstraints
from function_basis import FunctionBasis
from solver import ConstraintsSolver
from variables import VariablePartition, VariableBlock


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
    ksi: max mode of the GP, list of arrays of shape (L_b,) each
    variable_partition: partition of the variables
    function_basis: basis of the functions over the blocks
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
    x_proj = block.project(x_test)  # shape (n_points, n_variables_in_block)
    Phi = self.function_basis.evaluate_nD(block, x_proj)  # shape (L_b, n_points)
    y_pred = ksi_block @ Phi
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
class ConstrainedAdditiveGP:
  """Additive Gaussian process.
  
  Attributes:
    partition: partition of the input variables.
    function_basis: basis of the functions.
    kernel: kernel of the GP.
    regul: positive float, regularization parameter (default 1e-3). Square of Tau parameter in the paper.
    constraints: list of constraints on the GP (e.g none, monotonicity, convexity...).
                  Length must match the number of blocks in the partition.  
                  Use None in the entry for no constraints.
    solver: quadratic solver for the constraints (default ConstraintsSolver).
    debug: bool, whether to perform debug checks (default False).
  """
  partition: VariablePartition
  function_basis: FunctionBasis
  kernel: List[MultivariateKernel]
  constraints: List[Optional[BlockConstraints]]
  solver: ConstraintsSolver = ConstraintsSolver()
  regul: Union[float, List[float]] = 1e-3
  verbose: bool = False

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
    Phi = self.function_basis.evaluate_nD(variable_block, x_proj)  # Phi is of shape (L_b, n_points)
    multi_indices = variable_block.compute_subdivision_nd()  # shape (L_b, d)
    K = self.kernel(multi_indices)  # K is of shape (L_b, L_b)
    return BlockInterpolator(K, Phi)

  def fit_blocks(self, x_train: jnp.array) -> PartitionInterpolator:
    """Fit the GP on all variable blocks.
    
    Args:
      x_train: points at which to evaluate the GP of shape (n_points, n_variables)
      
    Returns:
      K: covariance matrix of the kernel of shape (L, L)
      Phi: values of the hat functions at x_train of shape (n_points, L)
    """
    K, Phi = [], []
    for block in self.partition.blocks:
      block_interpolator = self.fit_variable_block(block, x_train)
      K.append(block_interpolator.K)
      Phi.append(block_interpolator.Phi)
    return PartitionInterpolator(K, Phi)

  def _compute_mean(self,
                    K: jnp.array,
                    Phi: Any,
                    y_train: jnp.array,
                    compute_covariance: bool = False) -> List[jnp.array]:
    """Compute the mean of the additive GP.
    
    Args:
      K: covariance matrix of the kernel, list of arrays of shape (L_b, L_b)
      Phi: values of the hat functions at x_train, list of arrays of shape (L_b, n_points)
      y_train: values of the function at x_train of shape (n_points,)
      covariance: whether to return the full covariance matrix (default False)

    Returns:
      mean: mean of the additive GP, list of arrays of shape (L_b,)
    """
    K_Phi = tree_map(jnp.matmul, K, Phi)  # list of arrays of shape (L_b, n_points)

    K_Phi = jnp.concatenate(K_Phi, axis=0)  # shape (L, n_points)
    Phi = jnp.concatenate(Phi, axis=0)  # shape (L, n_points)

    Phi_K_Phi = Phi.T @ K_Phi  # array of shape (n_points, n_points)
    n_points = Phi_K_Phi.shape[0]
    A = Phi_K_Phi + self.regul * jnp.eye(n_points)  # shape (n_points, n_points)

    # TODO: A is PD so we can use Cholesky decomposition (maybe faster?)
    inv_A = jnp.linalg.inv(A)  # shape (n_points, n_points)

    # mean = K @ Phi @ ( Phi.T @ K @ Phi + regul * I )^-1 @ y_train
    mean = K_Phi @ inv_A @ y_train  # arrays of shape (L,)

    covariance = None
    if compute_covariance:
      covariance = jax.scipy.linalg.block_diag(*K) - K_Phi @ inv_A @ K_Phi.T  # Covariance in equation (29)
    
    return mean, covariance

  def _compute_inv_covariance(self,
                              K: List[jnp.array],
                              Phi: List[jnp.array]) -> jnp.array:
    """Compute the inverse of the covariance matrix of the additive GP.
    
    Based on Woodbury identity: https://en.wikipedia.org/wiki/Woodbury_matrix_identity

    Args:
      K: covariance matrix of the kernel, list of arrays of shape (L_b, L_b)
      Phi: values of the hat functions at x_train, list of arrays of shape (L_b, n_points)
    """
    # TODO: K is PD so we can use Cholesky decomposition (maybe faster?)
    K_inv = tree_map(jnp.linalg.inv, K)  # list of arrays of shape (L_b, L_b)

    # from list of blocks to single big matrix with blocks on diagonal
    K_inv = jax.scipy.linalg.block_diag(*K_inv)  # shape (L, L)

    Phi = jnp.concatenate(Phi, axis=0)  # shape (L, n_points)
    GramPhi = Phi @ Phi.T  # matrix of shape (L, L)

    if self.verbose:
      print("Kernel", K)
      print("Inv Kernel", K_inv)
      print("Gram Phi", GramPhi)

    inv_covariance = K_inv + ( 1 / self.regul) * GramPhi  # equation (30)

    return inv_covariance

  def build_additive_gp(self,
                        partition_interpolator: PartitionInterpolator,
                        y_train: jnp.array) -> FiniteDimensionalGP:
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

    mean, covariance = self._compute_mean(K, Phi, y_train, compute_covariance=self.verbose)

    # computing the inverse of the covariance matrix can be done more efficiently
    # than inverting the covariance matrix, since K is block diagonal.
    inv_covariance = self._compute_inv_covariance(K, Phi)

    if self.verbose:
      I_approx = covariance @ inv_covariance
      I = jnp.eye(covariance.shape[0])
      residuals = I - I_approx
      error = jnp.linalg.norm(residuals)
      print(f"Error ||Cov Conv^{-1} - I|| = {error}")
      print("Cov eigenvalues", jnp.linalg.eigvals(covariance)) 
      print("Cov^{-1} eigenvalues", jnp.linalg.eigvals(inv_covariance))
      print("inv(Cov) eigenvalues", jnp.linalg.eigvals(jnp.linalg.inv(covariance)))
      print("inv(Cov^{-1}) eigenvalues", jnp.linalg.eigvals(jnp.linalg.inv(inv_covariance)))

    return FiniteDimensionalGP(mean, inv_covariance)

  def fit(self, 
          x_train: jnp.array,
          y_train: jnp.array) -> AdditiveFunction:
    """Fit the additive GP.
    
    Args:
      x_train: points at which to fit the GP of shape (n_points, n_variables)
      y_train: values of the function at x_train of shape (n_points,)
    
    Returns:
      additive function over disjoint blocks.
    """
    # evaluate the kernel on knots coordinates, evaluate the hat functions on x_train.
    partition_interpolator = self.fit_blocks(x_train)
    # estimate the mean and the covariance matrix of the additive GP (with constraints).
    gp = self.build_additive_gp(partition_interpolator, y_train)
    # Find the maximum mode of the GP under the constraints.
    ksi = self.solver.find_maximum_a_posteriori(gp, self.partition, self.constraints)
    # change ksi from dense vector of shape (L,) to list of arrays of shape (L_b,)
    ksi = self.partition.split(ksi)  # list of arrays of shape (L_b,)
    return AdditiveFunction(ksi, self.partition, self.function_basis)
