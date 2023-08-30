from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Callable
from dataclasses import dataclass as python_dataclass

import numpy as onp
import jax.numpy as jnp
from variables import VariableBlock


class BlockConstraints(ABC):
  """Constraints of Gaussian processes."""

  def is_empty(self):
    """Whether the constraints are empty."""
    return False

  @abstractmethod
  def get_matvec(self):
    """Get the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
    raise NotImplementedError

  @abstractmethod
  def get_bounds(self):
    """Get the bounds l and u that appear in the constraints l <= Ax <= u."""
    raise NotImplementedError


class NoConstraints(BlockConstraints):
  """No constraints on the Gaussian process."""

  def is_empty(self):
    return True

  def get_matvec(self, block: VariableBlock) -> Callable:
    """Get the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
    raise NotImplementedError

  def get_bounds(self, block: VariableBlock):
    """Get the bounds l and u that appear in the constraints l <= Ax <= u."""
    raise NotImplementedError


@python_dataclass
class BoundedConstraints(BlockConstraints):
  """Bounded constraints on the Gaussian process.
  
  .. math::
      \\mathcal{C} = [\\text{min_value}, \\text{max_value}]^L

  Attributes:
    min_value: lower bound of the Gaussian process.
    max_value: upper bound of the Gaussian process.
    solver_kwargs: additional arguments to pass to the solver.
  """
  min_value: float = float('-inf')
  max_value: float = float('inf')

  def get_matvec(self, block: VariableBlock) -> Callable:
    """Get the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
    
    def matvec_A(params_A: Any, alpha: jnp.array) -> jnp.array:
      del params_A
      return alpha
    
    return matvec_A

  def get_bounds(self, block: VariableBlock) -> Tuple[jnp.array, jnp.array]:
    """Get the bounds l and u that appear in the constraints l <= Ax <= u."""
    subdivision_shape = block.subdivision_shape
    l = jnp.full(subdivision_shape, self.min_value)  # l = min_value * ones of shape (m_1, ..., m_d)
    u = jnp.full(subdivision_shape, self.max_value)  # u = max_value * ones of shape (m_1, ..., m_d)
    return (l ,u)   


@python_dataclass
class MonotoneConstraints(BlockConstraints):
  """Bounded constraints on the Gaussian process.
  
  Attributes:
    monotonicity: whether the Gaussian process is `increasing` or `decreasing` (default: `increasing`).
    solver_kwargs: additional arguments to pass to the solver.
  """
  monotonicity: str = 'increasing'

  def _get_diff_shapes(self, block: VariableBlock):
    """Get the shapes of the differences of the variables in the partition."""
    shapes = []
    subdivision_shape = block.subdivision_shape
    ndim = len(subdivision_shape)
    for axis in range(ndim):
      shape = list(subdivision_shape)
      shape[axis] -= 1  # difference along axis.  
      shapes.append(shape)
    return shapes

  def get_matvec(self, block: VariableBlock) -> Callable:
    del block  # unused

    def matvec_A(params_A: Any, alpha: jnp.array) -> List[jnp.array]:
      """Compute the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
      del params_A
      Ax = []
      for axis in range(alpha.ndim):
        monotonicity = jnp.diff(alpha, axis=axis)  # shape (m_1,..., m_i - 1,..., m_d)
        Ax.append(monotonicity)
      return Ax  # Ax is a list arrays of shape (m_1, ..., m_i - 1, ...,  m_d) each.  

    return matvec_A

  def get_bounds(self, block: VariableBlock) -> Tuple[List[jnp.array], List[jnp.array]]:
    """Get the bounds l and u that appear in the constraints l <= Ax <= u."""
    shapes = self._get_diff_shapes(block)
    
    if self.monotonicity == 'increasing':
      # 0 <= ksi[block][var][i+1] - ksi[block][var][i] <= +infty
      l = [jnp.zeros(shape) for shape in shapes]
      u = [jnp.full(shape, float('inf')) for shape in shapes]
    elif self.monotonicity == 'decreasing':
      # -infty <= ksi[block][var][i+1] - ksi[block][var][i] <= 0
      l = [jnp.full(shape, float('-inf')) for shape in shapes]
      u = [jnp.zeros(shape) for shape in shapes]
    else:
      raise ValueError(f"Unknown monotonicity {self.monotonicity}.")
    
    params_ineq = (l, u)
    return params_ineq


@python_dataclass
class CurvatureConstraints(BlockConstraints):
  """Bounded constraints on the Gaussian process.

  The concavity/convexity property is denoted `curvature` since for smooth functions (like GP)
  the concavity/convexity property is a statement on the spectrum of the Hessian matrix.
  
  Attributes:
    curvature: whether the Gaussian process is `convex` or `concave` (default: `convex`).
    solver_kwargs: additional arguments to pass to the solver.
  """
  curvature: str = 'convex'

  def _get_diff_shapes(self, block: VariableBlock):
    """Get the shapes of the differences of the variables in the partition."""
    shapes = []
    subdivision_shape = block.subdivision_shape
    ndim = len(subdivision_shape)
    for axis in range(ndim):
      shape = list(subdivision_shape)
      shape[axis] -= 2  # difference along axis.  
      shapes.append(shape)
    return shapes

  def get_matvec(self, block: VariableBlock) -> Callable:

    def matvec_A(params_A: Any, alpha: jnp.array) -> List[jnp.array]:
      """Compute the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
      del params_A
      Ax = []
      for axis in range(alpha.ndim):
        subdivision = block[axis].subdivision_without_sentinels

        a_delta = jnp.diff(alpha, n=1, axis=axis)  # shape (m_1,..., m_i - 1,..., m_d)
        t_delta = jnp.diff(subdivision, n=1, axis=axis)  # shape (m_i - 1,)

        # Numpy arrays are compile-time constant so they can be used in indexing.
        all_but_end = onp.arange(a_delta.shape[axis] - 1)  # indices of all but the last element along axis.
        all_but_first = onp.arange(1, a_delta.shape[axis])  # indices of all but the first element along axis.

        a_delta_no_end = jnp.take(a_delta, indices=all_but_end, axis=axis)  # shape (m_1,..., m_i - 2,..., m_d)
        a_delta_no_first = jnp.take(a_delta, indices=all_but_first, axis=axis)  # shape (m_1,..., m_i - 2,..., m_d)

        t_delta_no_end = jnp.take(t_delta, indices=all_but_end, axis=axis)  # shape (m_i - 2,)
        t_delta_no_first = jnp.take(t_delta, indices=all_but_first, axis=axis)  # shape (m_i - 2,)

        # broadcast t_delta_* to shape (m_1,..., m_i - 2,..., m_d)
        axis_size = len(subdivision) - 2
        broadcast_shape = [1] * axis + [axis_size] + [1] * (alpha.ndim - axis - 1)
        t_delta_no_end = jnp.reshape(t_delta_no_end, broadcast_shape)
        t_delta_no_first = jnp.reshape(t_delta_no_first, broadcast_shape)

        factor_1 = a_delta_no_first * t_delta_no_end  # shape (m_1,..., m_i - 2,..., m_d)
        factor_2 = a_delta_no_end * t_delta_no_first  # shape (m_1,..., m_i - 2,..., m_d)

        curvature = factor_1 - factor_2  # shape (m_1,..., m_i - 2,..., m_d)

        Ax.append(curvature)
      return Ax  # Ax is a list arrays of shape (m_1, ..., m_i - 2, ...,  m_d) each.  

    return matvec_A

  def get_bounds(self, block: VariableBlock) -> Tuple[List[jnp.array], List[jnp.array]]:
    """Get the bounds l and u that appear in the constraints l <= Ax <= u."""
    shapes = self._get_diff_shapes(block)
    
    if self.curvature == 'convex':
      # 0 <= ksi[block][var][i+1] - ksi[block][var][i] <= +infty
      l = [jnp.zeros(shape) for shape in shapes]
      u = [jnp.full(shape, float('inf')) for shape in shapes]
    elif self.curvature == 'concave':
      # -infty <= ksi[block][var][i+1] - ksi[block][var][i] <= 0
      l = [jnp.full(shape, float('-inf')) for shape in shapes]
      u = [jnp.zeros(shape) for shape in shapes]
    else:
      raise ValueError(f"Unknown curvature {self.monotonicity}.")
    
    params_ineq = (l, u)
    return params_ineq
