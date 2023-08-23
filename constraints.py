from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Callable
from dataclasses import dataclass as python_dataclass

import jax.numpy as jnp
from flax.struct import dataclass as pytree
from variables import VariablePartition, VariableBlock


@pytree
class FiniteDimensionalGP:
  """Finite dimensional Gaussian process evaluated at a set of points.

  Attributes:
    mean: mean of the Gaussian process of shape (L,).
    inv_covariance: inverse of the covariance matrix of the Gaussian process, of shape (L, L).
  """
  mean: jnp.array
  inv_covariance: jnp.array


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
  solver_kwargs: Optional[dict] = None

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


class ConstraintsSolver:
  """Solver for the constraints of the constrained Gaussian process.
  
  Solve the following optimization problem:

    .. math::
      \\max_{x} \\frac{1}{2} (x^T - \\mu) K^{-1} (x - \\mu)
        \\text{ s.t. } x \\in \\mathcal{C}

    Note that we can expand the objective function as:

    .. math::
      \\frac{1}{2} (x^T - \\mu) K^{-1} (x - \\mu) = \\frac{1}{2} x^T K^{-1} x - \\mu^T K^{-1} x + \\frac{1}{2} \\mu^T K^{-1} \\mu

    where we recognize the quadratic term as K^{-1},
    the affine term as -\\mu^T K^{-1} x and the constant term as \\frac{1}{2} \\mu^T K^{-1} \\mu.

    Observe that since K^{-1} is PSD, the objective function is convex, while the constraints $\\mathcal{C}$ are linear.
    Therefore, this is a quadratic program, which can be solved efficiently with OSQP routines from jaxopt.
  """
  
  def __init__(self, **kwargs):
    """Initialize the solver.
    
    Args:
      solver_kwargs: additional arguments to pass to the solver.
    """
    self.solver_kwargs = kwargs

  @staticmethod
  def _objective_fun(alpha: jnp.array, params_obj: FiniteDimensionalGP) -> jnp.array:
    """Objective function of the optimization problem.
    
    Args:
      alpha: vector of shape (L,) to optimize.
      params_obj: object of type FiniteDimensionalGP.
    """
    x = alpha - params_obj.mean
    linear_form = jnp.dot(params_obj.inv_covariance, x)
    scalar = jnp.vdot(x, linear_form)
    # we drop the 0.5 factor since it does not change the argmin.  
    return scalar

  @staticmethod
  def _build_bounds(partition: VariablePartition,
                    constraints: List[BlockConstraints]) -> Tuple[List[Any], List[Any]]:
    upper_bounds = []
    lower_bounds = []
    for block, block_cons in zip(partition, constraints):
      if block_cons.is_empty():
        continue
      l, u = block_cons.get_bounds(block)
      lower_bounds.append(l)
      upper_bounds.append(u)
    return lower_bounds, upper_bounds

  @staticmethod
  def _build_matvec(partition: VariablePartition,
                    constraints: List[BlockConstraints]) -> Callable:
    
    def Ax(params_A: Any, alphas: jnp.array) -> List[Any]:
      del params_A
      alphas = partition.split_and_reshape(alphas)  # list of arrays of shape (m_1, ..., m_d) each.
      Ax = []
      for block, block_cons, alpha in zip(partition, constraints, alphas):
        if block_cons.is_empty():
          continue
        matvec = block_cons.get_matvec(block)
        block_Ax = matvec(None, alpha)
        Ax.append(block_Ax)
      return Ax
    
    return Ax

  @staticmethod
  def no_constraints(constraints: List[BlockConstraints]):
    """Whether there are no constraints."""
    return all(constraint.is_empty() for constraint in constraints)

  @staticmethod
  def _canonicalize_constraints(partition: VariablePartition,
                                constraints: List[Optional[BlockConstraints]]):
    """Canonicalize the constraints to a list of BlockConstraints."""
    if constraints is None:
      constraints = [None] * len(partition)  # no constraints.
    elif isinstance(constraints, BlockConstraints):
      constraints = [constraints] * len(partition)  # broadcast the same constraint to all blocks.
    elif len(constraints) != len(partition):
      raise ValueError(f"Expected {len(partition)} constraints, got {len(constraints)}.")
    
    canonicalized = []
    for block_cons in constraints:
      if block_cons is None:
        canonicalized.append(NoConstraints())
      elif isinstance(block_cons, BlockConstraints):
        canonicalized.append(block_cons)
      else:
        raise ValueError(f"Unknown constraint {block_cons}.")

    return canonicalized

  def find_maximum_a_posteriori(self,
                                gp: FiniteDimensionalGP,
                                partition: VariablePartition,
                                constraints: List[Optional[BlockConstraints]]) -> jnp.array:
    """Find the maximum mode of the Gaussian process under the constraints."""
    try:
      from jaxopt import BoxOSQP
    except ImportError:
      raise ImportError("Please install jaxopt to use constraints.")

    constraints = ConstraintsSolver._canonicalize_constraints(partition, constraints)

    if ConstraintsSolver.no_constraints(constraints):
      return gp.mean

    objective_fun = ConstraintsSolver._objective_fun
    matvec_A = ConstraintsSolver._build_matvec(partition, constraints)
    params_ineq = ConstraintsSolver._build_bounds(partition, constraints)
    
    solver = BoxOSQP(fun=objective_fun, matvec_A=matvec_A, **self.solver_kwargs)

    # warm start the solver with the mean of the Gaussian process to speed up convergence.
    alpha0 = gp.mean
    hyper_params = dict(params_obj=gp, params_eq=None, params_ineq=params_ineq)
    kkt_sol = solver.init_params(init_x=alpha0, **hyper_params)  # initialize the solver with the mean of the Gaussian process.
    kkt_sol, state = solver.run(init_params=kkt_sol, **hyper_params)

    del state  # unused

    alpha_optimized = kkt_sol.primal[0]

    return alpha_optimized
