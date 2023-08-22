from typing import Any, Optional
from dataclasses import dataclass as python_dataclass

import numpy as onp
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from flax.struct import dataclass as pytree
from jaxopt.tree_util import tree_dot, tree_zeros_like
from utils import tree_matvec

from variables import VariablePartition


@pytree
class FiniteDimensionalGP:
  """Finite dimensional Gaussian process evaluated at a set of points.

  Attributes:
    mean: mean of the Gaussian process of shape (L,).
    inv_covariance: inverse of the covariance matrix of the Gaussian process, of shape (L, L).
  """
  mean: jnp.array
  inv_covariance: jnp.array


class GPConstraints:
  """Constraints of Gaussian processes.
  
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

  @staticmethod
  def objective_fun(alpha, params_obj):
    """Objective function of the optimization problem.
    
    Args:
      alpha: vector of shape (L,) to optimize.
      params_obj: object of type FiniteDimensionalGP.
    """
    x = alpha - params_obj.mean
    linear_form = params_obj.inv_covariance @ x
    scalar = x @ linear_form
    # we drop the 0.5 factor since it does not change the argmin.  
    return scalar

  @staticmethod
  def find_max_mode(self, 
                    gp: FiniteDimensionalGP,
                    partition: VariablePartition):
    """Find the maximum mode of the Gaussian process under the constraints.

    Args:
      gp: Gaussian process.
      partition: partition of the variables.
    """
    pass


class NoConstraints(GPConstraints):
  """No constraints on the Gaussian process."""

  def find_max_mode(self,
                    gp: FiniteDimensionalGP,
                    partition: VariablePartition):
    """Find the maximum mode of an unconstrained Gaussian process.
    
    Expectedly, the maximum mode is the mean of the Gaussian process.
    """
    del partition  # unused
    return gp.mean


@python_dataclass
class BoundedConstraints(GPConstraints):
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
  solver_kwargs: Optional[dict] = None

  def find_max_mode(self,
                    gp: FiniteDimensionalGP,
                    partition: VariablePartition):
    """Find the maximum mode of a bounded Gaussian process."""
    del partition  # unused

    try:
      from jaxopt import BoxOSQP
    except ImportError:
      raise ImportError("Please install jaxopt to use bounded constraints.")

    solver_kwargs = {} if self.solver_kwargs is None else self.solver_kwargs

    objective_fun = GPConstraints.objective_fun

    def matvec_A(params_A, alpha):
      del params_A
      return alpha

    params_ineq = (jnp.array(self.min_value), jnp.array(self.max_value))

    solver = BoxOSQP(fun=objective_fun, matvec_A=matvec_A, **solver_kwargs)
    # warm start the solver with the mean of the Gaussian process to speed up convergence.
    alpha0 = gp.mean
    kkt_sol = solver.init_params(alpha0)  # initialize the solver with the mean of the Gaussian process.
    kkt_sol, _ = solver.run(kkt_sol, params_obj=gp, params_eq=None, params_ineq=params_ineq)

    alpha_optimized = kkt_sol.primal[0]

    return alpha_optimized


@python_dataclass
class MonotoneConstraints(GPConstraints):
  """Bounded constraints on the Gaussian process.
  
  Attributes:
    monotonicity: whether the Gaussian process is `increasing` or `decreasing` (default: `increasing`).
    solver_kwargs: additional arguments to pass to the solver.
  """
  monotonicity: str = 'increasing'
  solver_kwargs: Optional[dict] = None

  def _get_diff_shapes(self, partition: VariablePartition):
    """Get the shapes of the differences of the variables in the partition."""
    shapes = []
    for block in partition:
      subdivision_shape = block.subdivision_shape
      for axis in range(subdivision_shape.ndim):
        shape = list(subdivision_shape)
        shape[axis] -= 1  # difference along axis.  
        shapes.append(shape)
    return shapes

  def find_max_mode(self,
                    gp: FiniteDimensionalGP,
                    partition: VariablePartition):
    """Find the maximum mode of a monotone Gaussian process."""
    try:
      from jaxopt import BoxOSQP
    except ImportError:
      raise ImportError("Please install jaxopt to use monotone constraints.")

    solver_kwargs = {} if self.solver_kwargs is None else self.solver_kwargs

    objective_fun = GPConstraints.objective_fun

    def matvec_A(params_A, alpha):
      """Compute the matrix-vector product Ax that appears in the constraints l <= Ax <= u."""
      del params_A
      # alpha is a list of arrays of shape (L_b,) each, with L = sum_b L_b
      alphas_per_block = partition.split_and_reshape(alpha)  # list of lists of arrays of shape (L_b,)
      Ax = []
      for alpha in alphas_per_block:
        for axis in range(alpha.ndim):
          monotonicity = jnp.diff(alpha, axis=axis)  # shape (m_1,..., m_i - 1,..., m_{L_b})
          Ax.append(monotonicity)
      return Ax  # Ax is a list arrays of shape (m_1, ..., m_i - 1, ...,  m_{L_b}) each.  

    shapes = self._get_diff_shapes(partition)
    
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

    solver = BoxOSQP(fun=objective_fun, matvec_A=matvec_A, **solver_kwargs)
    # warm start the solver with the mean of the Gaussian process to speed up convergence.
    alpha0 = gp.mean
    kkt_sol = solver.init_params(alpha0)  # initialize the solver with the mean of the Gaussian process.
    kkt_sol, _ = solver.run(kkt_sol, params_obj=gp, params_eq=None, params_ineq=params_ineq)

    alpha_optimized = kkt_sol.primal[0]

    return alpha_optimized
