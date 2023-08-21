from abc import abstractmethod
import jax
import jax.numpy as jnp
from flax.struct import dataclass as pytree

from variables import VariableBlock, Variable1D


class FunctionBasis:
  """Abstract class for function bases."""

  @abstractmethod
  def evaluate_1D(self, variable_1d: Variable1D, x: jnp.array) -> jnp.array:
    raise NotImplementedError

  @abstractmethod
  def evaluate_nD(self, variable_block: VariableBlock, x: jnp.array) -> jnp.array:
    raise NotImplementedError


def unstack(x, axis=0):
  return [jax.lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


@pytree
class HatFunctions(FunctionBasis):
  max_value: float = 1.

  def evaluate_single_hat(self,
                          left: jnp.array,
                          center: jnp.array,
                          right: jnp.array,
                          x: jnp.array) -> jnp.array:
    """Evaluate a single hat function over a dataset.
    
    Args:
      left: left boundary of the hat function of shape (subdivision_size,)
      center: center of the hat function of shape (subdivision_size,)
      right: right boundary of the hat function of shape (subdivision_size,)
      x: points at which to evaluate the hat function of shape (n_points,)
    
    Returns:
      value of the hat function at x of shape (n_points,)
    """
    x = jnp.clip(x, left, right)
    increase = (x - left) / (center - left)
    decrease = (right - x) / (right - center)
    return jnp.where(x < center, increase, decrease) * self.max_value

  def evaluate_1D(self,
                  variable_1d: Variable1D,
                  x_1d: jnp.array) -> jnp.array:
    """Evaluate the hat functions in 1D.
    
    Args:
      variable_1d: variable of shape (subdivision_size,)
      x_1d: points at which to evaluate the hat functions of shape (n_points,)
      
    Returns:
      values of the hat functions at x of shape (n_points, subdivision_size)
    """
    t_left = variable_1d.subdivision[:-2]
    t_center = variable_1d.subdivision[1:-1]
    t_right = variable_1d.subdivision[2:]
    parallelized = jax.vmap(self.evaluate_single_hat, in_axes=(0, 0, 0, None), out_axes=0)
    evaluations = parallelized(t_left, t_center, t_right, x_1d)  # parallelized over all hat functions.
    evaluations = evaluations.transpose()
    return evaluations

  def evaluate_nD(self,
                  variable_block: VariableBlock,
                  x: jnp.array,
                  reshape: bool = False) -> jnp.array:
    """Evaluate the hat functions in nD.
    
    Args:
      variable_block: block of variables containing n_variables.
      x: points at which to evaluate the hat functions, of shape (n_points, n_variables)
        the variables must be ordered as in the block.
      reshape: whether to reshape the output to (n_points, subdivision_size_0, ..., subdivision_size_{n_variables-1})

    Returns:
      values of the hat functions at x, of shape
        (n_points, subdivision_size_0, ..., subdivision_size_{n_variables-1}) if reshape is True
        (n_points, subdivision_size_0 * ... * subdivision_size_{n_variables-1}) otherwise
    """
    x_1ds = unstack(x, axis=1)  # x_1ds is a list of arrays of shape (n_points,)
    phi_1Ds = [self.evaluate_1D(variable_1d, x_1d) for variable_1d, x_1d in zip(variable_block, x_1ds)]
    # phi_1D is a list of arrays of shape (n_points, subdivision_size)
    Phi = phi_1Ds[0]
    for phi_1D in phi_1Ds[1:]:
      # p denotes the point index, t denotes the hat function index
      # Phi is indexed by (p, i), phi_1D is indexed by (p, j)
      # p index will go from 0 to n_points-1
      # j index will go from 0 to subdivision_size-1
      # i index will go from 0 to prod_{k=0}^{j-1} subdivision_size_k
      Phi = jnp.einsum('pi,pj->pij', Phi, phi_1D)  # Phi is now of shape (n_points, subdivision_size_prev, subdivision_size_next)
      # einsum without contraction is equivalent to outer product
      Phi = Phi.reshape((Phi.shape[0], -1))  # Phi is now of shape (n_points, subdivision_size_prev * subdivision_size_next)
    # Phi is now of shape (n_points, subdivision_size_0 * ... * subdivision_size_{n_variables-1})
    if reshape:
      shape_multi_indices = tuple([x.shape[0]] + [variable_1d.basis_size() for variable_1d in variable_block])
      Phi = Phi.reshape(shape_multi_indices)  # Phi is now of shape (n_points, subdivision_size_0, ..., subdivision_size_{n_variables-1})
    return Phi
