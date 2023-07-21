from typing import List
import numpy as onp
import jax.numpy as jnp
from flax.struct import dataclass as pytree


@pytree
class Variable1D:
  """Variable with 1D subdivision.

  Remark: the subdivision is assumed to be increasing.
  Moreover two sentinel points are added at the beginning and end
  of the subdivision. This is to ensure that the hat functions
  are well defined.
  
  Attributes:
    name: Name of the variable.
    index: Index of the variable in the full partition.
    subdivision: 1D array of subdivision points.
  """
  name: str
  index: int
  subdivision: jnp.array

  def basis_size(self):
    """Number of hat functions."""
    return len(self.subdivision)-2  # 2 values are sentinels

  def domain(self):
    """Domain of the variable."""
    return self.subdivision[1], self.subdivision[-2]

  def sentinels(self):
    """Sentinel values of the variable."""
    return self.subdivision[0], self.subdivision[-1]
  
  @property
  def subdivision_without_sentinels(self):
    """Subdivision of the variable without the sentinels."""
    return self.subdivision[1:-1]

  def __post_init__(self):
    assert len(self.subdivision.shape) == 1, "Subdivision must be 1D"
    assert len(self.subdivision) >= 3, "Subdivision must have at least 3 points"
    assert jnp.all(jnp.diff(self.subdivision) > 0), "Subdivision must be increasing"


@pytree
class VariableBlock:
  """Block of variables with possibly different subdivisions.

  Attributes:
    variables: List of variables.
  """
  variables: List[Variable1D]

  def basis_size(self):
    """Number of hat functions in the block, denoted L_b.
    
    Remark: named \mathcal{L}_j in the paper.
    """
    lengths = [len(v) for v in self.variables]
    return onp.prod(onp.array(lengths))

  def __len__(self):
    """Number of variables in the block."""
    return len(self.variables)

  def __iter__(self):
    return iter(self.variables)

  @property
  def names(self):
    """Names of the variables in the full partition."""
    return [v.name for v in self.variables]

  @property
  def indices(self):
    """Indices of the variables in the full partition."""
    return [v.index for v in self.variables]

  def project(self, x: jnp.array) -> jnp.array:
    """Select the relevant entries of x corresponding to the block.
    
    Args:
      x: array of shape (n_points, n_variable)
    
    Returns:
      array of shape (n_points, n_variable_block)
    """
    return x[:, self.indices]

  def compute_subdivision_nd(self) -> jnp.array:
    """Compute the subdivision of the block in nD.
    
    Returns:
      arrays of shape (m_1 * ... * m_d, d)
      where m_i is the number of hat functions of the i-th variable
    """
    subdivisions = [v.subdivision_without_sentinels for v in self.variables]
    coords = jnp.meshgrid(*subdivisions, indexing='ij')  # list of arrays of shape (m-1, ..., m_d)
    # TODO: for univariate kernels it could be faster to use coords directly instead of the stacked version.
    coords = jnp.stack(coords, axis=-1)
    return coords.reshape((-1, len(subdivisions)))  # array of shape (m_1 * ... * m_d, d)

  def __post_init__(self):
    names = [v.name for v in self.variables]
    assert len(names) == len(set(names)), "Variable names must be unique"


def isotropic_block(names, domain, num_ticks):
  """Create an isotropic block of variables."""
  eps = (domain[1] - domain[0]) / num_ticks
  sentinel_a, sentinel_b = domain[0]-eps, domain[1]+eps
  subdivision = jnp.linspace(domain[0], domain[1], num=num_ticks)
  subdivision = jnp.concatenate([jnp.array([sentinel_a]), subdivision, jnp.array([sentinel_b])])
  return VariableBlock([Variable1D(name, i, subdivision) for i, name in enumerate(names)])


@pytree
class VariablePartition:
  """Partition of variables with possibly different subdivisions.
  
  Attributes:
    blocks: List of variable blocks.
  """
  blocks: List[VariableBlock]

  def len(self):
    return len(self.blocks)

  def basis_size(self):
    lengths = [len(b) for b in self.blocks]
    return onp.prod(onp.array(lengths))

  @property
  def names(self):
    return [n for b in self.blocks for n in b.names]

  @property
  def indices(self):
    return [i for b in self.blocks for i in b.indices]

  def __post_init__(self):
    names = [n for b in self.blocks for n in b.names]
    assert len(names) == len(set(names)), "Variable names must be unique"
