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

  @property
  def basis_size(self):
    """Number of hat functions."""
    return len(self.subdivision)-2  # 2 values are sentinels

  @property
  def domain(self):
    """Domain of the variable."""
    return self.subdivision[1], self.subdivision[-2]

  @property
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

  @property
  def subdivision_shape(self):
    """Shape of the subdivision of the block."""
    return [v.basis_size for v in self.variables]

  def reshape_as_subdivision(self, x: jnp.array) -> jnp.array:
    """Reshape x as the subdivision of the block.
    
    Args:
      x: array of shape (L_b, batch_sizes) where
        batch_sizes is the shape of the batch dimensions (if any)
    
    Returns:
      array of shape (m_1, ..., m_d, batch_sizes)
    """
    return x.reshape(self.subdivision_shape + list(x.shape[1:]))

  @property
  def basis_size(self):
    """Number of hat functions in the block, denoted L_b.

    Note: L_b = prod_i m_i where m_i is the number of subdivisions of the i-th variable.
    
    Remark: named \mathcal{L}_j in the paper.
    """
    lengths = self.subdivision_shape
    return onp.prod(onp.array(lengths))

  def __len__(self):
    """Number of variables in the block."""
    return len(self.variables)

  def __iter__(self):
    return iter(self.variables)

  def __getitem__(self, item):
    return self.variables[item]

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


def isotropic_block(names, indices, domain, num_ticks):
  """Create an isotropic block of variables.
  
  Args:
    names: List of names of the variables.
    indices: indices of the variables in the full partition.
    domain: Domain of the variables.
    num_ticks: Number of subdivisions of each variable.
  """
  eps = (domain[1] - domain[0]) / num_ticks
  sentinel_a, sentinel_b = domain[0]-eps, domain[1]+eps
  subdivision = jnp.linspace(domain[0], domain[1], num=num_ticks)
  subdivision = jnp.concatenate([jnp.array([sentinel_a]), subdivision, jnp.array([sentinel_b])])
  return VariableBlock([Variable1D(name, i, subdivision) for i, name in zip(indices, names)])


def block_1D(name, index, domain, num_ticks):
  """Create a block containing a single 1D variable."""
  return isotropic_block([name], [index], domain, num_ticks)


@pytree
class VariablePartition:
  """Partition of variables with possibly different subdivisions.
  
  Attributes:
    blocks: List of disjoint variable blocks.
  """
  blocks: List[VariableBlock]

  def __len__(self):
    return len(self.blocks)

  def __iter__(self):
    return iter(self.blocks)
  
  @property
  def block_sizes(self):
    return [b.basis_size for b in self.blocks]

  @property
  def basis_size(self):
    lengths = self.block_sizes
    return jnp.sum(jnp.array(lengths))

  @property
  def names(self):
    return [n for b in self.blocks for n in b.names]

  @property
  def indices(self):
    return [i for b in self.blocks for i in b.indices]

  def split(self, ksi: jnp.array) -> List[jnp.array]:
    """Split x into blocks.
    
    Args:
      ksi: array of shape (L,)

    Returns:
      list of arrays of shape (L_b,) each
    """
    block_indices = onp.cumsum(onp.array(self.block_sizes))
    block_indices = block_indices[:-1]  # n blocks => n-1 indices
    return jnp.split(ksi, block_indices)

  def split_and_reshape(self, ksi: jnp.array) -> List[jnp.array]:
    """Split x into blocks and reshape each block.
    
    Args:
      ksi: array of shape (L,)

    Returns:
      list of arrays of shape (m_1, ..., m_d) each
    """
    pieces = self.split(ksi)
    splitted = []
    for piece, block in zip(pieces, self.blocks):
      # block contains many variables with subdivision of shape (m_1, ..., m_d)
      # Note: L_b = prod_i m_i where m_i is the number of subdivisions of the i-th variable.
      splitted.append(block.reshape_as_subdivision(piece))
    return splitted

  def __post_init__(self):
    names = [n for b in self.blocks for n in b.names]
    assert len(names) == len(set(names)), "Variable names must be unique"
