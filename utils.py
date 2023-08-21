from jax.tree_util import tree_map


def tree_matvec(A, x):
  """Matrix-vector product between a pytree and a vector."""
  return tree_map(lambda mat, vec: mat @ vec, A, x)
