import pandas as pd
import jax.numpy as jnp
from jax.tree_util import tree_map


def tree_matvec(A, x):
  """Matrix-vector product between a pytree and a vector."""
  return tree_map(lambda mat, vec: mat @ vec, A, x)


def regression_report(y_true, y_pred):
  abs_err = jnp.abs(y_pred - y_true)
  rel_err = jnp.where(y_true == 0, abs_err, abs_err / jnp.abs(y_true))
  evs = 1 - jnp.var(y_true - y_pred) / jnp.var(y_true)
  print(f"Mean absolute error: {jnp.mean(abs_err):.4f}")
  print(f"Mean relative error: {jnp.mean(rel_err):.4f}%")
  print(f"Explained variance score: {evs*100:.4f}%")
  df = pd.DataFrame({'absolute_error': abs_err, 'relative_error': rel_err}).T
  return df
