import pandas as pd
import jax.numpy as jnp
from jax.tree_util import tree_map


def tree_matvec(A, x):
  """Matrix-vector product between a pytree and a vector."""
  return tree_map(lambda mat, vec: mat @ vec, A, x)


def regression_report(y_true, y_pred):
  mse_err = (y_true - y_pred)**2
  abs_err = jnp.abs(y_pred - y_true)
  evs = 1 - jnp.var(y_true - y_pred) / jnp.var(y_true)
  print(f"Mean squared error: {jnp.mean(mse_err):.4f}")
  print(f"Mean absolute error: {jnp.mean(abs_err):.4f}")
  print(f"Explained variance score: {evs*100:.4f}%")
  df = pd.DataFrame({'absolute_error': abs_err}).T
  df.columns = [f'x[{i}]' for i in range(len(y_true))]
  return df
