from dataclasses import dataclass as python_dataclass

import jax

import gpjax as gpx
import jax.numpy as jnp
import optax as ox

from jax import jit
from flax.struct import dataclass as pytree

from kernels import MultivariateKernel


@python_dataclass
class FittedGPJaxKernel(MultivariateKernel):
  kernel : gpx.kernels.AbstractKernel

  def __call__(self, x: jnp.array) -> jnp.array:
    return self.kernel.gram(x).to_dense()
  
  def parameters(self):
    return self.kernel.trainables()


def maximum_mll_unconstrained_kernel(kernel: gpx.kernels.AbstractKernel,
                                     x_train: jnp.array,
                                     y_train: jnp.array) -> FittedGPJaxKernel:
  """Optimize the hyper-parameter of a kernel on a training set.
  
  Find the optimal hyper-parameters of a kernel by maximizing the marginal log-likelihood,
  on the unconstrained optimization problem.
  
  Args:
    kernel: gpx kernel to optimize
    x_train: training points of shape (n_points, n_variables)
    y_train: training values of shape (n_points,)

  Returns:
    a kernel with default hyper-parameters optimized on the training set.
  """

  dataset = gpx.Dataset(X=x_train, y=y_train.reshape((-1, 1)))
  meanf = gpx.mean_functions.Zero()
  prior = gpx.Prior(mean_function=meanf, kernel=kernel)
  likelihood = gpx.Gaussian(num_datapoints=dataset.n)
  posterior = prior * likelihood

  # optimize kernel hyper-parameters
  negative_mll = gpx.objectives.ConjugateMLL(negative=True)
  negative_mll = jit(negative_mll)

  key = jax.random.PRNGKey(123)
  optimiser = ox.adam(learning_rate=1e-2)
  opt_posterior, history = gpx.fit(
      model=posterior,
      objective=negative_mll,
      train_data=dataset,
      optim=optimiser,
      key=key,
  )

  return FittedGPJaxKernel(opt_posterior.prior.kernel)
