import numpy as onp
import jax.numpy as np

from jax import grad, jit, vmap, value_and_grad
from jax import random

import time

key = random.PRNGKey(1)

x = random.uniform(key, (1000, 1000))

t = time.time()
onp.dot(x, x)
print("Time: {} s".format(time.time() - t))
t = time.time()
np.dot(x, x)
print("Time: {} s".format(time.time() - t))
t = time.time()
np.dot(x, x).block_until_ready()
print("Time: {} s".format(time.time() - t))


def ReLU(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)


jit_ReLU = jit(ReLU)


def finite_diff_grad(x):
    """ Compute the finite difference derivative approx for the ReLU"""
    return np.array((ReLU(x + 1e-3) - ReLU(x - 1e-3)) / (2 * 1e-3))


print("Jax Grad: ", jit(grad(jit(ReLU)))(2.))
print("FD Gradient:", finite_diff_grad(2.))


batch_dim = 32
feature_dim = 100
hidden_dim = 512

X = random.normal(key, (batch_dim, feature_dim))

params = [random.normal(key, (feature_dim, hidden_dim)), random.normal(key, (hidden_dim,))]


def relu_layer(params, x):
    w, b = params
    return ReLU(np.dot(x, w) + b)


def vmap_relu(params, x):
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))(params, x)


out = np.stack([relu_layer(params, X[i, :]) for i in range(X.shape[0])])
out = vmap_relu(params, X)

from jax.scipy.special import logsumexp
from jax.example_libraries.optimizers import adam


