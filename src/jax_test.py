import numpy as onp
import jax.numpy as np

from jax import grad, jit, vmap
from jax import random

import time

key = random.PRNGKey(1)

X = random.uniform(key, (1000, 1000))

t = time.time()
onp.dot(X, X)
print("Time: {} s".format(time.time() - t))
t = time.time()
np.dot(X, X)
print("Time: {} s".format(time.time() - t))
t = time.time()
np.dot(X, X).block_until_ready()
print("Time: {} s".format(time.time() - t))


def relu(x):
    """ Rectified Linear Unit (ReLU) activation function """
    return np.maximum(0, x)


jit_ReLU = jit(relu)


def finite_diff_grad(x):
    """ Compute the finite difference derivative approx for the ReLU"""
    return np.array((relu(x + 1e-3) - relu(x - 1e-3)) / (2 * 1e-3))


print("Jax Grad: ", jit(grad(jit(relu)))(2.))
print("FD Gradient:", finite_diff_grad(2.))


batch_dim = 32
feature_dim = 100
hidden_dim = 512

X = random.normal(key, (batch_dim, feature_dim))

params = [random.normal(key, (feature_dim, hidden_dim)), random.normal(key, (hidden_dim,))]


def relu_layer(theta, x):
    w, b = theta
    return relu(np.dot(x, w) + b)


def vmap_relu(theta, x):
    return jit(vmap(relu_layer, in_axes=(None, 0), out_axes=0))(theta, x)


t = time.time()
np.stack([relu_layer(params, X[i, :]) for i in range(X.shape[0])])
print("Time: {} s".format(time.time() - t))
t = time.time()
vmap_relu(params, X)
print("Time: {} s".format(time.time() - t))
