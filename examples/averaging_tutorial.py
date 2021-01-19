"""
Pulled from the tutorials jupyter notebook at
https://github.com/cvxgrp/cvxpylayers/blob/master/examples/torch/tutorial.ipynb
This is just to check that everything is working for basic examples.
Javier, 1/19/21
"""
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
torch.set_default_dtype(torch.double)

np.set_printoptions(precision=3, suppress=True)

n = 7

# Define variables & parameters
x = cp.Variable()
y = cp.Parameter(n)

# Define objective and constraints
objective = cp.sum_squares(y - x)
constraints = []

# Synthesize problem
prob = cp.Problem(cp.Minimize(objective), constraints)

# Set parameter values
y.value = np.random.randn(n)

# Solve problem in one line
prob.solve(requires_grad=True)
print("solution:", "%.3f" % x.value)
print("analytical solution:", "%.3f" % np.mean(y.value))

# Set gradient wrt x
x.gradient = np.array([1.])

# Differentiate in one line
prob.backward()
print("gradient:", y.gradient)
print("analytical gradient:", np.ones(y.size) / n)
