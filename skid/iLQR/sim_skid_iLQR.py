import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class Skid_Steer_Simulator():
  def __init__(self, A):
    self.A = A

  def _f(self, x, u):
    A = self.A

    # Rotation matrix
    R = np.array([[cos(x[2]), -sin(x[2]), 0],
                  [sin(x[2]), cos(x[2]), 0],
                  [0, 0, 1]])

    sdot = R @ A @ u

    return sdot
  def F(self, xc, uc, dt):
    # Simulate the open loop skid steer for one step
    def f(_, x):
      return self._f(x, uc)
    sol = solve_ivp(f, (0, dt), xc, first_step=dt)
    return sol.y[:, -1].ravel()
