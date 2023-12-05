import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Dynamics for the skid Steer
def _f(x, u):
    # Skid steer dynamics parameters
    alpha_l = 0.9464
    alpha_r = 0.9253
    x_ICR_l = -0.2758
    x_ICR_r = 0.2998
    y_ICR_v = -0.0080

    A = np.array([[-y_ICR_v * alpha_l, y_ICR_v * alpha_r],
                  [x_ICR_r * alpha_l, -x_ICR_l * alpha_r],
                  [-alpha_l, alpha_r]])

    # Calculate continuous-time dynamics for global system
    R = np.array([[cos(x[2]), -sin(x[2]), 0],  # makes it go left
                  [sin(x[2]), cos(x[2]), 0],
                  [0, 0, 1]])
    sdot = R @ A @ u

    return sdot

def F(xc, uc, dt):
  # Simulate the open loop skid steer for one step
  def f(_, x):
    return _f(x, uc)
  sol = solve_ivp(f, (0, dt), xc, first_step=dt)
  return sol.y[:, -1].ravel()