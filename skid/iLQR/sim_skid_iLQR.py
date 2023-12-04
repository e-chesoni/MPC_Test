import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Dynamics for the skid Steer
def _f(x, u):
  alpha_l = 1
  alpha_r = 9
  x_ICR_l = -2
  x_ICR_r = 2

  v_x, v_y, w_z = x[0], x[1], x[2]
  V_l, V_r = u[0], u[1]

  # Calculate continuous-time dynamics for global system
  xdot = v_x * cos(x[2]) - v_y * sin(x[2])
  ydot = v_x * sin(x[2]) + v_y * cos(x[2])
  wdot = (V_r - V_l) / (x_ICR_r - x_ICR_l) * (-alpha_l + alpha_r)  # angular rate between wheels

  sdot = np.array([xdot, ydot, wdot])
  return sdot


def F(xc, uc, dt):
  # Simulate the open loop skid steer for one step
  def f(_, x):
    return _f(x, uc)
  sol = solve_ivp(f, (0, dt), xc, first_step=dt)
  return sol.y[:, -1].ravel()