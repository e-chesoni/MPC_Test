# This is a sample Python script.
import importlib

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables
import pydrake.symbolic as sym

import create_animation
import quadrotor
from quadrotor import Quadrotor
from quad_sim import simulate_quadrotor

# Need to reload the module to use the latest code
importlib.reload(quadrotor)
importlib.reload(create_animation)

# Global variables
tf = 10;

def create_quadrotor():
    """
    Load in the animation function
    """
    # Weights of LQR cost
    R = np.eye(2);
    Q = np.diag([10, 10, 1, 1, 1, 1]);
    Qf = Q;

    # End time of the simulation

    # Construct our quadrotor controller
    quadrotor = Quadrotor(Q, R, Qf);
    return quadrotor
def simulate_quadrotor_MPC(quadrotor, tf):
    # Set quadrotor's initial state and simulate
    x0 = np.array([0.5, 0.5, 0, 1, 1, 0])
    x, u, t = simulate_quadrotor(x0, tf, quadrotor)

    anim, fig = create_animation.create_animation(x, tf)
    plt.show()
    return anim, fig


if __name__ == '__main__':
    q = create_quadrotor()
    anim, fig = simulate_quadrotor_MPC(q, 10)
    print("done")