# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt

from quad.MPC.quadrotor import Quadrotor
from quad.MPC.animate_quad_MPC import quad_MPC_animation
from quad.MPC.sim_quad_MPC import simulate_quadrotor

# Global variables
tf = 10


def create_quadrotor():
    """
    Load in the animation function
    """
    # Weights of LQR cost
    R = np.eye(2)
    Q = np.diag([10, 10, 1, 1, 1, 1])
    Qf = Q

    # End time of the simulation

    # Construct our quad controller
    quadrotor = Quadrotor(Q, R, Qf)
    return quadrotor

def simulate_quadrotor_MPC(quadrotor, tf):
    print("Running Quadrotor MPC...")

    # Set quad's initial state and simulate
    x0 = np.array([0.5, 0.5, 0, 1, 1, 0])
    x, u, t = simulate_quadrotor(x0, tf, quadrotor)

    anim, fig = quad_MPC_animation(x, tf)
    plt.show()
    return anim, fig