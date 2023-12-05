# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt

from skid.MPC.animate_skid_MPC import skid_steer_MPC_animation
from skid.MPC.skid_MPC import SkidSteerVehicle
from skid.MPC.sim_skid_MPC import simulate_skid_steer

# Global variables
tf = 10
def create_skid_steer():
    """
    Load in the animation function
    """
    # Weights of LQR cost
    R = np.eye(2) * 5
    #Q = np.diag([10, 10, 1])
    Q = np.diag([10, 10, 0])
    Qf = Q

    # End time of the simulation

    # Construct our skid steer controller
    return SkidSteerVehicle(Q, R, Qf)

def simulate_skid_steer_MPC(skid_steer, tf):
    print("Running Skid Steer MPC...")

    # Set skid steer's initial state and simulate
    x0 = np.array([2, 2, 3])  # THIS WORKS (anything between 3 - 20 seems to get to target)

    # Set desired position
    x_d = np.zeros(3)
    x_d = np.array([0, -4, 0])
    x_d = np.array([0, 4, 0])
    # TODO: Fix trajectory; seems to have a very hard time turning
    skid_steer.set_destination(x_d)

    # Run MPC
    x, u, t = simulate_skid_steer(x0, tf, skid_steer)

    # Run LQR
    #x, u, t = simulate_skid_steer(x0, tf, skid, False)

    anim, fig = skid_steer_MPC_animation(x, x_d, tf)
    plt.show()

    return anim, fig