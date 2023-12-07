# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import matplotlib.pyplot as plt

from skid.MPC.animate_skid_MPC import skid_steer_MPC_animation
from skid.MPC.skid_MPC import SkidMPC
from skid.MPC.sim_skid_MPC import simulate_skid_steer


def create_skid_steer(context):
    start = context.start
    end = context.end
    N = context.N
    dt = context.dt
    """
    Load in the animation function
    """
    # Weights of LQR cost
    Q = context.Q
    R = context.R
    Qf = context.Qf

    u_guess = [np.zeros((2,))] * (N - 1)

    return SkidMPC(start, end, u_guess, N, dt, Q, R, Qf)


def simulate_skid_steer_MPC(skid_steer, context):
    print("Running Skid Steer MPC...")

    # Set skid steer's initial state and simulate
    x0 = context.start  # Working as long as this is not an array of zeros; MPC can't start at [0, 0, 0]

    # Set desired position
    x_d = context.end
    skid_steer.set_destination(context.end)

    # Run MPC
    x, u, t = simulate_skid_steer(x0, context.tf, context.dt, skid_steer)

    # Run LQR
    #x, u, t = simulate_skid_steer(x0, tf, skid, False)

    anim, fig = skid_steer_MPC_animation(x, x_d, context.tf)
    plt.show()

    return anim, fig