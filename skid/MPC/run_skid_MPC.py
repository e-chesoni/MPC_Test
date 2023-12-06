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
    '''
    R = np.eye(2) * 5
    #Q = np.diag([10, 10, 1])
    Q = np.diag([10, 10, 0])
    Qf = Q
    '''
    '''
    Q = .01 * np.eye(3)
    Q[2, 2] = 0  # Let system turn freely (no cost)
    R = np.eye(2) * 0.0000001

    Qf = 1e2 * np.eye(3)
    Qf[2, 2] = 0
    '''
    Q = context.Q
    R = context.R
    Qf = context.Qf

    u_guess = [np.zeros((2,))] * (N - 1)

    # End time of the simulation

    # Construct our skid steer controller
    return SkidMPC(start, end, u_guess, N, dt, Q, R, Qf)

def simulate_skid_steer_MPC(skid_steer, context):
    print("Running Skid Steer MPC...")

    # Set skid steer's initial state and simulate
    x0 = np.array([0, 0.05, 3])  # THIS WORKS (anything between 3 - 20 seems to get to target)

    # Set desired position
    #x_d = np.zeros(3)
    #x_d = np.array([0, 4, 0])
    x_d = np.array([0, -0.05, 3])

    # TODO: FIX PROBLEM:
    ## iLQR working trajectory is too small to work with MPC (car is already there)

    ''''
    x0 = start
    x_d = end
    '''
    # TODO: Fix trajectory; seems to have a very hard time turning
    #skid_steer.set_destination(x_d)
    skid_steer.set_destination(context.end)

    # Run MPC
    x, u, t = simulate_skid_steer(x0, context.tf, context.dt, skid_steer)
    #x, u, t = simulate_skid_steer(context.start, context.tf, context.dt, skid_steer)

    # Run LQR
    #x, u, t = simulate_skid_steer(x0, tf, skid, False)

    anim, fig = skid_steer_MPC_animation(x, x_d, context.tf)
    #anim, fig = skid_steer_MPC_animation(x, context.end, context.tf)
    plt.show()

    return anim, fig