import numpy as np
import importlib
import quad.iLQR.quad_iLQR
# importlib.reload(quad_iLQR)
from quad.iLQR.quad_iLQR import iLQR

import matplotlib.pyplot as plt


def run_quad_iLQR():
    print("Running Quadrotor iLQR...")

    # Set up the iLQR problem
    N = 200  # TODO: What are the units of N?
    dt = 0.02
    x_goal = np.array([-0.5, 1.8, 0, 0, 0, 0])

    # TODO: Adjust the costs as needed for convergence
    Q = np.eye(6)
    Qf = 1e2 * np.eye(6)
    R = 1e-3 * np.eye(2)

    ilqr = iLQR(x_goal, N, dt, Q, R, Qf)

    # Initial state at rest at the origin
    x0 = np.zeros((6,))

    # initial guess for the input is just hovering in place
    u_guess = [0.5 * 9.81 * ilqr.m * np.ones((2,))] * (N - 1)

    x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(x0, u_guess)

    # Visualize the solution
    xx = np.array(x_sol)
    plt.plot(x0[0], x0[1], marker='x', color='blue')
    plt.plot(x_goal[0], x_goal[1], marker='x', color='orange')
    plt.plot(xx[:, 0], xx[:, 1], linestyle='--', color='black')
    plt.title('iLQR Trajectory Solution')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(['Start', 'Goal', 'Trajectory'])

    # Show plot for pycharm
    plt.show()
