import numpy as np
from skid.iLQR.skid_iLQR import skid_iLQR

import matplotlib.pyplot as plt


def run_skid_iLQR(x0, x_goal):
    print("Running Skid Steer iLQR...")

    # Set up the iLQR problem
    N = 10  # 3000
    dt = 0.01  # 0.01

    # TODO: Adjust the costs as needed for convergence
    Q = .01 * np.eye(3)
    Q[2, 2] = 0  # Let system turn freely (no cost)
    R = np.eye(2) * 0.0000001

    Qf = 1e2 * np.eye(3)
    Qf[2, 2] = 0

    ilqr = skid_iLQR(x_goal, N, dt, Q, R, Qf)

    # initial guess for the input
    u_guess = [np.zeros((2,))] * (N - 1)

    x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(x0, u_guess)
    print(np.array(x_sol)[:, -1])
    print(u_sol)

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
