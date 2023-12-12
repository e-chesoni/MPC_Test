import numpy as np
from skid.iLQR.skid_iLQR import skid_iLQR
from context import *
import matplotlib.pyplot as plt

#def run_skid_iLQR(x0, x_goal, N, dt):
def run_skid_iLQR(context):
    print("Running Skid Steer iLQR...")
    x0 = context.start
    x_goal = context.end
    Q = context.Q
    R = context.R
    Qf = context.Qf

    ilqr = skid_iLQR(x_goal, N, dt, Q, R, Qf)

    # initial guess for the input
    u_guess = context.u_guess

    x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(x0, u_guess)
    print(f"x_sol: {x_sol}")
    print(f"u_sol: {u_sol}")

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
