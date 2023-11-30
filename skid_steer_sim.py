import numpy as np
from math import sin, cos, pi
from scipy.integrate import solve_ivp
from skid_steer import SkidSteerVehicle
import matplotlib.pyplot as plt


def simulate_skid_steer(x0, tf, skid_steer, use_mpc=True, use_mpc_with_clf=False, use_clf_qp=False):
    print("Simulating skid-steer")
    # Simulates a stabilized maneuver on the 2D skid_steer
    # system, with an initial value of x0
    t0 = 0.0
    n_points = 1000

    dt = 1e-2

    x = [x0]
    #u = [np.zeros((3,))]
    u = [np.zeros((2,))]
    t = [t0]
    print(f"Dimensions of x {x}")

    while np.linalg.norm(np.array(x[-1][0:2])) > 1e-3 and t[-1] < tf:
        current_time = t[-1]
        current_x = x[-1]
        current_u_command = np.zeros(2)

        if use_mpc:
            current_u_command = skid_steer.compute_mpc_feedback(current_x, use_mpc_with_clf)
        elif use_clf_qp:
            current_u_command = skid_steer.compute_clf_qp_feedback(current_x)
        else:
            current_u_command = skid_steer.compute_lqr_feedback(current_x)

        current_u_real = np.clip(current_u_command, skid_steer.umin, skid_steer.umax)

        # Autonomous ODE for constant inputs to work with solve_ivp
        def f(t, x):
            return skid_steer.continuous_time_full_dynamics(current_x, current_u_real)

        # Integrate one step
        sol = solve_ivp(f, (0, dt), current_x, first_step=dt)

        # Record time, state, and inputs
        t.append(t[-1] + dt)
        x.append(sol.y[:, -1])
        u.append(current_u_command)

    x = np.array(x)
    u = np.array(u)
    t = np.array(t)
    return x, u, t

'''
def plot_x_and_u(x, u, t, name):
    plt.figure()
    ax = plt.axes()
    plt.plot(0, 0, 'o', label='target position')
    plt.plot(x[0, 0], x[0, 1], 'o', label='initial position')
    plt.plot(x[:, 0], x[:, 1], label='actual trajectory')
    plt.xlabel("y (m)")
    plt.ylabel("z (m)")
    plt.legend()
    ax.set_aspect('equal', 'datalim')
    ax.legend(loc='upper right')
    plt.title("Skid Steer trajectory (" + name + ")")

    plt.figure()
    plt.plot(t[1:], u[1:])
    plt.xlabel("time (s)")
    plt.ylabel("u (N)")
    plt.legend(["u1", "u2"])
    plt.title(name + " commanded inputs")
'''
'''
if __name__ == '__main__':
    # Quadrotor
    #R = np.eye(2)
    #Q = np.diag([10, 10, 1, 1, 1, 1])

    # x0 params [y, z, 0, ydot, zdot, thetadot]
    # x0 = np.array([0.5, 0.5, 0, 1, 1, 0])

    # Skid steer
    R = np.eye(3)
    Q = np.diag([1, 1, 1])
    Qf = Q

    skid_steer = SkidSteerVehicle(Q, R, Qf)

    # Initial state
    d_rand = 1
    x0 = np.array([2, 2, 3])

    tf = 10

    x, u, t = simulate_skid_steer(x0, tf, skid_steer)
    plot_x_and_u(x, u, t, "MPC")
    x, u, t = simulate_skid_steer(x0, tf, skid_steer, False)
    plot_x_and_u(x, u, t, "LQR")
    x, u, t = simulate_skid_steer(x0, tf, skid_steer, True, True, False)

    # Initial state to remain in the 1-sublevel of V
    plot_x_and_u(x, u, t, "MPC using CLF")
    x0 = np.array([2, 2, 3])
    x, u, t = simulate_skid_steer(x0, tf, skid_steer, False, False, True)
    plot_x_and_u(x, u, t, "CLF QP-version")

    plt.show()
'''