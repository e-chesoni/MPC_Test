import numpy as np
from scipy.integrate import solve_ivp


def simulate_skid_steer(x0, tf, dt, skid_steer, use_mpc=True, use_mpc_with_clf=False, use_clf_qp=False):
    print("Simulating skid-steer")
    # Simulates a stabilized maneuver on the 2D skid
    # system, with an initial value of x0
    t0 = 0.0

    x = [x0]
    u = [np.zeros((2,))]
    t = [t0]

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
