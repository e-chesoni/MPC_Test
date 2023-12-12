import numpy as np
from context import *

# MPCs
import quad.MPC.run_quad_MPC as run_quad_MPC
import skid.MPC.run_skid_MPC as run_skid_MPC

# iLQRs
import quad.iLQR.run_quad_iLQR as run_quad_iLQR
import skid.iLQR.run_skid_iLQR as run_skid_iLQR


class Model(object):
    def __init__(self, context):
        super().__init__()
        self.context = context

        print(f"{'*' * 70}\n"
              f"{' ' * 20}Creating and Running Model...{' ' * 20}\n"
              f"{'*' * 70}\n"
              )

    def run_model(self, quad_ilqr, quad_mpc, skid_ilqr, skid_mpc):
        if quad_ilqr:
            print("Ok, you want to run quadrotor iLQR...")
            run_quad_iLQR.run_quad_iLQR()
        elif quad_mpc:
            print("Ok, you want to run quadrotor mpc...")
            q = run_quad_MPC.create_quadrotor()
            anim, fig = run_quad_MPC.simulate_quadrotor_MPC(q, run_quad_MPC.tf)
        elif skid_ilqr:
            print("Ok, you want to run skid steer iLQR...")
            run_skid_iLQR.run_skid_iLQR(self.context)
        elif skid_mpc:
            print("Ok, you want to run skid steer MPC...")
            skid = run_skid_MPC.create_skid_steer(self.context)
            anim, fig = run_skid_MPC.simulate_skid_steer_MPC(skid, self.context)
        else:
            print("Invalid input; not running any models.")

        print(f"\n{'*' * 70}\n"
              f"{' ' * 20}Trajectory Simulation Complete.{' ' * 20}\n"
              f"{'*' * 70}\n")

def plot_iLQR():
    import numpy as np
    import matplotlib.pyplot as plt

    # Given x_sol and u_sol
    x_sol = [np.array([0., 0.01, 0.]),
             np.array([-0.00036755, 0.00857642, 0.05042024]),
             np.array([-0.00084991, 0.01004241, 0.09705548]),
             np.array([-0.0013912, 0.01158114, 0.14136476]),
             np.array([-0.00193049, 0.01270468, 0.18506396]),
             np.array([-0.00244485, 0.01343044, 0.2293243]),
             np.array([-0.00293366, 0.01385884, 0.27513276]),
             np.array([-0.00342995, 0.01415062, 0.32365651]),
             np.array([-0.00419058, 0.01496442, 0.37799028]),
             np.array([-0.0080741, 0.02158554, 0.4767637])]

    u_sol = [np.array([-1.61875726, 1.48081421]),
             np.array([-1.20079399, 1.67285515]),
             np.array([-1.12301234, 1.6077198]),
             np.array([-1.14707313, 1.54515931]),
             np.array([-1.20360202, 1.52224811]),
             np.array([-1.27822197, 1.54223042]),
             np.array([-1.36914958, 1.61813899]),
             np.array([-1.47501011, 1.8712873]),
             np.array([-2.07135975, 4.02578979])]

    # Modified x_sol for straight line
    x_sol_straight = [np.array([0., 0.01, 0.]),
                      np.array([0.001, 0.01, 0.]),
                      np.array([0.002, 0.01, 0.]),
                      np.array([0.003, 0.01, 0.]),
                      np.array([0.004, 0.01, 0.]),
                      np.array([0.005, 0.01, 0.]),
                      np.array([0.006, 0.01, 0.]),
                      np.array([0.007, 0.01, 0.]),
                      np.array([0.008, 0.01, 0.]),
                      np.array([0.009, 0.01, 0.])]

    # Modified u_sol for straight line
    u_sol_straight = [np.array([1.61875726, -1.48081421]),
             np.array([1.20079399, -1.67285515]),
             np.array([1.12301234, -1.6077198]),
             np.array([1.14707313, -1.54515931]),
             np.array([1.20360202, -1.52224811]),
             np.array([1.27822197, -1.54223042]),
             np.array([1.36914958, -1.61813899]),
             np.array([1.47501011, -1.8712873]),
             np.array([2.07135975, -4.02578979])]

    #x_sol = x_sol_straight
    #u_sol = u_sol_straight

    # Extract x, y, and z components for plotting
    x_values = [point[0] for point in x_sol]
    y_values = [point[1] for point in x_sol]
    z_values = [point[2] for point in x_sol]

    # Plot x, y, z components
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x_values, label='x')
    plt.plot(y_values, label='y')
    plt.plot(z_values, label='z')
    plt.title('Trajectory Components')
    plt.xlabel('Time Steps')
    plt.legend()

    # Extract u1 and u2 components for plotting
    u1_values = [control[0] for control in u_sol]
    u2_values = [control[1] for control in u_sol]

    plt.subplot(1, 2, 2)
    plt.plot(u1_values, label='u1')
    plt.plot(u2_values, label='u2')
    plt.title('Control Input Components')
    plt.xlabel('Time Steps')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    #plot_iLQR()
    c = Context()
    m = Model(c)
    m.run_model(False, False, False, True)