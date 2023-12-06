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
        self.start = np.array([2, 2, 3])
        self.end = np.zeros(3)
        self.context = context

        Q = .01 * np.eye(3)
        Q[2, 2] = 0  # Let system turn freely (no cost)
        R = np.eye(2) * 0.0000001

        Qf = 1e2 * np.eye(3)
        Qf[2, 2] = 0
        self.context.set_cost(Q, R, Qf)

    def set_start(self, s):
        print(f"Setting Main model start to: {s}")
        self.start = s

    def set_end(self, e):
        print(f"Setting Main model destination to: {e}")
        self.end = e

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
            #run_skid_iLQR.run_skid_iLQR(self.context.start, self.context.end, self.context.N, self.context.dt)

            run_skid_iLQR.run_skid_iLQR(self.context)
        elif skid_mpc:
            print("Ok, you want to run skid steer MPC...")
            skid = run_skid_MPC.create_skid_steer(self.context)
            anim, fig = run_skid_MPC.simulate_skid_steer_MPC(skid, self.context)
        else:
            print("Invalid input; not running any models.")


if __name__ == '__main__':
    c = Context(start, end, N, dt, tf)
    m = Model(c)
    m.set_start(start)
    m.set_end(end)
    m.run_model(False, False, False, True)

    print("done")
