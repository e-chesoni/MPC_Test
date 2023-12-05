# MPCs
import numpy as np

import quad.MPC.run_quad_MPC as run_quad_MPC
import skid.MPC.run_skid_MPC as run_skid_MPC

# iLQRs
import quad.iLQR.run_quad_iLQR as run_quad_iLQR
import skid.iLQR.run_skid_iLQR as run_skid_iLQR


'''
TODO: MPC + iLQR
Incorp. ilqr with mpc:
you get uu, xx, and kk from ilqr
linearize about each time step in MPC:
    Get derivative of uu and xx
'''

# Global variables
tf = 10

start = np.array([2, 2, 3])
end = np.zeros(3)


class Model(object):
    def __init__(self):
        super().__init__()
        self.start = np.array([2, 2, 3])
        self.end = np.zeros(3)

    def set_start(self, start):
        print(f"Setting start to: {start}")
        self.start = start

    def set_end(self, end):
        print(f"Setting destiination to: {end}")
        self.end = end

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
            run_skid_iLQR.run_skid_iLQR()
        elif skid_mpc:
            print("Ok, you want to run skid steer MPC...")
            s = run_skid_MPC.create_skid_steer()
            anim, fig = run_skid_MPC.simulate_skid_steer_MPC(s, run_skid_MPC.tf)
        else:
            print("Invalid input; not running any models.")


def run_model(quad_ilqr, quad_mpc, skid_ilqr, skid_mpc):
    if quad_ilqr:
        print("Ok, you want to run quadrotor iLQR...")
        run_quad_iLQR.run_quad_iLQR()
    elif quad_mpc:
        print("Ok, you want to run quadrotor mpc...")
        q = run_quad_MPC.create_quadrotor()
        anim, fig = run_quad_MPC.simulate_quadrotor_MPC(q, run_quad_MPC.tf)
    elif skid_ilqr:
        print("Ok, you want to run skid steer iLQR...")
        run_skid_iLQR.run_skid_iLQR()
    elif skid_mpc:
        print("Ok, you want to run skid steer MPC...")
        s = run_skid_MPC.create_skid_steer()
        anim, fig = run_skid_MPC.simulate_skid_steer_MPC(s, run_skid_MPC.tf)
    else:
        print("Invalid input; not running any models.")


if __name__ == '__main__':
    m = Model()
    m.set_start(start)
    m.set_end(end)
    m.run_model(False, False, True, False)

    print("done")
