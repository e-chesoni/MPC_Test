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
# Skid Steer MPC start/end
start = np.array([2, 2, 3])
end = np.zeros(3)

# skid steer iLQR start/end
start = np.zeros((3,))
end = np.array([-0.001, 0.001, 0])  # works
#end = np.array([0.2, 0.8, 0])  # really, really slow but also works

class Model(object):
    def __init__(self):
        super().__init__()
        self.start = np.array([2, 2, 3])
        self.end = np.zeros(3)

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
            run_skid_iLQR.run_skid_iLQR(self.start, self.end)
        elif skid_mpc:
            print("Ok, you want to run skid steer MPC...")
            s = run_skid_MPC.create_skid_steer(start, end)
            anim, fig = run_skid_MPC.simulate_skid_steer_MPC(s, run_skid_MPC.tf)
        else:
            print("Invalid input; not running any models.")


if __name__ == '__main__':
    m = Model()
    m.set_start(start)
    m.set_end(end)
    m.run_model(False, False, False, True)

    print("done")
