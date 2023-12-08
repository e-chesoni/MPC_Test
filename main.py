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


if __name__ == '__main__':
    c = Context()
    m = Model(c)
    m.run_model(False, False, True, False)