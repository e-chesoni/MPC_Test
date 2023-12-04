# MPCs
import quad.MPC.run_quad_MPC as run_quad_MPC
import skid.MPC.run_skid_MPC as run_skid_MPC

# iLQRs
import quad.iLQR.run_quad_iLQR as run_quad_iLQR
import skid.iLQR.run_skid_iLQR as run_skid_iLQR


# Global variables
tf = 10


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
    run_model(False, False, False, True)

    print("done")
