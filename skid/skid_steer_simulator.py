from scipy.integrate import solve_ivp

from skid.skid_steer_system import SkidSteerSystem


class Skid_Steer_Simulator():
  def __init__(self):
    pass

  @staticmethod
  def f(x, u):
    # Get a matrix from self (defined in initialization by skid_iLQR)
    A = SkidSteerSystem.get_kinematics()

    # Get rotation matrix using helper
    R = SkidSteerSystem.rotate(x)

    sdot = R @ A @ u

    return sdot

  @staticmethod
  def F(xc, uc, dt):
    # Simulate the open loop skid steer for one step
    def f(_, x):
      return Skid_Steer_Simulator.f(x, uc)
    sol = solve_ivp(f, (0, dt), xc, first_step=dt)
    return sol.y[:, -1].ravel()
