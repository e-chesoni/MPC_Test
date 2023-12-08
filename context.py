import numpy as np

# Global variables
tf = 10  # tf 1 -> 100 iterations, 10 -> 1000
#N = 3000
N = 1000  # TODO: NOTE -- using different N to calculate iLQR and MPC
dt = 0.01

# Skid Steer MPC start/end
# start = np.array([2, 2, 3])
# end = np.zeros(3)

# start = np.array([2, 2, 3])
# end = np.array([0, -4, 0])

# skid steer iLQR start/end
start = np.array([0, 0.01, 0])
#end = np.array([-0.001, 0.001, 0])  # Runs in iLQR and MPC
end = np.array([-0.01, 0.02, 0])  # Runs in iLQR and MPC
#end = np.array([-0.02, 0.02, 0])  # Runs in iLQR; stalls in MPC
#end = np.array([-0.03, 0.03, 0])  # takes too long
#end = np.array([0.2, 0.8, 0])  # Works with N = 3000 (SLOWLY with iLQR; doesn't run in MPC)
#end = np.array([0.2, 0.8, 0])

Q = .01 * np.eye(3)
Q[2, 2] = 0  # Let system turn freely (no cost)
R = np.eye(2) * 0.0000001
#R = np.eye(2) * 1e-6

Qf = 1e2 * np.eye(3)
Qf[2, 2] = 0

# TODO: Pick a guess that matches start and end
#u_guess = [np.zeros((2,))] * (N - 1)
guess = np.array([-0.1, 0.1])
u_guess = [guess] * (N - 1)


class Context(object):
    def __init__(self):
        self.start = start
        self.end = end
        self.N = N
        self.dt = dt
        self.tf = tf

        self.u_guess = u_guess

        self.Q = np.eye(3)
        self.Qf = np.eye(3)
        self.R = np.eye(2)

        self.set_cost(Q, R, Qf)

        print(f"{'*' * 70}\n"
              f"{' '* 25}Creating Context...{' '* 25}\n"
              f"{'*' * 70}\n"
              f"Starting at: {self.start}\n"
              f"Navigating to: {self.end}\n"
              f"Cost Q: {self.Q}\n"
              f"Cost R: {self.R}\n"
              f"Final Q (Qf): {self.Qf}\n"
              f"N: {self.N}\n"
              f"Change in t (dt): {self.dt}\n"
              f"Final time (tf): {self.tf}\n"
              f"{'*' * 70}\n"
              f"{' '* 23}Context Setup Complete.{' '* 23}\n"
              f"{'*' * 70}\n"
              )

    def set_cost(self, Q, R, Qf):
        self.Q = Q
        self.Qf = Qf
        self.R = R
