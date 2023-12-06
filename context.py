import numpy as np

# Global variables
tf = 10  # tf 1 -> 100 iterations, 10 -> 1000
N = 10  # 3000
dt = 0.01  # 0.01

# Skid Steer MPC start/end
# start = np.array([2, 2, 3])
# end = np.zeros(3)

# start = np.array([2, 2, 3])
# end = np.array([0, -4, 0])

# skid steer iLQR start/end
start = np.zeros((3,))
end = np.array([-0.001, 0.001, 0])  # works

# end = np.array([0.2, 0.8, 0])  # really, really slow but also works

class Context(object):
    def __init__(self, s, e, n, step, final_time):
        self.start = s
        self.end = e
        self.N = n
        self.dt = step
        self.tf = final_time

        self.Q = np.eye(3)
        self.Qf = np.eye(3)
        self.R = np.eye(2)

    def set_cost(self, Q, R, Qf):
        self.Q = Q
        self.Qf = Qf
        self.R = R
