import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables


class Quadrotor(object):
    def __init__(self, Q, R, Qf):
        print("Initilizing Quadrotor...")
        self.g = 9.81
        self.m = 1
        self.a = 0.25
        self.I = 0.0625
        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Input limits
        self.umin = 0
        self.umax = 5.5

        self.n_x = 6
        self.n_u = 2

        self.DEBUG = 1  # 1 = log messages, 0 = don't log
        self.LOG_LEVEL = 3  # 0 = log nothing, 1 = shapes, 2 = variables, 3 = matrices

    ########################################################################
    #                                LOGGER                                #
    ########################################################################
    def log(self, message, arg, log_level, multi_line):
        if (self.DEBUG):
            if (self.LOG_LEVEL < log_level):
                pass
            else:
                if (multi_line):
                    print(f"Quadrotor -- {message}: \n {arg}")
                else:
                    print(f"Quadrotor -- {message}: {arg}")

    ########################################################################
    #                             END LOGGER                               #
    ########################################################################

    def x_d(self):
        # Nominal state
        return np.array([0, 0, 0, 0, 0, 0])

    def u_d(self):
        # Nominal input
        return np.array([self.m * self.g / 2, self.m * self.g / 2])

    def continuous_time_full_dynamics(self, x, u):
        # Dynamics for the quadrotor
        g = self.g
        m = self.m
        a = self.a
        I = self.I

        theta = x[2]
        ydot = x[3]
        zdot = x[4]
        thetadot = x[5]
        u0 = u[0]
        u1 = u[1]

        xdot = np.array([ydot,
                         zdot,
                         thetadot,
                         -sin(theta) * (u0 + u1) / m,
                         -g + cos(theta) * (u0 + u1) / m,
                         a * (u0 - u1) / I])
        return xdot

    def continuous_time_linearized_dynamics(self):
        # Dynamics linearized at the fixed point
        # This function returns A and B matrix
        A = np.zeros((6, 6))
        A[:3, -3:] = np.identity(3)
        A[3, 2] = -self.g

        B = np.zeros((6, 2))
        B[4, 0] = 1 / self.m
        B[4, 1] = 1 / self.m
        B[5, 0] = self.a / self.I
        B[5, 1] = -self.a / self.I

        return A, B

    def discrete_time_linearized_dynamics(self, T):
        # Discrete time version of the linearized dynamics at the fixed point
        # This function returns A and B matrix of the discrete time dynamics
        A_c, B_c = self.continuous_time_linearized_dynamics()
        A_d = np.identity(6) + A_c * T
        B_d = B_c * T

        return A_d, B_d

    def add_initial_state_constraint(self, prog, x, x_current):
        # TODO: impose initial state constraint.
        # Use AddBoundingBoxConstraint
        for i in range(len(x_current)):
            prog.AddBoundingBoxConstraint(x_current[i], x_current[i], x[0][i])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # TODO: impose input limit constraint.
        # Use AddBoundingBoxConstraint
        # The limits are available through self.umin and self.umax
        l_b = self.umin - self.u_d()
        u_b = self.umax - self.u_d()
        for k in range(N - 1):
            prog.AddBoundingBoxConstraint(l_b[0], u_b[0], u[k][0])
            prog.AddBoundingBoxConstraint(l_b[1], u_b[1], u[k][1])

    def add_dynamics_constraint(self, prog, x, u, N, T):
        # TODO: impose dynamics constraint.
        # Use AddLinearEqualityConstraint(expr, value)
        A, B = self.discrete_time_linearized_dynamics(T)

        for k in range(N - 1):
            for i in range(len(x[k])):
                x_next = A @ x[k] + B @ u[k]
                prog.AddLinearEqualityConstraint(x_next[i] - x[k + 1][i], 0)

    def add_cost(self, prog, x, u, N):
        # TODO: add cost.
        for k in range(N - 1):
            cost = x[k].T @ self.Q @ x[k] + u[k].T @ self.R @ u[k]
            prog.AddQuadraticCost(cost)

    def compute_mpc_feedback(self, x_current, use_clf=False):
        '''
        This function computes the MPC controller input u
        '''

        # Parameters for the QP
        N = 10
        T = 0.1

        # Initialize mathematical program and declare decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 6), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(6, "x_" + str(i))
        u = np.zeros((N - 1, 2), dtype="object")
        for i in range(N - 1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, x, u, N)
        self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_cost(prog, x, u, N)

        # Placeholder constraint and cost to satisfy QP requirements
        # TODO: Delete after completing this function
        # prog.AddQuadraticCost(0)
        # prog.AddLinearEqualityConstraint(0, 0)

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)

        u_mpc = np.zeros(2)
        # TODO: retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        # You should make use of result.GetSolution(decision_var) where decision_var
        # is the variable you want
        u_0 = result.GetSolution(u[0])
        u_mpc = u_0 + self.u_d()
        return u_mpc

    def compute_lqr_feedback(self, x):
        '''
        Infinite horizon LQR controller
        '''
        A, B = self.continuous_time_linearized_dynamics()
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -inv(self.R) @ B.T @ S
        u = self.u_d() + K @ x
        return u
