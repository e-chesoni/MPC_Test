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


class SkidSteerVehicle(object):
    def __init__(self, Q, R, Qf, umin, umax, a, I, m):
        print("Init SkidSteerVehicle...")

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Input limits
        self.umin = umin
        self.umax = umax

        # Vehicle parameters
        self.a = a  # Distance from the center of mass to the wheel
        self.I = I  # Moment of inertia
        self.m = m  # Mass

        self.n_x = 6  # number of states
        self.n_u = 2  # number of control inputs

    def x_d(self):
        # Nominal state
        return np.array([0, 0, 0, 0, 0, 0])

    def u_d(self):
        # Nominal input
        return np.array([self.m * self.g / 2, self.m * self.g / 2])

    # Continuous-time dynamics for the skid-steer vehicle
    def continuous_time_full_dynamics(self, x, u, alpha_l, alpha_r):
        # TODO: verify continuous time dynamics calculation for skid steer
        # Get state variables
        v_x, v_y, w_z = x[0], x[1], x[2]
        V_l, V_r = u[0], u[1]

        # Parameters from the kinematic model
        x_ICR_v = -v_y / w_z
        x_ICR_l = (alpha_l * V_l - v_y) / w_z
        x_ICR_r = (alpha_r * V_r - v_y) / w_z
        y_ICR_v = y_ICR_l = y_ICR_r = v_x / w_z

        # Calculate continuous-time dynamics
        xdot = v_x * cos(x[2]) - v_y * sin(x[2])
        ydot = v_x * sin(x[2]) + v_y * cos(x[2])
        wdot = (V_r - V_l) / (x_ICR_r - x_ICR_l) * (-alpha_l + alpha_r)

        # Update the state derivatives
        v_xdot = -v_y / (x_ICR_r - x_ICR_l) * (-y_ICR_v * alpha_l * V_l + y_ICR_v * alpha_r * V_r)
        v_ydot = -v_y / (x_ICR_r - x_ICR_l) * (x_ICR_r * alpha_l * V_l - x_ICR_l * alpha_r * V_r)
        w_zdot = -v_y / (x_ICR_r - x_ICR_l) * (-alpha_l * V_l + alpha_r * V_r)

        xdot = np.array([v_xdot, v_ydot, wdot, xdot, ydot, w_zdot])
        return xdot

    # Helper to compute A matrix
    def compute_A_matrix(self, V_l, V_r, alpha_l, alpha_r, v_x, v_y, w_z):
        # Compute A matrix based on provided dynamics
        x_ICR_v = -v_y / w_z
        x_ICR_l = (alpha_l * V_l - v_y) / w_z
        x_ICR_r = (alpha_r * V_r - v_y) / w_z
        y_ICR_v = y_ICR_l = y_ICR_r = v_x / w_z

        A = 1 / (x_ICR_r - x_ICR_l) * np.array([
            [-y_ICR_v * alpha_l, y_ICR_v * alpha_r],
            [x_ICR_r * alpha_l, -x_ICR_l * alpha_r],
            [-alpha_l, alpha_r]
        ])

        return A

    # Helper to compute B matrix
    # TODO: How do you calculate the B matrix?
    def compute_B_matrix(self):
        B = np.zeros((3, 2))

        return B

    # Linearized dynamics for Skid Steer Vehicle
    def continuous_time_linearized_dynamics(self):
        # Dynamics linearized at the fixed point
        # This function returns A and B matrix

        A = np.zeros((3, 2))
        # TODO: add parameters for A matrix
        # A = self.compute_A_matrix()

        B = np.zeros((3, 2))
        # TODO: may need to update based on B matrix calc verification result
        B = self.compute_B_matrix()

        return A, B

    # Discretization of linearized dyanmics for Skid Steer Vehicle
    def discrete_time_linearized_dynamics(self, T):
        # Discrete time version of the linearized dynamics at the fixed point
        # This function returns A and B matrix of the discrete time dynamics
        A_c, B_c = self.continuous_time_linearized_dynamics()
        A_d = np.identity(6) + A_c * T
        B_d = B_c * T

        return A_d, B_d

    def add_initial_state_constraint(self, prog, x, x_current):
        # TODO: verify you can use the same constraint as quadrotor for initial state
        for i in range(len(x_current)):
            prog.AddBoundingBoxConstraint(x_current[i], x_current[i], x[0][i])

        pass

    def add_input_saturation_constraint(self, prog, x, u, N):
        # TODO: update input limits (based on max velocity left and right)
        # Use AddBoundingBoxConstraint
        # The limits are available through self.umin and self.umax

        # TODO: understand why this limit works for quadrotor
        l_b = self.umin - self.u_d()
        u_b = self.umax - self.u_d()
        for k in range(N - 1):
            prog.AddBoundingBoxConstraint(l_b[0], u_b[0], u[k][0])
            prog.AddBoundingBoxConstraint(l_b[1], u_b[1], u[k][1])

        pass

    def add_dynamics_constraint(self, prog, x, u, N, T):
        # TODO: update dynamics constraint.
        # Use AddLinearEqualityConstraint(expr, value)

        A, B = self.discrete_time_linearized_dynamics(T)

        # TODO: understand why this limit works for quadrotor
        for k in range(N - 1):
            for i in range(len(x[k])):
                x_next = A @ x[k] + B @ u[k]
                prog.AddLinearEqualityConstraint(x_next[i] - x[k + 1][i], 0)

        pass

    def add_cost(self, prog, x, u, N):
        # TODO: verify you can use the same cost for skid steer simple model
        for k in range(N - 1):
            cost = x[k].T @ self.Q @ x[k] + u[k].T @ self.R @ u[k]
            prog.AddQuadraticCost(cost)

        pass

    def compute_mpc_feedback(self, x_current, use_clf=False):
        '''
        This function computes the MPC controller input u
        '''

        # Parameters for the QP
        N = 10
        T = 0.1

        # Initialize mathematical program and decalre decision variables
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
