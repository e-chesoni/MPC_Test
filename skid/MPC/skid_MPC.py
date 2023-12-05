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

from skid.skid_steer_simulator import Skid_Steer_Simulator
from skid.skid_steer_system import SkidSteerSystem


class SkidSteerVehicle(object):
    def __init__(self, Q, R, Qf):
        print("Init SkidSteerVehicle...")

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Input limits
        self.umin = 1  # should be 0; no movement though. moves with 1
        self.umax = 5

        # Vehicle parameters
        '''
        self.a = 4  # Distance from the center of mass to the wheel
        self.length = 4.2
        self.width = 3.1
        self.m = 2  # Mass
        self.I = (1/12) * self.m * (self.length**2 + self.width**2)  # Moment of inertia
        '''

        # State and input totals
        self.n_x = 3  # number of states
        self.n_u = 2  # number of control inputs

        self.x_d = np.zeros(3)  # default destination is [0,0,0]

    def set_destination(self, d):
        print(f"Setting skid steer MPC destination to: {d}")
        self.x_d = d
        pass

    def u_d(self):
        # Nominal input
        return np.array([0, 0])

    # Continuous-time dynamics for the skid-steer vehicle
    def continuous_time_full_dynamics(self, x, u):
        sdot = Skid_Steer_Simulator.f(x, u)

        return sdot

    # Linearized dynamics for Skid Steer Vehicle
    def continuous_time_linearized_dynamics(self):
        # Dynamics linearized at the fixed point
        # This function returns A and B matrix

        # There is no A matrix in simple skid-steer system;
        # Use I with NxN dim where N = len(xdot)
        A = np.eye(3)

        A_kinematics = SkidSteerSystem.get_kinematics()
        R = SkidSteerSystem.rotate(self.x_d)

        B = R @ A_kinematics

        return A, B

    # Discretization of linearized dyanmics for Skid Steer Vehicle
    def discrete_time_linearized_dynamics(self, T):
        # Discrete time version of the linearized dynamics at the fixed point
        # This function returns A and B matrix of the discrete time dynamics
        A_c, B_c = self.continuous_time_linearized_dynamics()
        A_d = A_c  # There is no A matrix; just using identity from continuous dynamics here
        B_d = B_c * T

        return A_d, B_d

    def add_initial_state_constraint(self, prog, x, x_current):
        # TODO: verify you can use the same constraint as quad for initial state
        for i in range(len(x_current)):
            prog.AddBoundingBoxConstraint(x_current[i], x_current[i], x[0][i])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # constrain left and right velocities
        for k in range(N - 1):
            for wheel in range(2):
                prog.AddBoundingBoxConstraint(self.umin, self.umax, u[k][wheel])

    def add_dynamics_constraint(self, prog, x, u, N, T):
        # TODO: Update to use iLQR
        A = SkidSteerSystem.get_kinematics()
        for k in range(N - 1):
            for i in range(len(x[k])):
                # From ed: x[k + 1] = x[k] + self.dt * (R_linearized[k] @ A @ u[k])
                # In our code it's more convenient to group like this:
                # x[k + 1] = x[k] + (self.dt * R_linearized[k]) @ A @ u[k]
                # discrete_time_linearized_dynamics(T) returns self.dt * R_linearized[k]
                A_d, B_d = self.discrete_time_linearized_dynamics(T)

                x_next = x[k] + (B_d @ u[k])
                prog.AddLinearEqualityConstraint(x_next[i] - x[k + 1][i], 0)

    def add_cost(self, prog, x, u, N):
        # TODO: verify you can use the same cost for skid steer simple model
        '''
        for k in range(N - 1):
            cost = x[k].T @ self.Q @ x[k] + u[k].T @ self.R @ u[k]
            prog.AddQuadraticCost(cost)
        '''
        for k in range(N - 1):
            state_cost = (x[k] - self.x_d).T @ self.Q @ (x[k] - self.x_d)
            control_cost = u[k].T @ self.R @ u[k]
            total_cost = state_cost + control_cost
            prog.AddQuadraticCost(total_cost)

    def compute_mpc_feedback(self, x_current, use_clf=False):
        """
        This function computes the MPC controller input u
        """
        # Parameters for the QP
        N = 10
        T = 0.1

        # Initialize mathematical program and declare decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 3), dtype="object")

        # TODO: Get iLQR feedback in here and pass iLQR x and u here?

        for i in range(N):
            x[i] = prog.NewContinuousVariables(3, "x_" + str(i))
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

        # Retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        u_0 = result.GetSolution(u[0])
        u_mpc = u_0 + self.u_d()

        return u_mpc

    def compute_lqr_feedback(self, x):
        """
        Infinite horizon LQR controller
        """
        A, B = self.continuous_time_linearized_dynamics()
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -inv(self.R) @ B.T @ S
        u = self.u_d() + K @ x

        return u
