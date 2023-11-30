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
    def __init__(self, Q, R, Qf):
        print("Init SkidSteerVehicle...")
        # TODO: Kinematic model parameters to tweak
        '''
        self.alpha_l = 0.9464
        self.alpha_r = 0.9253
        self.x_ICR_l = -0.2758
        self.x_ICR_r = 0.2998
        self.y_ICR_v = -0.0080
        '''
        self.alpha_l = 1
        self.alpha_r = 9
        self.x_ICR_l = -2
        self.x_ICR_r = 2
        self.y_ICR_v = -0.08

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

    def x_d(self):
        # Nominal state
        return np.array([0, 0, 0])

    def u_d(self):
        # Nominal input
        return np.array([0, 0])

    # Continuous-time dynamics for the skid-steer vehicle
    def continuous_time_full_dynamics(self, x, u):
        # TODO: verify continuous time dynamics calculation for skid steer
        #print(f"u shape: {u.shape}")
        # Get state variables
        v_x, v_y, w_z = x[0], x[1], x[2]
        V_l, V_r = u[0], u[1]

        # Parameters from the kinematic model
        x_ICR_v = -v_y / w_z
        x_ICR_l = (self.alpha_l * V_l - v_y) / w_z
        x_ICR_r = (self.alpha_r * V_r - v_y) / w_z
        y_ICR_v = y_ICR_l = y_ICR_r = v_x / w_z

        # Calculate continuous-time dynamics for global system
        xdot = v_x * cos(x[2]) - v_y * sin(x[2])
        ydot = v_x * sin(x[2]) + v_y * cos(x[2])
        wdot = (V_r - V_l) / (x_ICR_r - x_ICR_l) * (-self.alpha_l + self.alpha_r) # angular rate between wheels

        # TODO: Verify this is not needed; state vector = position vector = 3x1
        # Update the state derivatives
        #v_xdot = -v_y / (x_ICR_r - x_ICR_l) * (-y_ICR_v * self.alpha_l * V_l + y_ICR_v * self.alpha_r * V_r)
        #v_ydot = -v_y / (x_ICR_r - x_ICR_l) * (x_ICR_r * self.alpha_l * V_l - x_ICR_l * self.alpha_r * V_r)
        #w_zdot = -v_y / (x_ICR_r - x_ICR_l) * (-self.alpha_l * V_l + self.alpha_r * V_r)

        fxu = np.array([xdot, ydot, wdot])
        return fxu

    # CALLED A in PAPER
    def get_kinematics(self, x, u):
        v_x, v_y, w_z = x[0], x[1], x[2]
        V_l, V_r = u[0], u[1]

        # Compute A matrix based on provided dynamics (paper)
        x_ICR_v = -v_y / w_z
        x_ICR_l = (self.alpha_l * V_l - v_y) / w_z
        x_ICR_r = (self.alpha_r * V_r - v_y) / w_z
        y_ICR_v = y_ICR_l = y_ICR_r = v_x / w_z

        A = 1 / (x_ICR_r - x_ICR_l) * np.array([
            [-y_ICR_v * self.alpha_l, y_ICR_v * self.alpha_r],
            [x_ICR_r * self.alpha_l, -x_ICR_l * self.alpha_r],
            [-self.alpha_l, self.alpha_r]
        ])

        return A

    # Helper to compute A matrix
    # TODO: There is no A matrix in simple dynamics for skid steer
    def compute_A_matrix(self):
        A = np.zeros((3, 2))

        return A

    # Linearized dynamics for Skid Steer Vehicle
    def continuous_time_linearized_dynamics(self):
        # Dynamics linearized at the fixed point
        # This function returns A and B matrix

        # There is no A matrix in simple skid-steer system;
        # Use I with NxN dim where N = len(xdot)
        A = np.eye(3)

        B = np.zeros((3, 3))
        # TODO: get const parameters for B matrix from self
        # B = self.compute_B_matrix()
        theta0 = self.x_d()[2]
        # 1st row
        B[0, 0] = -sin(theta0)
        B[0, 1] = cos(theta0)
        # 2nd row
        B[1, 0] = cos(theta0)
        B[1, 1] = sin(theta0)
        # 3rd row
        B[2, 2] = 1

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
        # TODO: verify you can use the same constraint as quadrotor for initial state
        for i in range(len(x_current)):
            prog.AddBoundingBoxConstraint(x_current[i], x_current[i], x[0][i])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # constrain left and right velocities
        for k in range(N - 1):
            for wheel in range(2):  # Two wheels: left (0) and right (1)
                prog.AddBoundingBoxConstraint(self.umin, self.umax, u[k][wheel])
        # TODO: how do we constrain vehicle orientation?

        # constrains difference in wheel velocities to be 0
        # TODO: Make this a cost
        '''
        for k in range(N - 1):
            prog.AddLinearEqualityConstraint(u[k][0] - u[k][1], 0.0)
        '''

    def add_dynamics_constraint(self, prog, x, u, N, T):
        A = self.get_kinematics(x, u)  # <--this won't work
        for k in range(N - 1):
            for i in range(len(x[k])):
                x_next = A @ u[k]
                prog.AddLinearEqualityConstraint(x_next[i] - x[k + 1][i], 0)

    def add_dynamics_constraint_TEST(self, prog, x, u, N, T):
        # TODO: update dynamics constraint.
        # Use AddLinearEqualityConstraint(expr, value)

        for k in range(N - 1):
            x_current = x[k]
            u_current = u[k]
            x_next = x[k + 1]

            # Compute the continuous-time dynamics for the given x and u
            fxu = self.continuous_time_full_dynamics(x_current, u_current)

            # Discretize the continuous-time dynamics using Taylor series expansion
            A, B = self.discrete_time_linearized_dynamics(T)

            # Apply the dynamics constraint
            for i in range(len(x_current)):
                x_dot = A @ x_current + B @ fxu
                prog.AddLinearEqualityConstraint(x_next[i] - x[k + 1][i], 0)

    def add_cost(self, prog, x, u, N):
        # TODO: verify you can use the same cost for skid steer simple model
        '''
        for k in range(N - 1):
            cost = x[k].T @ self.Q @ x[k] + u[k].T @ self.R @ u[k]
            prog.AddQuadraticCost(cost)
        '''
        for k in range(N - 1):
            # State cost: (x - x_d).T @ Q @ (x - x_d)
            # Control cost: u.T @ R @ u
            state_cost = (x[k] - self.x_d()).T @ self.Q @ (x[k] - self.x_d())
            control_cost = u[k].T @ self.R @ u[k]
            total_cost = state_cost + control_cost
            prog.AddQuadraticCost(total_cost)

        # TODO: How to we tax bad orientation
        '''
        for k in range(N - 1):
            cost = self.Q * (u[k][0] - u[k][1]) ** 2
            prog.AddQuadraticCost(cost)
        '''
    def compute_mpc_feedback(self, x_current, use_clf=False):
        '''
        This function computes the MPC controller input u
        '''
        # Parameters for the QP
        N = 10
        T = 0.1

        # Initialize mathematical program and decalre decision variables
        prog = MathematicalProgram()
        x = np.zeros((N, 3), dtype="object")

        for i in range(N):
            x[i] = prog.NewContinuousVariables(3, "x_" + str(i))
        u = np.zeros((N - 1, 2), dtype="object")
        #print(f"u: {u[0]}")

        for i in range(N - 1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, x, u, N)
        #self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_cost(prog, x, u, N)

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)

        u_mpc = np.zeros(2)
        # Retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
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

    def compute_qp(self, x):
        pass

    def compute_non_linear_program(self, x):
        pass