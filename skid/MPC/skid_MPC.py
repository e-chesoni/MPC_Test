import numpy as np
from typing import List, Tuple, Any
import matplotlib.pyplot as plt
from numpy import ndarray, dtype
from numpy.linalg import inv
from numpy.linalg import cholesky
from math import sin, cos
import math
from scipy.interpolate import interp1d
from scipy.integrate import ode
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.linalg import solve_continuous_are

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver, SolverOptions
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

from skid.iLQR.skid_iLQR import skid_iLQR
from skid.skid_state_calc import SkidSteerCalculateState
from skid.skid_steer_system import SkidSteerSystem


class SkidMPC(object):
    def __init__(self, start, end, u_guess, N, dt, Q, R, Qf):
        print("Init SkidSteerVehicle...")

        self.start = start
        self.end = end
        self.u_guess = u_guess

        self.N = N  # TODO: NOTE -- using different N to calculate iLQR and MPC
        self.dt = dt

        self.Q = Q
        self.R = R
        self.Qf = Qf

        # Input limits
        self.umin = -5  # if this is 0, turns in a circle
        self.umax = 5

        # Initialize and call iLQR
        self.dt_step = 0
        self.ilqr = skid_iLQR(self.end, self.N, self.dt, self.Q, self.R, self.Qf)
        self.x_sol, self.u_sol, self.K_sol = self.ilqr.calculate_optimal_trajectory(self.start, self.u_guess)

        # Modified x_sol for straight line
        x_sol_straight = [np.array([0., 0.01, 0.]),
                          np.array([0.001, 0.01, 0.]),
                          np.array([0.002, 0.01, 0.]),
                          np.array([0.003, 0.01, 0.]),
                          np.array([0.004, 0.01, 0.]),
                          np.array([0.005, 0.01, 0.]),
                          np.array([0.006, 0.01, 0.]),
                          np.array([0.007, 0.01, 0.]),
                          np.array([0.008, 0.01, 0.]),
                          np.array([0.009, 0.01, 0.])]

        # Modified u_sol for straight line
        u_sol_straight = [np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421]),
                          np.array([-1.61875726, 1.48081421])]

        u_sol_straight = [np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0]),
                          np.array([1.0, -1.0])]

        #self.x_sol = x_sol_straight
        #self.u_sol = u_sol_straight

        print(f"x_sol: {self.x_sol}")
        print(f"u_sol: {self.u_sol}")

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
        sdot = SkidSteerCalculateState.f(x, u)

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
        for i in range(len(x_current)):
            prog.AddBoundingBoxConstraint(x_current[i], x_current[i], x[0][i])

    def add_input_saturation_constraint(self, prog, x, u, N):
        # constrain left and right velocities
        for k in range(N - 1):
            for wheel in range(2):
                prog.AddBoundingBoxConstraint(self.umin, self.umax, u[k][wheel])

    def add_dynamics_constraint(self, prog, x, u, N, T):
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

    def add_ilqr_constraints(self, prog, x, u):
        slack_tol = 0.7  # tolerance

        for k in range(len(self.u_sol)):
            for i in range(len(self.x_sol[k])):
                # Add a constraint to enforce equality between predicted and actual state
                #prog.AddLinearEqualityConstraint(self.x_sol[k + 1][i] - x[k + 1][i], 0)

                # Add a bounding box constraint to enforce closeness between predicted and actual state
                x_lb = self.x_sol[k + 1][i] - slack_tol
                x_ub = self.x_sol[k + 1][i] + slack_tol
                prog.AddBoundingBoxConstraint(x_lb, x_ub, x[k + 1][i])

        for k in range(len(self.u_sol)):
            for j in range(len(self.u_sol[k])):
                # Add a constraint to enforce equality between iLQR and actual control input
                #prog.AddLinearEqualityConstraint(u_ilqr[k][j] - u[k][j], 0)

                # Add a bounding box constraint to enforce closeness between iLQR and actual control input
                u_lb = self.u_sol[k][j] - slack_tol
                u_ub = self.u_sol[k][j] + slack_tol
                prog.AddBoundingBoxConstraint( u_lb, u_ub, u[k][j])

    def add_cost(self, prog, x, u, N):
        '''
        for k in range(N - 1):
            cost = x[k].T @ self.Q @ x[k] + u[k].T @ self.R @ u[k]
            prog.AddQuadraticCost(cost)
        '''
        # TODO: Try original cost
        for k in range(N - 1):
            state_cost = (x[k] - self.x_sol[k]).T @ self.Q @ (x[k] - self.x_sol[k])
            control_cost = u[k].T @ self.R @ u[k]
            total_cost = state_cost + control_cost
            prog.AddQuadraticCost(total_cost)

    # HW5 osc.py SetupAndSolveQP
    def SetupAndSolveILQR(self, x_current):
        # Initialize mathematical program and declare decision variables
        prog = MathematicalProgram()

        x = np.zeros((self.N, 3), dtype="object")
        for i in range(self.N):
            x[i] = prog.NewContinuousVariables(3, "x_" + str(i))

        u = np.zeros((self.N - 1, 2), dtype="object")
        for i in range(self.N - 1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        #self.add_input_saturation_constraint(prog, x, u, self.N)  # PROBLEMATIC CONSTRAINTS HERE

        # Constrain dynamics (ONLY USE ONE OF THESE)
        #self.add_dynamics_constraint(prog, x, u, self.N, self.dt)  # Works, but vehicle is very large
        # Add iLQR constraint
        self.add_ilqr_constraints(prog, x, u)

        # Add cost constraint
        self.add_cost(prog, x, u, self.N)

        # Solve the QP
        solver = OsqpSolver()
        result = solver.Solve(prog)

        if not result.is_success():
            print("Solver did not converge successfully.")

        return u, result

    def compute_mpc_feedback(self, x_current, use_clf=False):
        """
        This function computes the MPC controller input u
        """

        u, result = self.SetupAndSolveILQR(x_current)

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
