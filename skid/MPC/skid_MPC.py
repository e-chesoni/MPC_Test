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

from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver
import pydrake.symbolic as sym

from pydrake.all import MonomialBasis, OddDegreeMonomialBasis, Variables

from skid.iLQR.skid_iLQR import skid_iLQR
from skid.skid_steer_simulator import Skid_Steer_Simulator
from skid.skid_steer_system import SkidSteerSystem


class SkidMPC(object):
    def __init__(self, start, end, u_guess, N, dt, Q, R, Qf):
        print("Init SkidSteerVehicle...")

        self.start = start
        self.end = end
        self.u_guess = u_guess

        self.N = N
        self.dt = dt

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
        print("constraint types:")
        print(type(x))  # TODO: list in new solver; convert to np.array
        print(type(x_current))
        print(type(x[0]))

        print(f"x: {x.shape}")  # TODO: list in new solver; convert to np.array
        print(f"x_current.shape: {x_current.shape}")
        print(f"x[0]: {x[0]}")

        print("The first value in x_current and x[0] are:")
        print(f"x_current[0]: {x_current[0]}")
        print(f"x[0]: {x[0]}")

        for i in range(len(x_current)):
            print("in i loop...")
            print(f"x[0][i]: {x[0][i]}")
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

    def add_ilqr_constraints(self, prog, x, u, x_ilqr, u_ilqr):
        for k in range(len(u_ilqr)):
            for i in range(len(x_ilqr[k])):
                # Update the state prediction based on iLQR solution
                x_next_ilqr = x_ilqr[k + 1]

                # Add a constraint to enforce equality between predicted and actual state
                prog.AddLinearEqualityConstraint(x_next_ilqr[i] - x[k + 1][i], 0)

        for k in range(len(u_ilqr)):
            for j in range(len(u_ilqr[k])):
                # Add a constraint to enforce equality between iLQR and actual control input
                prog.AddLinearEqualityConstraint(u_ilqr[k][j] - u[k][j], 0)

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

    # TODO: modify osc SetupAndSolverQP()
    # TODO: remove 176 - 198 from osc.py SetupAndSolverQP()
    # HW5 osc.py SetupAndSolveQP
    def SetupAndSolveILQR(self, x_current) -> tuple[ndarray[Any, dtype[Any]], Any]:

        # First get the state, time, and fsm state
        # x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        # t = context.get_time()
        # fsm = get_fsm(t) # TODO: Probably don't need? What was fsm used for in osc

        # TODO: Get from iLQR
        ilqr = skid_iLQR(self.end, self.N, self.dt, self.Q, self.R, self.Qf)
        x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(self.start, self.u_guess)

        # Convert lists to arrays
        x_sol = np.array(x_sol)
        u_sol = np.array(u_sol)

        # Parameters for the QP
        # TODO: Use these or self.N and self.dt?
        N = 10
        T = 0.1  # 0.01 in iLQR dynamics

        # Initialize mathematical program and declare decision variables
        prog = MathematicalProgram()

        x = np.zeros((N, 3), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(3, "x_" + str(i))

        u = np.zeros((N - 1, 2), dtype="object")
        for i in range(N - 1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # TODO: Is this the right way to add constraints?
        # TODO: How do i add these constraints to variables returned by iLQR
        # TODO: use these or self.N and self.dt?
        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)  # x should be a decision variable, not a number
        self.add_input_saturation_constraint(prog, x, u, N)
        self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_cost(prog, x, u, N)

        # Solve the QP
        # TODO: Call custom solver
        solver = OsqpSolver()

        result = solver.Solve(prog)

        return u_sol, result

    def SolveOldMPC(self, x_current):
        N = 10
        T = 0.1

        # Initialize mathematical program and declare decision variables
        prog = MathematicalProgram()

        x = np.zeros((N, 3), dtype="object")
        for i in range(N):
            x[i] = prog.NewContinuousVariables(3, "x_" + str(i))

        u = np.zeros((N - 1, 2), dtype="object")
        for i in range(N - 1):
            u[i] = prog.NewContinuousVariables(2, "u_" + str(i))

        # Initialize and call iLQR
        ilqr = skid_iLQR(self.end, N, T, self.Q, self.R, self.Qf)
        x_sol, u_sol, K_sol = ilqr.calculate_optimal_trajectory(self.start, self.u_guess)

        # Add constraints and cost
        self.add_initial_state_constraint(prog, x, x_current)
        self.add_input_saturation_constraint(prog, x, u, N)

        # Constrain dynamics
        # Add iLQR-based constraints (instead of dynamics constraint)
        #self.add_dynamics_constraint(prog, x, u, N, T)
        self.add_ilqr_constraints(prog, x, u, x_sol, u_sol)

        # Add cost constraint
        self.add_cost(prog, x, u, N)

        # Solve the QP
        # TODO: Call custom solver
        solver = OsqpSolver()  # modify osc setupandsolverqp and call instead of this

        result = solver.Solve(prog)

        return u, result

    def compute_mpc_feedback(self, x_current, use_clf=False):
        """
        This function computes the MPC controller input u
        """
        # TODO: How do we use iLQR x_sol and u_sol instead of these?
        '''
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
        # TODO: Call custom solver
        #solver = OsqpSolver()  # modify osc setupandsolverqp and call instead of this
        # remove 176 - 198

        #result = solver.Solve(prog)
        '''
        u, result = self.SolveOldMPC(x_current)
        #u, result = self.SetupAndSolveILQR(x_current)

        # Retrieve the controller input from the solution of the optimization problem
        # and use it to compute the MPC input u
        u_0 = result.GetSolution(u[0])
        u_mpc = u_0 + self.u_d()

        return u_mpc

    # TODO: Only used when running iLQR
    def compute_lqr_feedback(self, x):
        """
        Infinite horizon LQR controller
        """
        A, B = self.continuous_time_linearized_dynamics()
        S = solve_continuous_are(A, B, self.Q, self.R)
        K = -inv(self.R) @ B.T @ S
        u = self.u_d() + K @ x

        return u
