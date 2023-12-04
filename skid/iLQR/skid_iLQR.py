import numpy as np
from numpy import ndarray, dtype, floating
from scipy.signal import cont2discrete
from typing import List, Tuple, Any
import skid.iLQR.sim_skid_iLQR as sim_skid_iLQR

from math import sin, cos


class skid_iLQR(object):

    def __init__(self, x_goal: np.ndarray, N: int, dt: float, Q: np.ndarray, R: np.ndarray, Qf: np.ndarray):
        """
        Constructor for the iLQR solver
        :param N: iLQR horizon
        :param dt: timestep
        :param Q: weights for running cost on state
        :param R: weights for running cost on input
        :param Qf: weights for terminal cost on input
        """
        # starting position
        self.m = 1

        # Skid steer dynamics parameters
        self.alpha_l = 1
        self.alpha_r = 9
        self.x_ICR_l = -2
        self.x_ICR_r = 2
        self.y_ICR_v = -0.08

        # State and input dimensions
        self.nx = 3
        self.nu = 2

        # iLQR constants
        self.N = N
        self.dt = dt

        # Solver parameters
        self.alpha = 1
        self.max_iter = 1e3
        self.tol = 1e-4

        # target state
        self.x_goal = x_goal
        self.u_goal = 0.5 * 9.81 * np.ones((2,))

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def rotate(self, x):
        theta = x[2]
        R = np.array([[-sin(theta), -cos(theta), 0],
                      [cos(theta), sin(theta), 0],
                      [0, 0, 1]])

        return R

    def get_kinematics(self):
        # Compute A matrix based on provided dynamics (paper)
        '''
        A = 1 / (self.x_ICR_r - self.x_ICR_l) * np.array([
            [-self.y_ICR_v * self.alpha_l, self.y_ICR_v * self.alpha_r],
            [self.x_ICR_r * self.alpha_l, -self.x_ICR_l * self.alpha_r],
            [-self.alpha_l, self.alpha_r]
        ])
        '''

        A = np.array([[-self.y_ICR_v * self.alpha_l, self.y_ICR_v * self.alpha_r],
                      [self.x_ICR_r * self.alpha_l, -self.x_ICR_l * self.alpha_r],
                      [-self.alpha_l, self.alpha_r]])

        return A

    # TODO: Add conversion to global calc
    def input_to_local_coord(self, u: np.ndarray) -> np.ndarray:
        # u is a 2x1 array
        # multiply by u by A matrix (from paper) to get local coord
        # print(f"u.shape: {u.shape}")

        local_u = self.get_kinematics() @ u

        return local_u

    def local_to_global(self, local_u: np.ndarray) -> np.ndarray:
        # local coord is a  3x1
        # multiply local coord by rotation matrix (R) to get global coord

        omega = local_u[1]

        global_u = np.zeros(3)
        global_u[0] = cos(omega) * local_u[0] - sin(omega) * local_u[1]
        global_u[1] = sin(omega) * local_u[0] - cos(omega) * local_u[1]
        global_u[2] = omega

        return global_u

    def get_linearized_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: skid steer state
        :param u: input
        :return: A and B, the linearized continuous skid steer dynamics about some state x
        """
        # Set up for A matrix: There is no A matrix in simple skid-steer system;
        # Use I with NxN dim where N = len(xdot)
        A = np.eye(3)

        A_kinematics = self.get_kinematics()
        R = self.rotate(x)

        B = R @ A_kinematics
        print(f"B: {B}")
        '''
        print(f"A.shape from get_linearized_dynamics: {A_kinematics.shape}")
        print(f"R.shape from get_linearized_dynamics: {R.shape}")
        print(f"B.shape from get_linearized_dynamics: {B.shape}")
        '''
        return A, B

    def get_linearized_discrete_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: state
        :param u: input
        :return: the discrete linearized dynamics matrices, A, B as a tuple
        """
        A, B = self.get_linearized_dynamics(x, u)

        # Euler's method for state transition
        # Linearize about the current state
        # xdot = f(x,u) = x[k + 1] = x[k] + (self.dt * R_linearized) @ B @ u[k]
        # Where R_linearized is the rotation matrix
        # R_linearized is what is returned by get_linearized_dynamics I think?
        Ad = A
        Bd = self.dt * B
        # TODO: can we convert B to a 2x2 or something here?

        return Ad, Bd

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])

    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> float:
        """
        :param xk: state
        :param uk: input
        :return: l(xk, uk), the running cost incurred by xk, uk
        """
        # Standard LQR cost on the goal state
        lqr_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                          (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))

        return lqr_cost

    def grad_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: [∂l/∂xᵀ, ∂l/∂uᵀ]ᵀ (this is just a writing convention), evaluated at xk, uk
        """
        grad = np.zeros((8,))

        # TODO: Compute the gradient
        # get the running cost (don't actually need this, but good to look at)
        running_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                              (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))

        # calculate gradient with respect to x
        # gradient of x should be 3 because fx is length 3 (dimensions of the skid steer system)
        grad_running_cost_x = (xk - self.x_goal).T @ self.Q

        # calculate gradient with respect to  u
        grad_running_cost_u = (uk - self.u_goal).T @ self.R

        # combine arrays for gradient vector
        grad = np.hstack((grad_running_cost_x, grad_running_cost_u))

        return grad

    def hess_running_cost(self, xk: np.ndarray, uk: np.ndarray) -> np.ndarray:
        """
        :param xk: state
        :param uk: input
        :return: The hessian of the running cost
        [[∂²l/∂x², ∂²l/∂x∂u],
         [∂²l/∂u∂x, ∂²l/∂u²]], evaluated at xk, uk
        """
        H = np.zeros((self.nx + self.nu, self.nx + self.nu))
        # TODO: Compute the hessian
        '''
        print(f"H.shape: {H.shape}")
        print(f"xk.shape: {xk}")
        print(f"uk.shape: {uk.shape}")
        print(f"self.x_goal.shape: {self.x_goal.shape}")
        print(f"self.u_goal.shape: {self.u_goal.shape}")
        '''
        # Translate u_goal into global coord
        u_goal_local = self.input_to_local_coord(self.u_goal)
        u_goal_global = self.local_to_global(u_goal_local)

        # get the running cost
        running_cost = 0.5 * ((xk - self.x_goal).T @ self.Q @ (xk - self.x_goal) +
                              (uk - self.u_goal).T @ self.R @ (uk - self.u_goal))

        # find ∂²l/∂x² (use running cost func and calc by hand)
        H[:self.nx, :self.nx] = self.Q  # the second derivative in the x direction is just Q

        # find ∂²l/∂u² (use running cost func and calc by hand)
        H[self.nx:, self.nx:] = self.R

        return H

    def terminal_cost(self, xf: np.ndarray) -> ndarray[Any, dtype[floating[Any]]]:
        """
        :param xf: state
        :return: Lf(xf), the running cost incurred by xf
        """
        return 0.5 * (xf - self.x_goal).T @ self.Qf @ (xf - self.x_goal)

    def grad_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂Lf/∂xf
        """
        # TODO: Compute the gradient
        # differentiate ∂Lf/∂xf with respect to x
        grad = (xf - self.x_goal).T @ self.Qf

        return grad

    def hess_terminal_cost(self, xf: np.ndarray) -> np.ndarray:
        """
        :param xf: final state
        :return: ∂²Lf/∂xf²
        """
        # TODO: Compute H
        # differentiate ∂²Lf/∂xf² with respect to x
        H = self.Qf

        return H

    # TODO: Fix forward pass; we think this is returning 0 and 0 for x and y every step
    def forward_pass(self, xx: List[np.ndarray], uu: List[np.ndarray], dd: List[np.ndarray], KK: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: list of states, should be length N
        :param uu: list of inputs, should be length N-1
        :param dd: list of "feed-forward" components of iLQR update, should be length N-1
        :param KK: list of "Feedback" LQR gain components of iLQR update, should be length N-1
        :return: A tuple (xx, uu) containing the updated state and input
                 trajectories after applying the iLQR forward pass
        """

        xtraj = [np.zeros((self.nx,))] * self.N
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        # TODO: compute forward pass
        for k in range(self.N - 1):
            utraj[k] = uu[k] + (KK[k] @ (xtraj[k] - xx[k])) + self.alpha * dd[k]
            # use quad_sim to simulate
            xtraj[k + 1] = sim_skid_iLQR.F(xtraj[k], utraj[k], self.dt)

        #print(f"xtraj: {xtraj}, utraj: {utraj}")

        return xtraj, utraj

    def backward_pass(self, xx: List[np.ndarray], uu: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :param xx: state trajectory guess, should be length N
        :param uu: input trajectory guess, should be length N-1
        :return: KK and dd, the feedback and feedforward components of the iLQR update
        """
        dd = [np.zeros((self.nu,))] * (self.N - 1)  # dk
        KK = [np.zeros((self.nu, self.nx))] * (self.N - 1)

        # TODO: compute backward pass
        Hk = self.hess_terminal_cost(xx[self.N - 1])
        gk = self.grad_terminal_cost(xx[self.N - 1])

        for k in range(self.N - 2, -1, -1):
            # TODO: convert uu to 3x1 global coord
            local_uk = self.input_to_local_coord(uu[k])
            global_uk = self.local_to_global(local_uk)

            # get the linearized dynamics
            A, B = self.get_linearized_discrete_dynamics(xx[k], global_uk)

            # apply hessian running cost
            Qk = self.hess_running_cost(xx[k], uu[k])

            lx = self.grad_running_cost(xx[k], uu[k])[:self.nx]
            lxx = self.hess_running_cost(xx[k], uu[k])[:self.nx, :self.nx]
            lu = self.grad_running_cost(xx[k], uu[k])[self.nx:]
            luu = self.hess_running_cost(xx[k], uu[k])[self.nx:, self.nx:]
            lux = self.hess_running_cost(xx[k], uu[k])[self.nx:, :self.nx]

            # write expansion coefficients
            Qx = lx + A.T @ gk
            Qu = lu + B.T @ gk  # TODO: lu is 2x1 and B is 3x1; when should I convert u?
            Qxx = lxx + A.T @ Hk @ A
            Quu = luu + B.T @ Hk @ B
            Qux = lux + B.T @ Hk @ A

            Kk = -np.linalg.inv(Quu) @ Qux
            dk = -np.linalg.inv(Quu) @ Qu

            KK[k] = Kk
            dd[k] = dk

            gk = Qx - Kk.T @ Quu @ dk
            Hk = Qxx - Kk.T @ Quu @ Kk

        return dd, KK

    def calculate_optimal_trajectory(self, x: np.ndarray, uu_guess: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:

        """
        Calculate the optimal trajectory using iLQR from a given initial condition x,
        with an initial input sequence guess uu
        :param x: initial state
        :param uu_guess: initial guess at input trajectory
        :return: xx, uu, KK, the input and state trajectory and associated sequence of LQR gains
        """
        assert (len(uu_guess) == self.N - 1)

        # Get an initial, dynamically consistent guess for xx by simulating the skid steer
        xx = [x]
        for k in range(self.N - 1):
            xx.append(sim_skid_iLQR.F(xx[k], uu_guess[k], self.dt))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        print(f'cost: {Jnext}')
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)
            print(f"xx while: {xx}")
            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            print(f'cost: {Jnext}')
            i += 1
        print(f'Converged to cost {Jnext}')
        return xx, uu, KK
