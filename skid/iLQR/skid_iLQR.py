import numpy as np
from numpy import ndarray, dtype, floating
from typing import List, Tuple, Any
from skid.skid_steer_simulator import Skid_Steer_Simulator
from skid.skid_steer_system import SkidSteerSystem


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
        #self.m = 1

        # Simulator
        self.sim_skid_iLQR = Skid_Steer_Simulator()

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
        self.u_goal = np.zeros((2,))

        # Cost terms
        self.Q = Q
        self.R = R
        self.Qf = Qf

    def total_cost(self, xx, uu):
        J = sum([self.running_cost(xx[k], uu[k]) for k in range(self.N - 1)])
        return J + self.terminal_cost(xx[-1])

    def get_linearized_dynamics(self, x: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: skid steer state
        :param u: input
        :return: A and B, the linearized continuous skid steer dynamics about some state x
        """
        # Set up for A matrix: There is no A matrix in simple skid-steer system;
        # Use I with NxN dim where N = len(xdot)
        A = np.eye(3)

        A_kinematics = SkidSteerSystem.get_kinematics()
        R = SkidSteerSystem.rotate(x)

        B = R @ A_kinematics

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

        return Ad, Bd

    def running_cost(self, xk: np.ndarray, uk: np.ndarray) -> ndarray[Any, dtype[floating[Any]]]:
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

        # find ∂²l/∂x² (use running cost func and calc by hand)
        H[:self.nx, :self.nx] = self.Q  # the second derivative in the x direction is just Q

        # find ∂²l/∂u² (use running cost func and calc by hand)
        H[self.nx:, self.nx:] = self.R

        # populate H with the second order derivatives

        return H

    def terminal_cost(self, xf: np.ndarray) -> float:
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
        H = np.zeros((self.nx, self.nx))

        # TODO: Compute H
        # differentiate ∂²Lf/∂xf² with respect to x
        H = self.Qf

        return H

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

        xtraj = [np.zeros((self.nx,))] * self.N  # was np.zeros
        utraj = [np.zeros((self.nu,))] * (self.N - 1)
        xtraj[0] = xx[0]

        # TODO: compute forward pass
        for k in range(self.N - 1):
            utraj[k] = uu[k] + (KK[k] @ (xtraj[k] - xx[k])) + self.alpha * dd[k]
            # use sim_skid_iLQR to simulate
            xtraj[k + 1] = self.sim_skid_iLQR.F(xtraj[k], utraj[k], self.dt)

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
            # get the linearized dynamics
            A, B = self.get_linearized_discrete_dynamics(xx[k], uu[k])

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
        assert (uu_guess[0].shape[0] == 2)

        # Get an initial, dynamically consistent guess for xx by simulating the skid steer
        xx = [x]

        for k in range(self.N - 1):
            xx.append(self.sim_skid_iLQR.F(xx[k], uu_guess[k], self.dt))

        Jprev = np.inf
        Jnext = self.total_cost(xx, uu_guess)
        uu = uu_guess
        KK = None

        i = 0
        # print(f'cost: {Jnext}')
        while np.abs(Jprev - Jnext) > self.tol and i < self.max_iter:
            dd, KK = self.backward_pass(xx, uu)
            xx, uu = self.forward_pass(xx, uu, dd, KK)

            Jprev = Jnext
            Jnext = self.total_cost(xx, uu)
            i += 1

        print(f'Converged to cost {Jnext}')

        return xx, uu, KK
