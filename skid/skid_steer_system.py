import numpy as np
from math import sin, cos


class SkidSteerSystem:
    def __int__(self):
        pass

    @staticmethod
    def get_kinematics():
        alpha_l = 0.9464
        alpha_r = 0.9253
        x_ICR_l = -0.2758
        x_ICR_r = 0.2998
        y_ICR_v = -0.0080

        # Compute A matrix based on provided dynamics (paper)

        A = np.array([[-y_ICR_v * alpha_l, y_ICR_v * alpha_r],
                     [x_ICR_r * alpha_l, -x_ICR_l * alpha_r],
                     [-alpha_l, alpha_r]])

        # Apply scaling factor to A (as done in paper)
        scale_factor = 1 / (x_ICR_r - x_ICR_l)
        A = scale_factor * A

        return A

    @staticmethod
    def rotate(x: np.ndarray):
        R = np.array([[cos(x[2]), -sin(x[2]), 0],
                      [sin(x[2]), cos(x[2]), 0],
                      [0, 0, 1]])

        return R
