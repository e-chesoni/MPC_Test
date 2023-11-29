import numpy as np

# Assuming A_c is a 3x2 matrix and T is the time step
A_c = np.array([[1, 2],
                [3, 4],
                [5, 6]])

T = 0.1  # Example time step

# Create a 3x3 matrix with a 2x2 identity-like block
identity_like = np.eye(2)

# Construct a 3x3 matrix with the identity-like block
#A_d = np.eye(3) + A_c * T
A_d = np.block([[np.eye(3), A_c * T]])
print (A_d)

# A_d now represents the discretized dynamics

A = np.zeros((6, 6))
A[:3, -3:] = np.identity(3)
print(A)

A_d1 = np.identity(6) + A * T
print(A_d1)