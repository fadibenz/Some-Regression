import numpy as np

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def calculate_next_iteration(X, Y, W):
    S = sigmoid(np.dot(X, W))
    Omega = np.diag(S) @ np.diag(1-S)
    hessian = X.T @ (Omega @ X)
    gradient = X.T @ (Y - S)
    e = np.linalg.solve(hessian, gradient)
    return e, S


X = np.array([
    [0.2, 3.1, 1],
    [1.0, 3.0, 1],
    [-0.2, 1.2, 1],
    [1.0, 1.1, 1]]
)

Y = [1, 1, 0, 0]

W = [-1, 1, 0]


for i in range(2):
    e, S = calculate_next_iteration(X, Y, W)
    W = W + e
    print(f'sigmoid {i}:', S)
    print(f'Weight vector {i}:', W)

