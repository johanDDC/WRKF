import numpy as np
from scipy.linalg import lu_solve, lu_factor, cho_solve, cho_factor

norm = np.linalg.norm


def retraction(x, v):
    res = x + v
    return res / norm(res)


orthoprojector = lambda x, v: v - (x @ v) * x


def armijo_backtracking(func, x, grad, mul=1e-4, beta=0.8, alpha=1):
    iters = 0
    while func(x) - func(retraction(x, -alpha * grad)) < \
            mul * alpha * norm(grad) ** 2:
        alpha *= beta
        if iters >= 40:
            break
        iters += 1
    return alpha


def eigenpair(A, mode="max", tol=1e-6, maxiter=100):
    x = np.random.normal(0, 2 / (A.shape[0] + 1), size=A.shape[0])
    alpha = 1
    for i in range(maxiter):
        Ax = A @ x
        grad = orthoprojector(x, 2 * Ax)
        alpha = armijo_backtracking(lambda x: x @ Ax, x, grad, x @ (A @ grad) / (x @ Ax))
        x = retraction(x, alpha * grad)
        if norm(grad) <= tol:
            break

    return x, (A @ x) @ x


def bisection(S, D, p, tol=1e-6, maxiter=100):
    """
    :param S: currently estimated covariance matrix of the process
    :param D: gradiend of f(S)
    :param p: Wasserstein radius
    :param tol: tolerance
    :param maxiter: maxiter
    :return: linear approfimation of f(S)
    """
    def h(inv):
        I = np.eye(D.shape[0])
        mat2 = I - inv
        mat2 = mat2 @ mat2
        return p ** 2 - (S * mat2).sum()

    l1, v1 = np.linalg.eigh(D)
    v, l = v1[:, -1], l1[-1]
    LB = l * (1 + np.sqrt(S @ v @ v) / p)
    UB = l * (1 + np.sqrt(np.trace(S)) / p)
    I = np.eye(D.shape[0])
    inv_D = lambda g: v1 @ np.diag(1 / (g - l1)) @ v1.T
    L = None
    for i in range(maxiter):
        gamma = (LB + UB) / 2
        inv = gamma * inv_D(gamma)
        L = inv @ S @ inv
        h_g = h(inv)
        if h_g < 0:
            LB = gamma
        else:
            UB = gamma
        delta = gamma * (p ** 2 - np.trace(S)) - (D * L).sum() + gamma * (S * inv).sum()
        if (h_g >= 0 and delta < tol) or UB - LB < 1e-3:
            break
    return L


def Frank_Wolfe(S, p, n, m, tol=1e-4, maxiter=100):
    """
    :param S: convariance matrix of the process
    :param p: Wasserstein radius
    :param n: x_dim
    :param m: y_dim
    :param tol: tolerance
    :param maxiter: maxiter
    :return: new estimated covariance matrix of the process
    """
    sigma = np.min(np.linalg.eigvals(S))
    sigma_bar = (p + np.sqrt(np.trace(S))) ** 2
    C_bar = 2 * sigma_bar ** 4 / sigma ** 3
    S_k = S
    I_m = np.eye(m)
    I_n = np.eye(n)
    for i in range(1, maxiter + 1):
        alpha = 2 / (i + 2)
        S_yy_inv = S_k[n:, n:]
        S_yy_inv = cho_solve(cho_factor(S_yy_inv), I_m)
        G = S_k[:n, n:] @ S_yy_inv
        residual_prev = np.trace(S_k[:n, :n] - G @ S_k[n:, :n])
        D = np.concatenate([I_n, -G], axis=1)
        D = D.T @ D
        L = bisection(S_k, D, p, alpha * tol * C_bar)
        residual = np.abs(((L - S_k) * D).sum())
        if residual / residual_prev < tol:
            break
        S_k = (1 - alpha) * S_k + alpha * L
    return S_k, G
