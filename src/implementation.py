import numpy as np
from utils import *
from scipy.linalg import cho_factor, cho_solve


class BaseFilter:
    def __init__(self, A, B, C, D, x_0=None, S_0=None):
        """
        Creates Kalman Filter.

        :param A: process matrix
        :param B: noise matrix for x
        :param C: observation matrix
        :param D: noise matrix for y
        :param x_0: initial state
        :param S_0: initial covariance matrix
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n = A.shape[0]
        self.m = C.shape[0]
        self.d = self.n + self.m
        if x_0 is None:
            self.x = np.random.normal(0, 2 / (self.n + 1), self.n)
        else:
            self.x = np.copy(x_0)
        if S_0 is None:
            self.S = np.random.normal(0, 1 / self.n, (self.n, self.n))
            self.S = self.S.T @ self.S
        else:
            self.S = np.copy(S_0)

    def fit(self, y):
        """
        Filter estimates distribution for next state prediction.

        :param y: current observation
        :return: self
        """
        raise NotImplementedError

    def predict(self, tol=1e-6):
        """
        Predicts state of the system in current time step.
        Warning: you can predict state only after fiting filter, and you
        can do it only ones. After prediction the last observation will be foggoten

        :param tol: tolerance for Frank-Wolfe algorythm.
        :return: current state
        """
        raise NotImplementedError

    def evolution(self, Y):
        """
        Starts evolution of process and predicts state for each observation.

        :param Y: matrix of all observation of the process
        :return: matrix of all states of the process
        """
        xs = []
        for i in range(Y.shape[0]):
            self.fit(Y[i, :])
            x_hat = self.predict()
            xs.append(x_hat)

        return np.array(xs)


class WRKF(BaseFilter):
    def __init__(self, A, B, C, D, x_0=None, S_0=None, p=None):
        super(WRKF, self).__init__(A, B, C, D, x_0, S_0)
        self.p = np.sqrt(self.d) if p is None else p

    def fit(self, y):
        mat1 = np.concatenate([self.A, self.C @ self.A], axis=0)
        mat2 = np.concatenate([self.B, self.C @ self.B + self.D], axis=0)
        mu = mat1 @ self.x
        Sigma = mat1 @ self.S @ mat1.T + mat2 @ mat2.T
        self.nominal_dist = (mu, Sigma)
        self.y = y
        return self

    def predict(self, tol=1e-6):
        mu, Sigma = self.nominal_dist
        S_star, G = Frank_Wolfe(Sigma, self.p, self.n, self.m, tol)
        self.S = S_star[:self.n, :self.n] - G @ S_star[self.n:, :self.n]
        self.x = G @ (self.y - mu[self.n:]) + mu[:self.n]
        del self.y
        return self.x


class NaiveKalman(BaseFilter):
    def __init__(self, A, B, C, D, x_0=None, S_0=None, p=None):
        super(NaiveKalman, self).__init__(A, B, C, D, x_0, S_0)

    def fit(self, y):
        mat1 = np.concatenate([self.A, self.C @ self.A], axis=0)
        mat2 = np.concatenate([self.B, self.C @ self.B + self.D], axis=0)
        mu = mat1 @ self.x
        Sigma = mat1 @ self.S @ mat1.T + mat2 @ mat2.T
        self.nominal_dist = (mu, Sigma)
        self.y = y
        return self

    def predict(self, tol=1e-6):
        mu, Sigma = self.nominal_dist
        I_m = np.eye(self.m)
        S_yy_inv = Sigma[self.n:, self.n:]
        S_yy_inv = cho_solve(cho_factor(S_yy_inv), I_m)
        G = Sigma[:self.n, self.n:] @ S_yy_inv
        self.S = Sigma[:self.n, :self.n] - G @ Sigma[self.n:, :self.n]
        self.x = G @ (self.y - mu[self.n:]) + mu[:self.n]
        del self.y
        return self.x