import numpy as np
from scipy.spatial.transform import Rotation as R

np.set_printoptions(precision=5, suppress=True)

"""
TODO:
- Combine this with simulation class? Or best to keep them seperate?
"""


class ArmSE3:
    """
    ANGLES BEFORE POSITION in all algebra representations,
    including screws, adjoint, skew, etc
    """

    def __init__(self, screws, zero_config) -> None:
        self.screws = screws
        self.zero_config = zero_config
        self.N = screws.shape[0]

    def _skew(self, w):
        if len(w) == 3:
            return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        else:
            return np.array(
                [
                    [0, -w[2], w[1], w[3]],
                    [w[2], 0, -w[0], w[4]],
                    [-w[1], w[0], 0, w[5]],
                    [0, 0, 0, 0],
                ]
            )

    def _vec_se3(self, q):
        T = np.eye(4)
        T[:3, :3] = R.from_quat(q[3:]).as_matrix()
        T[:3, 3] = q[:3]
        return T

    def _se3_vec(self, T):
        vec = np.zeros(7)
        vec[:3] = T[:3, 3]
        vec[3:] = R.from_matrix(T[:3, :3]).as_quat()
        return vec

    def _exp(self, s):
        w = s[:3]
        u = s[3:]
        wx = self._skew(w)
        theta = np.linalg.norm(w)
        I = np.eye(3)  # noqa

        if theta < 1e-2:
            R = I + wx / 2 + wx @ wx / 6 + wx @ wx @ wx / 24
            V = I + wx / 2 + wx @ wx / 6 + wx @ wx @ wx / 24
        else:
            A = np.sin(theta) / theta
            B = (1 - np.cos(theta)) / theta**2
            C = (1 - A) / theta**2

            R = I + A * wx + B * wx @ wx
            V = I + B * wx + C * wx @ wx

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = V @ u
        return T

    def _adjoint(self, T):
        R = T[:3, :3]
        p = T[:3, 3]
        adjoint = np.zeros((6, 6))
        adjoint[:3, :3] = R
        adjoint[3:, 3:] = R
        adjoint[3:, :3] = self._skew(p) @ R

        return adjoint

    def fk(self, theta, idx_start=0, idx_end=None, use_zero=True):
        if idx_end is None:
            idx_end = self.N
        assert len(theta) == idx_end - idx_start
        assert idx_start >= 0
        assert idx_end <= self.N

        T = np.eye(4)
        for i in range(idx_start, idx_end):
            T = T @ self._exp(theta[i - idx_start] * self.screws[i])

        if use_zero:
            T = T @ self.zero_config

        return T

    def fk_prop_cov(self, theta, var):
        cov = np.zeros((6, 6))
        for i in range(self.N):
            adj_inv = np.linalg.inv(
                self._adjoint(self.fk(theta[i + 1 :], idx_start=i + 1))
            )
            xi = adj_inv @ self.screws[i]
            cov += np.outer(xi, xi) * var[i]

        return cov
