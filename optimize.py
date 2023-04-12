import os

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from fk import ArmSE3

# ------------------------- Parameters ------------------------- #
file_joints = ""
dir_camera = ""


# ------------------------- Setup arm ------------------------- #
# Setup robot arm
# TODO Switch to iiwa screws
M = 7
zero = np.eye(4)

w = np.array([[0, 0, 1], [0, 0, 1]])
p = np.array([[0, 0, 0], [0, 0, 0]])

screws = np.zeros((M, 6))
for i in range(2):
    screws[i, 3:] = -np.cross(w[i], p[i])
    screws[i, :3] = w[i]

arm = ArmSE3(screws, zero)


# ------------------------- Load data ------------------------- #
joints = np.load(file_joints)
cov = np.ones(M)

files_camera = [
    f for f in os.listdir(dir_camera) if os.path.isfile(os.path.join(dir_camera, f))
]
images = [cv2.imread(f) for f in files_camera]
N = joints.shape[0]


# ------------------------- Run factor graph ------------------------- #

graph = gtsam.NonlinearFactorGraph()
theta = gtsam.Values()

for i in range(N):
    # Add joint prior factor
    fk_est = arm.fk(joints[i])
    # TODO: Implement arm.cov
    fk_propagated_cov = arm.cov(joints[i], cov)

    prior = gtsam.PriorFactorPose3(
        X(i), fk_est, gtsam.noiseModel.Gaussian.Covariance(fk_propagated_cov)
    )
    graph.push_back(prior)
    theta.insert(X(i), fk_est)

    # TODO: Implement camera factors
