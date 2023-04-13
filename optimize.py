import argparse
import os

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import X

from fk import ArmSE3


def optimize(args):
    # ------------------------- Setup arm ------------------------- #
    # Setup robot arm
    M = 7

    dbs = 0.340
    dse = 0.400
    dew = 0.400
    dwf = 0.126

    zero = np.eye(4)
    zero[:3, 3] = [0, 0, dbs + dse + dew + dwf]
    w = np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    p = np.array(
        [
            [0, 0, 0],
            [0, 0, dbs],
            [0, 0, dbs],
            [0, 0, dbs + dse],
            [0, 0, dbs + dse],
            [0, 0, dbs + dse + dew],
            [0, 0, dbs + dse + dew + dwf],
        ]
    )

    screws = np.zeros((M, 6))
    screws[:, :3] = w
    for i in range(M):
        screws[i, 3:] = -np.cross(w[i], p[i])

    arm = ArmSE3(screws, zero)

    # ------------------------- Load data ------------------------- #
    file_joints = os.path.joint(args.data_folder, "joints.csv")

    # TODO: Save joint covariance at top of file?
    cov = np.ones(M)

    joints = np.load(file_joints)
    N = joints.shape[0]

    files_camera = [
        f
        for f in os.listdir(args.data_folder)
        if os.path.isfile(os.path.join(args.data_folder, f))
    ]
    [cv2.imread(f) for f in files_camera]

    # ------------------------- Run factor graph ------------------------- #

    graph = gtsam.NonlinearFactorGraph()
    theta = gtsam.Values()

    for i in range(N):
        # Get SE3 estimate and propagated covariance
        fk_est = arm.fk(joints[i])
        fk_propagated_cov = arm.fk_prop_cov(joints[i], cov)

        # Add factor to graph
        prior = gtsam.PriorFactorPose3(
            X(i), fk_est, gtsam.noiseModel.Gaussian.Covariance(fk_propagated_cov)
        )
        graph.push_back(prior)
        theta.insert(X(i), fk_est)

        # TODO: Implement camera factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_folder",
        type=str,
        default="data",
        help="The folder containing joint data and camera images",
    )
    args = parser.parse_args()
    optimize(args)
