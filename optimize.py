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

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_image = cv2.imread(files_camera[0])
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_image, None)
    NOT_MATCHED = -1
    prev_keypoint_indices = [NOT_MATCHED for prev_keypoint in prev_keypoints]
    keypoint_count = 0

    for i, camera_file in enumerate(files_camera[1:]):
        new_image = cv2.imread(camera_file)
        new_keypoints, new_descriptors = orb.detectAndCompute(new_image, None)
        matches = matcher.match(prev_keypoints, new_keypoints)
        matches = sorted(matches, lambda match: match.distance)
        new_keypoint_indices = [NOT_MATCHED for match in matches]
        for match in matches[:10]:
            prev_index = match.trainIdx
            new_index = match.queryIdx
            if prev_keypoint_indices[prev_index] == NOT_MATCHED:
                prev_keypoint_indices[prev_index] = keypoint_count
                keypoint_count += 1
                prev_factor = gtsam.GenericProjectionFactorCal3_S2(
                    prev_keypoints[prev_index],
                    noise,
                    X(i - 1),
                    L(prev_keypoint_indices[prev_index]),
                    calibration,
                )
                graph.push_back(prev_factor)
                theta.insert(
                    L(prev_keypoint_indices[prev_index]),
                    gtsam.triangulatePoint3(
                        [theta.atPose3(X(i - 1)), theta.atPose3(X(i))],
                        calibration,
                        measuments,
                    ),
                )
            new_keypoint_indices[new_index] = prev_keypoint_indices[prev_index]
            factor = gtsam.GenericProjectionFactorCal3_S2(
                new_keypoint_indices[new_index],
                noise,
                X(i),
                L(new_keypoint_indices[new_index]),
                calibration,
            )
            graph.push_back(factor)
            prev_keypoints = new_keypoints
            prev_descriptors = new_descriptors
            prev_keypoint_indices = new_keypoint_indices

        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, theta)
        solution = optimizer.optimize()


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
