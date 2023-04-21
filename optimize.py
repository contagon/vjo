import argparse
import os

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtsam.symbol_shorthand import L, X
from gtsam.utils import plot

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
            [0, 0, dbs + dse + dew + dwf + 0.1],
        ]
    )

    screws = np.zeros((M, 6))
    screws[:, :3] = w
    for i in range(M):
        screws[i, 3:] = -np.cross(w[i], p[i])

    arm = ArmSE3(screws, zero)

    # ------------------------- Load data ------------------------- #
    file_joints = os.path.join(args.data_folder, "joints.csv")

    # TODO: Save joint covariance at top of file?
    cov = np.ones(M)

    data = pd.read_csv(file_joints)
    joints = data.to_numpy()
    N = joints.shape[0]

    files_camera = [
        os.path.join(args.data_folder, f)
        for f in os.listdir(args.data_folder)
        if os.path.splitext(f)[1] == ".png"
    ]
    files_camera = sorted(files_camera)

    # ------------------------- Run factor graph ------------------------- #

    graph = gtsam.NonlinearFactorGraph()
    theta = gtsam.Values()

    for i in range(N):
        # Get SE3 estimate and propagated covariance
        joint_configuration = joints[i, 2:9]
        fk_est = arm.fk(joint_configuration)
        fk_propagated_cov = arm.fk_prop_cov(joint_configuration, cov)

        # Add factor to graph
        prior = gtsam.PriorFactorPose3(
            X(i),
            gtsam.gtsam.Pose3(fk_est),
            gtsam.noiseModel.Gaussian.Covariance(fk_propagated_cov),
        )
        graph.push_back(prior)
        theta.insert(X(i), gtsam.gtsam.Pose3(fk_est))

    orb = cv2.ORB_create()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_image = cv2.imread(files_camera[0])
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_image, None)
    NOT_MATCHED = -1
    prev_keypoint_indices = [NOT_MATCHED for prev_keypoint in prev_keypoints]
    keypoint_count = 0
    intrinsics_matrix = np.loadtxt("intrinsics.txt")
    calibration = gtsam.Cal3_S2(
        intrinsics_matrix[0, 0],
        intrinsics_matrix[1, 1],
        intrinsics_matrix[0, 1],
        intrinsics_matrix[0, -1],
        intrinsics_matrix[1, 1],
    )
    camera_cov = np.diag([0.1, 0.1])
    camera_noise = gtsam.noiseModel.Gaussian.Covariance(camera_cov)
    robust_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), camera_noise
    )

    for i, camera_file in enumerate(files_camera[1:], start=1):
        new_image = cv2.imread(camera_file)
        new_keypoints, new_descriptors = orb.detectAndCompute(new_image, None)
        matches = matcher.match(prev_descriptors, new_descriptors)
        matches = sorted(matches, key=lambda match: match.distance)
        new_keypoint_indices = [NOT_MATCHED for keypoint in new_keypoints]
        poses = gtsam.Pose3Vector(
            [
                theta.atPose3(X(i - 1)),
                theta.atPose3(X(i)),
            ]
        )

        num_matches = 10
        outImg = cv2.drawMatches(
            prev_image,
            prev_keypoints,
            new_image,
            new_keypoints,
            matches[:num_matches],
            None,
        )
        cv2.imshow("matches", outImg)
        cv2.waitKey(1)

        fig = plt.figure(0)
        axes = fig.add_subplot(projection="3d")
        plot.plot_pose3_on_axes(axes, poses[0])
        plot.plot_pose3_on_axes(axes, poses[1])

        for match in matches:
            prev_index = match.queryIdx
            new_index = match.trainIdx
            try:
                if prev_keypoint_indices[prev_index] == NOT_MATCHED:
                    measurements = gtsam.Point2Vector(
                        [prev_keypoints[prev_index].pt, new_keypoints[new_index].pt]
                    )
                    triangulatedPoint3 = gtsam.triangulatePoint3(
                        poses, calibration, measurements, rank_tol=1e-5, optimize=True
                    )
                    prev_keypoint_indices[prev_index] = keypoint_count
                    keypoint_count += 1

                    prev_factor = gtsam.GenericProjectionFactorCal3_S2(
                        measurements[0],
                        robust_noise,
                        X(i - 1),
                        L(prev_keypoint_indices[prev_index]),
                        calibration,
                    )
                    graph.push_back(prev_factor)
                    plot.plot_point3_on_axes(axes, triangulatedPoint3, "o")
                    theta.insert(
                        L(prev_keypoint_indices[prev_index]), triangulatedPoint3
                    )

                new_keypoint_indices[new_index] = prev_keypoint_indices[prev_index]
                factor = gtsam.GenericProjectionFactorCal3_S2(
                    np.array(new_keypoints[new_index].pt),
                    robust_noise,
                    X(i),
                    L(new_keypoint_indices[new_index]),
                    calibration,
                )
                graph.push_back(factor)
            except:
                print("Cheirality: point not added")

        prev_keypoints = new_keypoints
        prev_descriptors = new_descriptors
        prev_keypoint_indices = new_keypoint_indices

        prev_image = new_image

        plot.set_axes_equal(0)
        plt.show()

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, theta)
    solution = optimizer.optimize()
    with open(os.path.join(args.data_folder, "odometry.csv"), "w") as save_file:
        save_file.write("r_11,r_12,r_13,p_x,r_21,r_22,r_23,p_y,r_31,r_32,r_33,p_z")
        for k in range(N):
            pose = solution.atPose3(X(k))
            pose: gtsam.Pose3
            save_file.write("\n")
            for value in np.ravel(pose.matrix())[:11]:
                save_file.write(str(value))
                save_file.write(",")
            save_file.write(str(np.ravel(pose.matrix())[11]))


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
