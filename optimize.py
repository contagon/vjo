import argparse
import os

import cv2
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gtsam.symbol_shorthand import L, X
from gtsam.utils import plot
from tqdm import tqdm

from vjo.fk import iiwa7


def optimize(args):
    # ------------------------- Setup arm ------------------------- #
    # Setup robot arm
    arm = iiwa7()

    # ------------------------- Load data ------------------------- #
    file_joints = os.path.join(args.data_folder, "joints.csv")

    # TODO: Save joint covariance at top of file?
    cov = 0.0001 * np.ones(arm.N)

    joints = np.loadtxt(file_joints, skiprows=1)
    N = joints.shape[0]

    files_camera = [
        os.path.join(args.data_folder, f)
        for f in os.listdir(args.data_folder)
        if os.path.splitext(f)[1] == ".png"
    ]
    files_camera = sorted(files_camera)

    rng = np.random.default_rng(12345)
    rng: np.random.Generator
    noisy_joints = joints + rng.normal(0.0, 0.01, joints.shape)

    # ------------------------- Run factor graph ------------------------- #

    graph = gtsam.NonlinearFactorGraph()
    theta = gtsam.Values()

    # Add in joint priors
    for i in range(N):
        # Get SE3 estimate and propagated covariance
        joint_configuration = noisy_joints[i, 2:9]
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

    # Iterate through images
    for i, camera_file in tqdm(enumerate(files_camera[1:], start=1)):
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

        # Run RANSAC
        prev_matched_keypoints = np.array(
            [prev_keypoints[m.queryIdx].pt for m in matches]
        )
        new_matched_keypoints = np.array(
            [new_keypoints[m.trainIdx].pt for m in matches]
        )
        E, inliers = cv2.findEssentialMat(
            prev_matched_keypoints,
            new_matched_keypoints,
            intrinsics_matrix,
            method=cv2.RANSAC,
            threshold=1,
            prob=0.999,
        )
        matches = [m for i, m in enumerate(matches) if inliers[i] == 1]

        for match in matches:
            prev_index = match.queryIdx
            new_index = match.trainIdx
            if prev_keypoint_indices[prev_index] == NOT_MATCHED:
                measurements = gtsam.Point2Vector(
                    [prev_keypoints[prev_index].pt, new_keypoints[new_index].pt]
                )
                try:
                    triangulatedPoint3 = gtsam.triangulatePoint3(
                        poses, calibration, measurements, rank_tol=1e-5, optimize=True
                    )
                except Exception:
                    print("Cheirality: point not added")
                    continue
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
                theta.insert(L(prev_keypoint_indices[prev_index]), triangulatedPoint3)

            new_keypoint_indices[new_index] = prev_keypoint_indices[prev_index]
            factor = gtsam.GenericProjectionFactorCal3_S2(
                np.array(new_keypoints[new_index].pt),
                robust_noise,
                X(i),
                L(new_keypoint_indices[new_index]),
                calibration,
            )
            graph.push_back(factor)

        prev_keypoints = new_keypoints
        prev_descriptors = new_descriptors
        prev_keypoint_indices = new_keypoint_indices

        prev_image = new_image

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, theta)
    solution = optimizer.optimize()
    with open(os.path.join(args.data_folder, "odometry.csv"), "w") as save_file:
        save_file.write(
            "r_11,r_12,r_13,p_x,r_21,r_22,r_23,p_y,r_31,r_32,r_33,p_z,measured_joint0,measured_joint1,measured_joint2,measured_joint3,measured_joint4,measured_joint5,measured_joint6,true_joint0,true_joint1,true_joint2,true_joint3,true_joint4,true_joint5,true_joint6"
        )
        for k in range(N):
            pose = solution.atPose3(X(k))
            pose: gtsam.Pose3
            flattened_pose = np.ravel(pose.matrix())[:12]
            save_file.write("\n")

            for value in flattened_pose:
                save_file.write(str(value))
                save_file.write(",")

            for measured_joint in noisy_joints[k, 2:]:
                save_file.write(str(measured_joint))
                save_file.write(",")

            for true_joint in joints[k, 2:-1]:
                save_file.write(str(true_joint))
                save_file.write(",")

            save_file.write(str(joints[k, -1]))


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
