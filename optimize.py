import argparse
import os

import cv2
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
from tqdm import tqdm

from vjo.fk import iiwa7


def optimize(args):
    # ------------------------- Setup arm ------------------------- #
    # Setup robot arm
    arm = iiwa7()

    std = 0.1
    cov = std**2 * np.ones(arm.N)

    # ------------------------- Load data ------------------------- #
    file_joints = os.path.join(args.data_folder, "joints.csv")
    joints = np.loadtxt(file_joints, skiprows=1, delimiter=",")[:, 2:9]
    N = joints.shape[0]

    intrinsics_matrix = np.loadtxt(os.path.join(args.data_folder, "intrinsics.csv"))

    files_camera = [
        os.path.join(args.data_folder, f)
        for f in os.listdir(args.data_folder)
        if os.path.splitext(f)[1] == ".png"
    ]
    files_camera = sorted(files_camera)

    rng = np.random.default_rng(1)
    rng: np.random.Generator
    noisy_joints = joints + rng.normal(0.0, std, joints.shape)

    # ------------------------- Run factor graph ------------------------- #

    graph = gtsam.NonlinearFactorGraph()
    theta = gtsam.Values()

    # Add in joint priors
    for i in range(N):
        # Get SE3 estimate and propagated covariance
        joint_configuration = noisy_joints[i]
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

    # Setup feature matching
    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    NOT_MATCHED = -1
    keypoint_count = 0
    calibration = gtsam.Cal3_S2(
        intrinsics_matrix[0, 0],
        intrinsics_matrix[1, 1],
        intrinsics_matrix[0, 1],
        intrinsics_matrix[0, -1],
        intrinsics_matrix[1, -1],
    )
    camera_cov = np.diag([0.01, 0.01])
    camera_noise = gtsam.noiseModel.Gaussian.Covariance(camera_cov)
    robust_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(1.345), camera_noise
    )

    # Match first image
    prev_image = cv2.imread(files_camera[0])
    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_image, None)
    prev_keypoint_indices = [NOT_MATCHED for prev_keypoint in prev_keypoints]
    num_landmarks = np.zeros(len(files_camera))

    # Iterate through images
    print("Matching images...")
    for i, camera_file in tqdm(
        enumerate(files_camera[1:], start=1), total=len(files_camera[1:])
    ):
        new_image = cv2.imread(camera_file)
        new_keypoints, new_descriptors = orb.detectAndCompute(new_image, None)
        matches = matcher.match(prev_descriptors, new_descriptors)
        matches = sorted(matches, key=lambda match: match.distance)
        new_keypoint_indices = [NOT_MATCHED for keypoint in new_keypoints]

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
            prob=0.9999,
        )
        num_landmarks[i] = inliers.sum()
        poses = gtsam.Pose3Vector(
            [
                # theta.atPose3(X(i - 1)),
                # theta.atPose3(X(i)),
                gtsam.Pose3(arm.fk(joints[i - 1])),
                gtsam.Pose3(arm.fk(joints[i])),
            ]
        )
        matches = [m for i, m in enumerate(matches) if inliers[i] == 1]

        # outimg = cv2.drawMatches(
        #           prev_image, prev_keypoints, new_image, new_keypoints, matches, None
        # )
        # cv2.imshow('match', outimg)
        # cv2.waitKey()

        for match in matches:
            prev_index = match.queryIdx
            new_index = match.trainIdx
            # If it hasn't been matched before
            if prev_keypoint_indices[prev_index] == NOT_MATCHED:
                measurements = gtsam.Point2Vector(
                    [prev_keypoints[prev_index].pt, new_keypoints[new_index].pt]
                )
                try:
                    triangulatedPoint3 = gtsam.triangulatePoint3(
                        poses, calibration, measurements, rank_tol=1e-5, optimize=True
                    )
                except Exception:
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

    print(
        (
            f"Matched {keypoint_count} landmarks,"
            f" {keypoint_count/len(files_camera[1:]):.2f} per frame"
        )
    )
    print("Optimizing...")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, theta)
    solution = optimizer.optimize()

    # Save noisy joint values
    header = (
        "measured_joint0,"
        "measured_joint1,"
        "measured_joint2,"
        "measured_joint3,"
        "measured_joint4,"
        "measured_joint5,"
        "measured_joint6"
    )
    np.savetxt(
        os.path.join(args.data_folder, "measurements.csv"),
        noisy_joints,
        header=header,
        delimiter=",",
    )

    # Save solution
    header = "q_w,q_x,q_y,q_z,p_x,p_y,p_z"
    results = [solution.atPose3(X(i)) for i in range(N)]
    results = np.array(
        [np.append(i.rotation().quaternion(), i.translation()) for i in results]
    )
    np.savetxt(
        os.path.join(args.data_folder, "odometry.csv"),
        results,
        header=header,
        delimiter=",",
    )

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_folder",
        type=str,
        default="latest",
        help="The folder containing joint data and camera images",
    )
    args = parser.parse_args()
    optimize(args)
