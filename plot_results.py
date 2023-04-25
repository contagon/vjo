import argparse
import os

import gtsam
import matplotlib.pyplot as plt
import numpy as np
from gtsam.utils import plot

from vjo.fk import ArmFK


def iiwa7():
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

    return ArmFK(screws, zero)


def run(args):
    arm = iiwa7()
    result = np.loadtxt(
        os.path.join(args.data_folder, "odometry.csv"), skiprows=1, delimiter=","
    )[:, :12]
    noisy_joints = np.loadtxt(
        os.path.join(args.data_folder, "odometry.csv"), skiprows=1, delimiter=","
    )[:, 12:19]
    joints = np.loadtxt(
        os.path.join(args.data_folder, "joints.csv"),
        skiprows=1,
    )[:, 2:]

    poses_opt = [gtsam.Pose3(d.reshape((3, 4))) for d in result]
    poses_gt = [gtsam.Pose3(arm.fk(j)) for j in joints]
    noisy_poses = [gtsam.Pose3(arm.fk(j)) for j in noisy_joints]

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    for po, pg in zip(poses_opt, poses_gt):
        plot.plot_pose3_on_axes(ax, po, scale=0.9)
        plot.plot_pose3_on_axes(ax, pg, scale=1.1)

    plot.set_axes_equal(1)

    fig2, axs = plt.subplots(nrows=3, sharex=True)
    axs = np.ravel(axs)
    axs[0].plot(result[:, 3], label="VJO")
    axs[0].plot([pose_gt.translation()[0] for pose_gt in poses_gt], label="GT")
    axs[0].plot(
        [pose_gt.translation()[0] for pose_gt in noisy_poses], label="noisy_poses"
    )
    axs[0].set_ylabel("x (m)")
    axs[0].legend()
    axs[1].plot(result[:, 7], label="VJO")
    axs[1].plot([pose_gt.translation()[1] for pose_gt in poses_gt], label="GT")
    axs[1].plot(
        [pose_gt.translation()[1] for pose_gt in noisy_poses], label="noisy_poses"
    )
    axs[1].set_ylabel("y (m)")
    axs[1].legend()
    axs[2].plot(result[:, 11], label="VJO")
    axs[2].plot([pose_gt.translation()[2] for pose_gt in poses_gt], label="GT")
    axs[2].plot(
        [pose_gt.translation()[2] for pose_gt in noisy_poses], label="noisy_poses"
    )
    axs[2].set_ylabel("z (m)")
    axs[2].legend()

    plt.show()


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
    run(args)
