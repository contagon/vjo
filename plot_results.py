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
    joints = np.loadtxt(
        os.path.join(args.data_folder, "joints.csv"),
        skiprows=1,
    )[:, 2:]

    poses_opt = [gtsam.Pose3(d.reshape((3, 4))) for d in result]
    poses_gt = [gtsam.Pose3(arm.fk(j)) for j in joints]

    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection="3d"))
    for po, pg in zip(poses_opt, poses_gt):
        plot.plot_pose3_on_axes(ax, po, scale=0.9)
        plot.plot_pose3_on_axes(ax, pg, scale=1.1)

    plot.set_axes_equal(1)
    plt.show()


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
    run(args)
