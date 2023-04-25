import argparse
import os

import gtsam
import matplotlib.pyplot as plt
import numpy as np

from vjo.fk import iiwa7
from vjo.utils import setup_plot


def run(args):
    arm = iiwa7()

    # ------------------------- Load data ------------------------- #
    result = np.loadtxt(os.path.join(args.data_folder, "odometry.csv"), skiprows=1)
    noisy_joints = np.loadtxt(
        os.path.join(args.data_folder, "measurements.csv"), skiprows=1
    )
    joints = np.loadtxt(
        os.path.join(args.data_folder, "joints.csv"),
        skiprows=1,
    )
    t = joints[:, 1]

    poses_opt = [gtsam.Pose3(gtsam.Rot3(*i[:4]), i[4:]) for i in result]
    poses_gt = [gtsam.Pose3(arm.fk(j[2:])) for j in joints]
    poses_noisy = [gtsam.Pose3(arm.fk(j)) for j in noisy_joints]

    # ------------------------- Plot! ------------------------- #

    setup_plot()

    fig2, axs = plt.subplots(nrows=3, ncols=2, sharex=True, layout="constrained")
    axs = axs.T.flatten()
    names = ["X (m)", "Y (m)", "z (m)", "Roll (deg)", "Pitch (deg)", "Yaw (deg)"]
    for i in range(3):
        axs[i].plot(t, [p.translation()[i] for p in poses_opt], marker=".", label="GT")
        axs[i].plot(t, [p.translation()[i] for p in poses_gt], marker=".", label="VJO")
        axs[i].plot(
            t,
            [p.translation()[i] for p in poses_noisy],
            marker=".",
            label="Encoders",
        )
        # axs[i].legend()
        axs[i].set_ylabel(names[i])

    for i in range(3):
        axs[i + 3].plot(
            t, [p.rotation().rpy()[i]*180/np.pi for p in poses_opt], marker=".", label="GT"
        )
        axs[i + 3].plot(
            t, [p.rotation().rpy()[i]*180/np.pi for p in poses_gt], marker=".", label="VJO"
        )
        axs[i + 3].plot(
            t,
            [p.rotation().rpy()[i]*180/np.pi for p in poses_noisy],
            marker=".",
            label="Encoders",
            alpha=0.6
        )
        # axs[i+3].legend()
        axs[i + 3].set_ylabel(names[i + 3])

    axs[3].legend()

    plt.savefig('figures/joints.png')
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
