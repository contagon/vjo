import argparse
import datetime
import os

import cv2
import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from tqdm import tqdm

from vjo.sim import ArmSim


def run_sim(args):
    # ------------------------- Set up simulation ------------------------- #
    # Setup simulation environment
    sim = ArmSim(viz=args.show_vis, time_step=args.delta_t)
    # Setup everything in environment
    sim.add_arm()

    if args.add_objects:
        # Add sdf files
        sim.add_mesh(
            "meshes/table.sdf",
            "table_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, -0.05]),
        )
        sim.add_mesh(
            "meshes/002_master_chef_can/master_chef_can.sdf",
            "master_chef_can_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [0.1, 1.05, 0]),
        )
        sim.add_mesh(
            "meshes/003_cracker_box/cracker_box.sdf",
            "cracker_box_link",
            RigidTransform(RollPitchYaw([0, 0, 1.47]), [0.1, 1.15, 0]),
        )
        sim.add_mesh(
            "meshes/004_sugar_box/sugar_box.sdf",
            "sugar_box_link",
            RigidTransform(RollPitchYaw([0, 0, 1.57]), [0.25, 1.15, 0]),
        )
        sim.add_mesh(
            "meshes/005_tomato_soup_can/tomato_soup_can.sdf",
            "tomato_soup_can_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [0.3, 0.9, 0]),
        )
        sim.add_mesh(
            "meshes/006_mustard_bottle/mustard_bottle.sdf",
            "mustard_bottle_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [0.3, 1, 0]),
        )
        sim.add_mesh(
            "meshes/007_tuna_fish_can/tuna_fish_can.sdf",
            "tuna_fish_can_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [0.2, 1, 0]),
        )
        sim.add_mesh(
            "meshes/008_pudding_box/pudding_box.sdf",
            "pudding_box_link",
            RigidTransform(RollPitchYaw([0, 0, 0]), [-0.1, 0.9, 0]),
        )

        delta = RigidTransform(RollPitchYaw(0, 0, 0), [0.4, 0.5, 0])
        sim.add_mesh(
            "meshes/011_banana/banana.sdf",
            "banana_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, -0.5, 0]),
        )
        sim.add_mesh(
            "meshes/012_strawberry/strawberry.sdf",
            "strawberry_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, -0.5, 0]),
        )
        sim.add_mesh(
            "meshes/013_apple/apple.sdf",
            "apple_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.6, -0.55, 0]),
        )
        sim.add_mesh(
            "meshes/014_lemon/lemon.sdf",
            "lemon_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.7, -0.5, 0]),
        )
        sim.add_mesh(
            "meshes/015_peach/peach.sdf",
            "peach_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.7, -0.6, 0]),
        )
        sim.add_mesh(
            "meshes/016_pear/pear.sdf",
            "pear_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0.3]), [0.45, -0.7, 0]),
        )
        sim.add_mesh(
            "meshes/017_orange/orange.sdf",
            "orange_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, -0.7, 0]),
        )
        sim.add_mesh(
            "meshes/018_plum/plum.sdf",
            "plum_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, -0.3, 0]),
        )

        delta = RigidTransform(RollPitchYaw(0, 0, 0), [0.2, 0.2, 0])
        sim.add_mesh(
            "meshes/024_bowl/bowl.sdf",
            "bowl_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, 0.5, 0]),
        )
        sim.add_mesh(
            "meshes/025_mug/mug.sdf",
            "mug_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, 0.55, 0]),
        )
        sim.add_mesh(
            "meshes/026_sponge/sponge.sdf",
            "sponge_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [1, 0, 0]),
        )
        sim.add_mesh(
            "meshes/028_skillet_lid/skillet_lid.sdf",
            "skillet_lid_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, 0.8, 0]),
        )
        sim.add_mesh(
            "meshes/029_plate/plate.sdf",
            "plate_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 1.3]), [0.6, 0.3, 0]),
        )
        sim.add_mesh(
            "meshes/030_fork/fork.sdf",
            "fork_link",
            delta @ RigidTransform(RollPitchYaw([0, 0.2, 0.57]), [0.65, 0.4, 0]),
        )
        sim.add_mesh(
            "meshes/031_spoon/spoon.sdf",
            "spoon_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, 0]), [0.65, 0.45, 0]),
        )
        sim.add_mesh(
            "meshes/032_knife/knife.sdf",
            "knife_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, -0.57]), [0.7, 0.5, 0]),
        )
        sim.add_mesh(
            "meshes/033_spatula/spatula.sdf",
            "spatula_link",
            delta @ RigidTransform(RollPitchYaw([0, 0, -1.5]), [0.9, 0.5, 0]),
        )

    sim.plant_finalize()

    # Add in sensors and controller
    sim.add_controller()
    sim.add_camera(show=args.show_cam)
    # Visualizer end effector pose
    sim.add_frame(7)

    # Get sim ready
    q0 = np.array([np.pi / 2, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2])
    sim.sim_setup(q0, wait_load=0)

    N = int(args.num_seconds // args.delta_t)
    N_fourth = int(N // 4)
    N = 4 * N_fourth

    # Define waypoints to move to
    diff = 0.25
    qd = np.array(
        [
            [0, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2],
            [np.pi / 2, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2],
            [0, 0, 0, -np.pi / 2.0 + diff, 0, np.pi / 4 + diff * 2, -np.pi / 2],
            [np.pi / 2, 0, 0, -np.pi / 2.0 + diff, 0, np.pi / 4 + diff * 2, -np.pi / 2],
        ]
    )

    def get_q_des(i):
        # joint0 = np.linspace(np.pi / 2.0, 0, N)
        # qd_now = np.array([0.0, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2])
        # qd_now[0] = joint0[i]

        qd_now = qd[int(i // N_fourth)]

        return qd_now

    # ------------------------- Run simulation ------------------------- #
    dirname = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    os.mkdir(dirname)

    # Run simulation
    joints = []
    for i in tqdm(range(N)):
        qd_now = get_q_des(i)
        t, image, plant_state = sim.step(qd=qd_now)
        joints.append(np.insert(plant_state[:7], 0, [i, t]))
        cv2.imwrite(str(os.path.join(dirname, f"image{i:03d}.png")), image)

    # ------------------------- Save data ------------------------- #
    header = "index,time,joint0,joint1,joint2,joint3,joint4,joint5,joint6"
    np.savetxt(os.path.join(dirname, "joints.csv"), joints, header=header)

    if os.path.exists("latest"):
        os.unlink("latest")
    os.symlink(dirname, "latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add_objects",
        type=bool,
        default=True,
        help="Whether to fill in the environment",
    )
    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.1,
        help="Amount of time between timesteps",
    )
    parser.add_argument(
        "-t",
        "--num_seconds",
        type=float,
        default=10,
        help="Length of simulation",
    )
    parser.add_argument(
        "--show_cam",
        action="store_true",
        help="Whether to render camera live",
    )
    parser.add_argument(
        "--show_vis",
        action="store_true",
        help="Whether to launch meshcat",
    )
    args = parser.parse_args()
    run_sim(args)
