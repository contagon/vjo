# Import some basic libraries and functions for this tutorial.
import datetime
import os
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import pydot
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, StartMeshcat
from pydrake.geometry.render import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor

from utils import AddMultibodyTriad

"""
TODO LIST

Functionality:
- Figure out how to pull joint data
- Make camera scene more interesting
- Probably need to make more interesting trajectories

Code Quality:
- Test? That may be overkill for this

Links that helped for ticking:
https://github.com/RussTedrake/manipulation/blob/master/manipulation/drake_gym.py
https://github.com/RussTedrake/manipulation/blob/master/manipulation/envs/box_flipup.py
https://github.com/RobotLocomotion/drake/issues/15508
"""

# class NoisyJointSensor(LeafSystem):
#     def __init__(self, variance: float = 0.01):
#         super.__init__()
#         self.variance = variance

#     def input_port_ticket(self, plant)

#     def output_port(self, )


class ArmSim:
    def __init__(
        self, viz: bool = True, time_step: float = 0.1, sim_time_step: float = 0.0001
    ) -> None:
        """Arm simulation class.

        Args:
            viz (bool, optional): Whether to run in meshcat visualizer.
                Defaults to True.
            time_step (float, optional): Amount of time between simulation ticks.
                Defaults to 0.1.
            sim_time_step (float, optional): Integration step used for dynamics.
                Defaults to 0.0001.
        """
        # Save parameters
        self.viz = viz
        self.time_step = time_step
        self.image_count = 0

        if self.viz:
            self.meshcat = StartMeshcat()

        # Builder sets up the diagram
        # Diagram is all of the seperate systems combined
        self.builder = DiagramBuilder()
        # Plant = robot arm and anything else in the scene
        # Scene_graph = i don't really know
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=sim_time_step
        )
        # Parser reads .sdf and .urdf files and puts them into the plant
        self.parser = Parser(self.plant)
        self.camera = None

    def add_arm(
        self, arm: str = "iiwa7", offset: Optional[RigidTransform] = None
    ) -> None:
        """Adds robot arm to the simulation.

        Args:
            arm (str, optional): Name of robot arm to install. Defaults to "iiwa7".
            offset (RigidTransform, optional): Offset from world origin to be
                installed at. Defaults to None.
        """
        if arm == "iiwa7":
            sdf = (
                "package://drake/manipulation/models/"
                "iiwa_description/iiwa7/iiwa7_with_box_collision.sdf"
            )
            self.N = 7
            self.link_names = lambda i: f"iiwa_link_{i}"
        else:
            raise TypeError("Invalid robot arm type")

        if offset is None:
            offset = RigidTransform()

        self.parser.AddModels(url=sdf)

        # Connect arm to world origin
        L0 = self.plant.GetFrameByName(self.link_names(0))
        self.plant.WeldFrames(self.plant.world_frame(), L0, offset)

    def plant_finalize(self):
        """Finalize the plant - means we're not adding anything else to it"""
        self.plant.Finalize()

    def add_frame(
        self, frame: Union[int, str], length: float = 0.25, radius: float = 0.01
    ) -> None:
        """Visualize a frame in the simulation.

        Args:
            frame (Union[int, str]): Either the name of a frame, or a
                integer of the arm link.
            length (float, optional): Length of frame axes. Defaults to 0.25.
            radius (float, optional): Radius of frame axes. Defaults to 0.01.
        """

        if type(frame) is int:
            frame = self.plant.GetFrameByName(self.link_names(frame))
        elif type(frame) is str:
            frame = self.plant.GetFrameByName(frame)

        AddMultibodyTriad(frame, self.scene_graph, length=length, radius=radius)

    def add_mesh(self, model: str, frame_name: str, offset: RigidTransform) -> None:
        """Add another welded mesh to the simulation.

        Args:
            model (str): SDF file location.
            frame_name (str): Name of frame to weld to world origin.
            offset (RigidTransform): Offset from world origin to put it in.
        """
        sim.parser.AddModels(model)
        self.plant.WeldFrames(
            self.plant.world_frame(), self.plant.GetFrameByName(frame_name), offset
        )

    def add_mesh1(self, model: str) -> None:
        """Add another welded mesh to the simulation.

        Args:
            model (str): SDF file location.
            frame_name (str): Name of frame to weld to world origin.
            offset (RigidTransform): Offset from world origin to put it in.
        """
        sim.parser.AddModels(model)
        # self.plant.WeldFrames(
        #     self.plant.world_frame(), self.plant.GetFrameByName(frame_name), offset
        # )

    def add_camera(
        self,
        width: int = 640,
        height: int = 480,
        fov_y: float = np.pi / 4,
        offset: Optional[RigidTransform] = None,
    ) -> np.ndarray:
        """Add a camera to the simulation

        Args:
            width (int, optional): Image width. Defaults to 640.
            height (int, optional): Image height. Defaults to 480.
            fov_y (float, optional): Field of view for x/y directions.
                Defaults to np.pi/4.
            offset (RigidTransform, optional): Offset from the last end effector link.
                Defaults to x=0.1.

        Returns:
            np.ndarray: _description_
        """
        if offset is None:
            offset = RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, 0.1])

        # Make renderer for cameras
        renderer_name = "renderer"
        self.scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

        # Add in sensors
        # TODO: Customize this a bit more?
        # https://drake.mit.edu/doxygen_cxx/classdrake_1_1systems_1_1sensors_1_1_camera_info.html
        intrinsics = CameraInfo(
            width=width,
            height=height,
            fov_y=fov_y,
        )
        core = RenderCameraCore(
            renderer_name,
            intrinsics,
            ClippingRange(0.01, 10.0),
            RigidTransform(),
        )
        color_camera = ColorRenderCamera(core, show_window=True)
        depth_camera = DepthRenderCamera(core, DepthRange(0.01, 10.0))
        L7 = self.plant.GetFrameByName(self.link_names(7))
        # Make camera
        self.camera = RgbdSensor(
            self.plant.GetBodyFrameIdOrThrow(L7.body().index()),
            offset,
            color_camera=color_camera,
            depth_camera=depth_camera,
        )
        np.savetxt("intrinsics.txt", intrinsics.intrinsic_matrix())

        # Connect with outputs
        self.builder.AddSystem(self.camera)
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.camera.query_object_input_port(),
        )

        self.builder.ExportOutput(self.camera.color_image_output_port(), "color_image")
        # self.builder.ExportOutput(self.camera.depth_image_32F_output_port(), "depth_image") # noqa 501

        return intrinsics.intrinsic_matrix()

    def add_controller(self, kp: float = 10, kd: float = 5, ki: float = 1) -> None:
        """Add a PID controller to control arm.

        Args:
            kp (float, optional): Proportional gain. Defaults to 10.
            kd (float, optional): Derivative gain. Defaults to 5.
            ki (float, optional): Integral gain. Defaults to 1.
        """
        cont = InverseDynamicsController(
            self.plant,
            kp=np.full(self.N, kp),
            kd=np.full(self.N, kd),
            ki=np.full(self.N, ki),
            has_reference_acceleration=False,
        )
        self.controller = self.builder.AddNamedSystem("controller", cont)

        # Connect arm output to controller input
        self.builder.Connect(
            self.plant.get_state_output_port(),
            self.controller.get_input_port_estimated_state(),
        )
        # Connect controller output to arm input
        self.builder.Connect(
            self.controller.get_output_port_control(),
            self.plant.get_actuation_input_port(),
        )

        # Make input of entire system the controller input
        self.builder.ExportInput(self.controller.get_input_port_desired_state())
        # Make the arm state an output from the diagram.
        self.builder.ExportOutput(self.plant.get_state_output_port())

    def sim_setup(self, q0: Optional[np.ndarray] = None, wait_load: float = 2) -> None:
        """Setup final simulation environment

        Args:
            q0 (np.ndarray, optional): Initial state of the robot arm. Defaults to None.
            wait_load (float, optional): Sometimes meshcat takes a bit to load and you
                can miss everything. Time to wait before starting. Defaults to 2.
        """
        if self.viz:
            # Add simulation to meshcat
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder,
                self.scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(),
            )

        # Build final diagram
        self.diagram = self.builder.Build()

        # Put it into a simulator & set all the parameters
        self.simulator = Simulator(self.diagram)
        self.simulator.Initialize()
        self.simulator.set_target_realtime_rate(1.0)
        # Context = all state information of simulator
        self.context = self.simulator.get_mutable_context()

        # Parse inputs / desired
        if q0 is None:
            q0 = np.zeros(7)
        # Append velocities of 0s
        q0 = np.append(q0, np.zeros(7))

        # Set starting state
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, self.context
        )
        plant_context.get_mutable_discrete_state_vector().SetFromVector(q0)

        # It takes a second for meshcat to load
        time.sleep(wait_load)

    def save_diagram(self, file: str) -> None:
        """Save diagram. MUST call sim_setup first

        Args:
            file (str): File to save diagram to.
        """
        svg = pydot.graph_from_dot_data(self.diagram.GetGraphvizString(max_depth=2))[
            0
        ].create_svg()
        with open(file, "wb") as f:
            f.write(svg)

    def step(self, qd: np.ndarray) -> Tuple[float, np.ndarray]:
        """Make a single simulation step

        Args:
            qd (np.ndarray): Desire joint angle at this step.

        Returns:
            Tuple[float, np.ndarray]: time, image
        """
        time = self.context.get_time()

        # Append velocities of 0s
        qd = np.append(qd, np.zeros(7))

        # Set the desired joint angle
        self.diagram.get_input_port(0).FixValue(self.context, qd)

        self.simulator.AdvanceTo(time + self.time_step)

        # Get image if cammera has been added
        if self.camera is not None:
            image = (
                self.diagram.GetOutputPort("color_image")
                .Eval(self.context)
                .data[..., :3]
            )
            plant_state = self.diagram.GetOutputPort("plant_state").Eval(self.context)
        else:
            image = None
            plant_state = None

        return time, image, plant_state


if __name__ == "__main__":
    # ------------------------- Set up simulation ------------------------- #
    # Setup simulation environment
    sim = ArmSim(viz=True, time_step=1)
    # Setup everything in environment
    sim.add_arm()

    # Add sdf files
    sim.add_mesh(
        "meshes/table.sdf",
        "table_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0, 0, -0.05]),
    )
    # sim.add_mesh(
    #     "meshes/001_chips_can/chips_can.sdf",
    #     "chips_can_link",
    #     RigidTransform(RollPitchYaw([0, 0, 0]), [0, 1, 0]),
    # )
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

    sim.add_mesh(
        "meshes/011_banana/banana.sdf",
        "banana_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, -0.5, 0]),
    )
    sim.add_mesh(
        "meshes/012_strawberry/strawberry.sdf",
        "strawberry_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, -0.5, 0]),
    )
    sim.add_mesh(
        "meshes/013_apple/apple.sdf",
        "apple_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.6, -0.55, 0]),
    )
    sim.add_mesh(
        "meshes/014_lemon/lemon.sdf",
        "lemon_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.7, -0.5, 0]),
    )
    sim.add_mesh(
        "meshes/015_peach/peach.sdf",
        "peach_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.7, -0.6, 0]),
    )
    sim.add_mesh(
        "meshes/016_pear/pear.sdf",
        "pear_link",
        RigidTransform(RollPitchYaw([0, 0, 0.3]), [0.45, -0.7, 0]),
    )
    sim.add_mesh(
        "meshes/017_orange/orange.sdf",
        "orange_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, -0.7, 0]),
    )
    sim.add_mesh(
        "meshes/018_plum/plum.sdf",
        "plum_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, -0.3, 0]),
    )

    sim.add_mesh(
        "meshes/024_bowl/bowl.sdf",
        "bowl_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, 0.5, 0]),
    )
    sim.add_mesh(
        "meshes/025_mug/mug.sdf",
        "mug_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.55, 0.55, 0]),
    )
    sim.add_mesh(
        "meshes/026_sponge/sponge.sdf",
        "sponge_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [1, 0, 0]),
    )
    sim.add_mesh(
        "meshes/028_skillet_lid/skillet_lid.sdf",
        "skillet_lid_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.5, 0.8, 0]),
    )
    sim.add_mesh(
        "meshes/029_plate/plate.sdf",
        "plate_link",
        RigidTransform(RollPitchYaw([0, 0, 1.3]), [0.6, 0.3, 0]),
    )
    sim.add_mesh(
        "meshes/030_fork/fork.sdf",
        "fork_link",
        RigidTransform(RollPitchYaw([0, 0.2, 0.57]), [0.65, 0.4, 0]),
    )
    sim.add_mesh(
        "meshes/031_spoon/spoon.sdf",
        "spoon_link",
        RigidTransform(RollPitchYaw([0, 0, 0]), [0.65, 0.45, 0]),
    )
    sim.add_mesh(
        "meshes/032_knife/knife.sdf",
        "knife_link",
        RigidTransform(RollPitchYaw([0, 0, -0.57]), [0.7, 0.5, 0]),
    )
    sim.add_mesh(
        "meshes/033_spatula/spatula.sdf",
        "spatula_link",
        RigidTransform(RollPitchYaw([0, 0, -1.5]), [0.9, 0.5, 0]),
    )

    sim.plant_finalize()

    # Add in sensors and controller
    sim.add_controller()
    sim.add_camera()
    # Visualizer end effector pose
    sim.add_frame(7)

    # Get sim ready
    N = 10
    q0 = np.array([np.pi / 2 + 0.1, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2])
    sim.sim_setup(q0, wait_load=3)

    # ------------------------- Run simulation ------------------------- #
    # Run simulation
    qd = np.array([0.0, 0, 0, -np.pi / 2.0, 0, np.pi / 4, -np.pi / 2])
    joint0 = np.linspace(np.pi / 2.0, 0, N)
    dirname = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    os.mkdir(dirname)
    with open(os.path.join(dirname, "joints.csv"), "w") as state_file:
        state_file.write(
            "index,time,joint0,joint1,joint2,joint3,joint4,joint5,joint6,joint7"
        )
        for i in range(N):
            qd[0] = joint0[i]
            t, image, plant_state = sim.step(qd=qd)
            state_file.write("\n")
            line = ",".join([str(i), str(t)])
            line += "," + ",".join(plant_state[:7].astype(str))
            state_file.write(line)
            cv2.imwrite(str(os.path.join(dirname, f"image{i:03d}.png")), image)
