# Import some basic libraries and functions for this tutorial.
import time

import numpy as np
import pydot
from pydrake.geometry import (
    StartMeshcat,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
)
from pydrake.geometry.render import (
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
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
- Figure out how to pull joint/camera data at a specified interval
    - Do we want to ROS publish it to run live, or just pull it and save it somewhere?
- Make camera scene more interesting
- Probably need to make more interesting trajectories

Code Quality:
- Add TypeHints
- Test? That may be overkill for this
"""


class ArmSim:
    def __init__(self, viz=True, sim_time_step=0.0001):
        self.viz = viz

        if self.viz:
            self.meshcat = StartMeshcat()

        # Builder sets up the diagram
        # Diagram is all of the seperate systems combined
        self.builder = DiagramBuilder()
        # Plant = robot arm and anything else in the scene
        # Scene_graph = i don't really know?
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=sim_time_step
        )
        # Parser reads .sdf and .urdf files and puts them into the plant
        self.parser = Parser(self.plant)

        self.camera = None

    def add_arm(self, arm="iiwa7"):
        """Add robot arm to plant"""
        if arm == "iiwa7":
            sdf = (
                "package://drake/manipulation/models/"
                "iiwa_description/iiwa7/iiwa7_with_box_collision.sdf"
            )
            self.N = 7
            self.link_names = lambda i: f"iiwa_link_{i}"
        else:
            raise TypeError("Invalid robot arm type")

        (self.iiwa,) = self.parser.AddModels(url=sdf)

        # Connect arm to world origin
        L0 = self.plant.GetFrameByName(self.link_names(0))
        self.plant.WeldFrames(self.plant.world_frame(), L0)

    def plant_finalize(self):
        """Finalize the plant - means we're not adding anything else to ti"""
        # Finalize the plant after loading all of the robot arm.
        self.plant.Finalize()

    def add_frame(self, frame, length=0.25, radius=0.01):
        """Helper to visualize frames in the simulation
        
        If integer, defaults to that frame on the arm. If string
        gets the frame with that name.
        """
        if type(frame) is int:
            frame = self.plant.GetFrameByName(self.link_names(frame))
        elif type(frame) is str:
            frame = self.plant.GetFrameByName(frame)

        AddMultibodyTriad(frame, self.scene_graph, length=length, radius=radius)

    def add_mesh(self, model : str, frame_name : str, offset : RigidTransform):
        sim.parser.AddModels(model)
        self.plant.WeldFrames(
                    self.plant.world_frame(),
                    self.plant.GetFrameByName(frame_name),
                    offset
                )

    def add_camera(self):
        """Add camera attached to end effector"""
        # Make renderer for cameras
        renderer_name = "renderer"
        self.scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

        # Add in sensors
        intrinsics = CameraInfo(
            width=640,
            height=480,
            fov_y=np.pi / 4,
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
            RigidTransform(RollPitchYaw([0,0,np.pi/2]), [0, 0, 0.1]),
            color_camera=color_camera,
            depth_camera=depth_camera,
        )

        # Connect with outputs
        self.builder.AddSystem(self.camera)
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.camera.query_object_input_port(),
        )

        self.builder.ExportOutput(self.camera.color_image_output_port(), "color_image")
        # self.builder.ExportOutput(self.camera.depth_image_32F_output_port(), "depth_image")

    def add_controller(self):
        """Add a PID controller to control arm"""
        cont = InverseDynamicsController(
            self.plant,
            kp=np.full(self.N, 10),
            kd=np.full(self.N, 5),
            ki=np.full(self.N, 1),
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

        # TODO: Put this elsewhere?
        # Make input of entire system the controller input
        self.builder.ExportInput(self.controller.get_input_port_desired_state())
        # Make the arm state an output from the diagram.
        self.builder.ExportOutput(self.plant.get_state_output_port())

    def sim_setup(self, wait_load=2):
        """Setup actual simulation using diagram"""
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

        # It takes a second for meshcat to load
        time.sleep(wait_load)

    def save_diagram(self, file):
        """Save diagram. MUST call sim_setup first"""
        svg = pydot.graph_from_dot_data(self.diagram.GetGraphvizString(max_depth=2))[
            0
        ].create_svg()
        with open(file, "wb") as f:
            f.write(svg)

    def sim_run(self, q0=None, qd=None):
        """Run simple simulation"""
        # Parse inputs / desired
        if q0 is None:
            q0 = np.zeros(7)
        if qd is None:
            qd = np.zeros(7)

        # Append velocities of 0s
        q0 = np.append(q0, np.zeros(7))
        qd = np.append(qd, np.zeros(7))

        # Set starting state
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, self.context
        )
        plant_context.get_mutable_discrete_state_vector().SetFromVector(q0)

        # Fix the desired joint angles
        # There is a way to make these change through the simulation that we might want to figure out
        self.diagram.get_input_port(0).FixValue(self.context, qd)

        # TODO: Camera stuff
        if self.camera is not None:
            image = self.diagram.GetOutputPort("color_image").Eval(self.context)

        if self.viz:
            self.visualizer.StartRecording()
        self.simulator.AdvanceTo(5.0)


if __name__ == "__main__":
    sim = ArmSim(viz=True)
    # Setup everything in environment
    sim.add_arm()
    # Add mustard bottle in
    sim.add_mesh("meshes/cylinder.sdf", "cylinder_link", RigidTransform(RollPitchYaw([0.1,0,0]), [1,0,0.75]))
    # Visualizer end effector pose
    sim.add_frame(7)
    sim.plant_finalize()

    sim.add_controller()
    sim.add_camera()

    # Get sim ready
    q0 = np.array([0, np.pi / 2, 0, np.pi / 2, 0, np.pi / 2, 0])
    qd = np.zeros(7)
    sim.sim_setup(wait_load=3)
    # sim.save_diagram("diagram.svg")
    sim.sim_run(q0=q0, qd=qd)
