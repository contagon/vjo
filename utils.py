# From https://github.com/RussTedrake/manipulation/blob/master/manipulation/scenarios.py#L506
import os
import sys
import warnings

import numpy as np
from pydrake.all import (
    AbstractValue,
    Adder,
    AddMultibodyPlantSceneGraph,
    BallRpyJoint,
    BaseField,
    Box,
    CameraInfo,
    Capsule,
    ClippingRange,
    CoulombFriction,
    Cylinder,
    Demultiplexer,
    DepthImageToPointCloud,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    GeometryInstance,
    InverseDynamicsController,
    LeafSystem,
    MakeMultibodyStateToWsgStateSystem,
    MakePhongIllustrationProperties,
    MakeRenderEngineVtk,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PassThrough,
    PrismaticJoint,
    RenderCameraCore,
    RenderEngineVtkParams,
    RevoluteJoint,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SchunkWsgPositionController,
    SpatialInertia,
    Sphere,
    StateInterpolatorWithDiscreteDerivative,
    UnitInertia,
)


def AddMultibodyTriad(frame, scene_graph, length=0.25, radius=0.01, opacity=1.0):
    plant = frame.GetParentPlant()
    AddTriad(
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(frame.body().index()),
        scene_graph,
        length,
        radius,
        opacity,
        frame.GetFixedPoseInBodyFrame(),
    )


def AddTriad(
    source_id,
    frame_id,
    scene_graph,
    length=0.25,
    radius=0.01,
    opacity=1.0,
    X_FT=RigidTransform(),
    name="frame",
):
    """
    Adds illustration geometry representing the coordinate frame, with the
    x-axis drawn in red, the y-axis in green and the z-axis in blue. The axes
    point in +x, +y and +z directions, respectively.
    Args:
      source_id: The source registered with SceneGraph.
      frame_id: A geometry::frame_id registered with scene_graph.
      scene_graph: The SceneGraph with which we will register the geometry.
      length: the length of each axis in meters.
      radius: the radius of each axis in meters.
      opacity: the opacity of the coordinate axes, between 0 and 1.
      X_FT: a RigidTransform from the triad frame T to the frame_id frame F
      name: the added geometry will have names name + " x-axis", etc.
    """
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)
