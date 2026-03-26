# pygeometry_fake.py
import numpy
from typing import *

__all__ = [
    "Camera",
    "CameraParameters",
    "Pose",
    "ProjectionType",
    "Similarity",
    "absolute_pose_n_points",
    "absolute_pose_n_points_known_rotation",
    "absolute_pose_three_points",
    "compute_camera_mapping",
    "epipolar_angle_two_bearings_many",
    "essential_five_points",
    "essential_n_points",
    "point_refinement",
    "relative_pose_from_essential",
    "relative_pose_refinement",
    "relative_rotation_n_points",
    "triangulate_bearings_dlt",
    "triangulate_bearings_midpoint",
    "triangulate_two_bearings_midpoint",
    "triangulate_two_bearings_midpoint_many",
    "BROWN",
    "DUAL",
    "FISHEYE",
    "FISHEYE62",
    "FISHEYE624",
    "FISHEYE_OPENCV",
    "PERSPECTIVE",
    "RADIAL",
    "SIMPLE_RADIAL",
    "SPHERICAL",
    "aspect_ratio",
    "cx",
    "cy",
    "focal",
    "k1",
    "k2",
    "k3",
    "k4",
    "k5",
    "k6",
    "none",
    "p1",
    "p2",
    "s0",
    "s1",
    "s2",
    "s3",
    "transition",
]

# ------------------------
# Classes
# ------------------------
class Camera:
    def __init__(self) -> None:
        pass
    @staticmethod
    def create_brown(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_dual(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_fisheye(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_fisheye62(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_fisheye624(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_fisheye_opencv(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_perspective(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_radial(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_simple_radial(*args, **kwargs) -> "Camera": return Camera()
    @staticmethod
    def create_spherical() -> "Camera": return Camera()
    
    def get_K(self) -> numpy.ndarray: return numpy.zeros((3,3))
    def get_parameters_map(self) -> dict: return {}
    def get_parameters_types(self) -> list: return []
    def get_parameters_values(self) -> numpy.ndarray: return numpy.zeros(10)
    
    def normalized_to_pixel_coordinates(self, arg0: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros_like(arg0)
    def pixel_bearing(self, arg0: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros_like(arg0)
    def pixel_to_normalized_coordinates(self, arg0: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros_like(arg0)
    def project(self, arg0: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros_like(arg0)
    def project_many(self, arg0: numpy.ndarray) -> numpy.ndarray:
        return numpy.zeros_like(arg0)
    
    # Properties
    aspect_ratio = 0.0
    distortion = numpy.zeros(5)
    focal = 0.0
    height = 0
    id = ""
    k1 = 0.0
    k2 = 0.0
    k3 = 0.0
    k4 = 0.0
    k5 = 0.0
    k6 = 0.0
    p1 = 0.0
    p2 = 0.0
    principal_point = numpy.zeros(2)
    projection_type = "PERSPECTIVE"
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    transition = 0.0
    width = 0

class CameraParameters:
    def __init__(self, value: int = 0) -> None: pass
    name = ""
    value = 0

class Pose:
    def __init__(self, rotation: numpy.ndarray = None, translation: numpy.ndarray = None) -> None:
        self.rotation = rotation if rotation is not None else numpy.eye(3)
        self.translation = translation if translation is not None else numpy.zeros(3)
    def compose(self, arg0: "Pose") -> "Pose": return Pose()
    def inverse(self) -> "Pose": return Pose()
    def transform(self, arg0: numpy.ndarray) -> numpy.ndarray: return numpy.zeros_like(arg0)
    def transform_inverse(self, arg0: numpy.ndarray) -> numpy.ndarray: return numpy.zeros_like(arg0)

class ProjectionType:
    PERSPECTIVE = "PERSPECTIVE"
    BROWN = "BROWN"
    FISHEYE = "FISHEYE"
    FISHEYE_OPENCV = "FISHEYE_OPENCV"
    FISHEYE62 = "FISHEYE62"
    FISHEYE624 = "FISHEYE624"
    DUAL = "DUAL"
    SPHERICAL = "SPHERICAL"
    RADIAL = "RADIAL"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"

class Similarity:
    def __init__(self, rotation: numpy.ndarray = None, translation: numpy.ndarray = None, scale: float = 1.0) -> None:
        self.rotation = rotation if rotation is not None else numpy.eye(3)
        self.translation = translation if translation is not None else numpy.zeros(3)
        self.scale = scale
    def inverse(self) -> "Similarity": return Similarity()
    def transform(self, arg0: numpy.ndarray) -> numpy.ndarray: return numpy.zeros_like(arg0)

# ------------------------
# Functions
# ------------------------
def absolute_pose_n_points(arg0: numpy.ndarray, arg1: numpy.ndarray) -> numpy.ndarray: return numpy.zeros(6)
def absolute_pose_n_points_known_rotation(arg0: numpy.ndarray, arg1: numpy.ndarray) -> numpy.ndarray: return numpy.zeros(6)
def absolute_pose_three_points(arg0: numpy.ndarray, arg1: numpy.ndarray) -> list: return [numpy.zeros(6)]
def compute_camera_mapping(arg0: Camera, arg1: Camera, arg2: int, arg3: int) -> tuple: return (numpy.zeros((3,3)), numpy.zeros((3,3)))
def epipolar_angle_two_bearings_many(*args, **kwargs) -> numpy.ndarray: return numpy.zeros(10)
def essential_five_points(*args, **kwargs) -> list: return [numpy.zeros((3,3))]
def essential_n_points(*args, **kwargs) -> list: return [numpy.zeros((3,3))]
def point_refinement(*args, **kwargs) -> numpy.ndarray: return numpy.zeros(3)
def relative_pose_from_essential(*args, **kwargs) -> numpy.ndarray: return numpy.zeros((3,3))
def relative_pose_refinement(*args, **kwargs) -> numpy.ndarray: return numpy.zeros((3,3))
def relative_rotation_n_points(*args, **kwargs) -> numpy.ndarray: return numpy.zeros((3,3))
def triangulate_bearings_dlt(*args, **kwargs) -> tuple: return (False, numpy.zeros(3))
def triangulate_bearings_midpoint(*args, **kwargs) -> tuple: return (False, numpy.zeros(3))
def triangulate_two_bearings_midpoint(*args, **kwargs) -> tuple: return (False, numpy.zeros(3))
def triangulate_two_bearings_midpoint_many(*args, **kwargs) -> list: return [(False, numpy.zeros(3))]

# ------------------------
# Constants
# ------------------------
BROWN = ProjectionType.BROWN
DUAL = ProjectionType.DUAL
FISHEYE = ProjectionType.FISHEYE
FISHEYE62 = ProjectionType.FISHEYE62
FISHEYE624 = ProjectionType.FISHEYE624
FISHEYE_OPENCV = ProjectionType.FISHEYE_OPENCV
PERSPECTIVE = ProjectionType.PERSPECTIVE
RADIAL = ProjectionType.RADIAL
SIMPLE_RADIAL = ProjectionType.SIMPLE_RADIAL
SPHERICAL = ProjectionType.SPHERICAL

aspect_ratio = CameraParameters()
cx = CameraParameters()
cy = CameraParameters()
focal = CameraParameters()
k1 = CameraParameters()
k2 = CameraParameters()
k3 = CameraParameters()
k4 = CameraParameters()
k5 = CameraParameters()
k6 = CameraParameters()
none = CameraParameters()
p1 = CameraParameters()
p2 = CameraParameters()
s0 = CameraParameters()
s1 = CameraParameters()
s2 = CameraParameters()
s3 = CameraParameters()
transition = CameraParameters()