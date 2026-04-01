# pygeometry_fake.py
# Pure Python implementation replacing C++ pybind11 pygeometry bindings

import numpy
import numpy as np
from typing import *
import cv2

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


# ---------------------------------------------------------------------------
# ProjectionType enum
# ---------------------------------------------------------------------------

class ProjectionType:
    PERSPECTIVE  = "PERSPECTIVE"
    BROWN        = "BROWN"
    FISHEYE      = "FISHEYE"
    FISHEYE_OPENCV = "FISHEYE_OPENCV"
    FISHEYE62    = "FISHEYE62"
    FISHEYE624   = "FISHEYE624"
    DUAL         = "DUAL"
    SPHERICAL    = "SPHERICAL"
    RADIAL       = "RADIAL"
    SIMPLE_RADIAL = "SIMPLE_RADIAL"


# ---------------------------------------------------------------------------
# CameraParameters (enum-like constants)
# ---------------------------------------------------------------------------

class CameraParameters:
    def __init__(self, value: int = 0) -> None:
        self.value = value
        self.name  = ""


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """
    Pure-Python camera model.
    Supports PERSPECTIVE, BROWN, FISHEYE, FISHEYE_OPENCV,
    FISHEYE62, FISHEYE624, DUAL, SPHERICAL, RADIAL, SIMPLE_RADIAL.
    """

    def __init__(self) -> None:
        self.projection_type  = "perspective"
        self.focal            = 1.0
        self.aspect_ratio     = 1.0
        self.principal_point  = numpy.zeros(2, dtype=numpy.float64)
        self.k1  = 0.0
        self.k2  = 0.0
        self.k3  = 0.0
        self.k4  = 0.0
        self.k5  = 0.0
        self.k6  = 0.0
        self.p1  = 0.0
        self.p2  = 0.0
        self.s0  = 0.0
        self.s1  = 0.0
        self.s2  = 0.0
        self.s3  = 0.0
        self.transition = 0.0
        self.distortion = numpy.zeros(5, dtype=numpy.float64)
        self.width  = 0
        self.height = 0
        self.id     = ""

    # ------------------------------------------------------------------ #
    # Factory methods
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_perspective(focal: float, k1: float, k2: float) -> "Camera":
        c = Camera()
        c.projection_type = "perspective"
        c.focal = focal
        c.k1 = k1
        c.k2 = k2
        c.distortion = numpy.array([k1, k2, 0, 0, 0], dtype=numpy.float64)
        return c

    @staticmethod
    def create_brown(focal: float, aspect_ratio: float,
                     principal_point: numpy.ndarray,
                     distortion: numpy.ndarray) -> "Camera":
        if len(distortion) != 5:
            raise RuntimeError("Brown model needs exactly 5 distortion coefficients")
        c = Camera()
        c.projection_type = "brown"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = distortion[0]; c.k2 = distortion[1]; c.k3 = distortion[2]
        c.p1 = distortion[3]; c.p2 = distortion[4]
        c.distortion = numpy.array(distortion, dtype=numpy.float64)
        return c

    @staticmethod
    def create_fisheye(focal: float, k1: float, k2: float) -> "Camera":
        c = Camera()
        c.projection_type = "fisheye"
        c.focal = focal
        c.k1 = k1
        c.k2 = k2
        c.distortion = numpy.array([k1, k2, 0, 0], dtype=numpy.float64)
        return c

    @staticmethod
    def create_fisheye_opencv(focal: float, aspect_ratio: float,
                               principal_point: numpy.ndarray,
                               distortion: numpy.ndarray) -> "Camera":
        if len(distortion) != 4:
            raise RuntimeError("FisheyeOpenCV needs 4 distortion coefficients")
        c = Camera()
        c.projection_type = "fisheye_opencv"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = distortion[0]; c.k2 = distortion[1]
        c.k3 = distortion[2]; c.k4 = distortion[3]
        c.distortion = numpy.array(distortion, dtype=numpy.float64)
        return c

    @staticmethod
    def create_fisheye62(focal: float, aspect_ratio: float,
                          principal_point: numpy.ndarray,
                          distortion: numpy.ndarray) -> "Camera":
        if len(distortion) != 8:
            raise RuntimeError("Fisheye62 needs 8 distortion coefficients")
        c = Camera()
        c.projection_type = "fisheye62"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = distortion[0]; c.k2 = distortion[1]; c.k3 = distortion[2]
        c.k4 = distortion[3]; c.k5 = distortion[4]; c.k6 = distortion[5]
        c.p1 = distortion[6]; c.p2 = distortion[7]
        c.distortion = numpy.array(distortion, dtype=numpy.float64)
        return c

    @staticmethod
    def create_fisheye624(focal: float, aspect_ratio: float,
                           principal_point: numpy.ndarray,
                           distortion: numpy.ndarray) -> "Camera":
        if len(distortion) != 12:
            raise RuntimeError("Fisheye624 needs 12 distortion coefficients")
        c = Camera()
        c.projection_type = "fisheye624"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = distortion[0];  c.k2 = distortion[1];  c.k3 = distortion[2]
        c.k4 = distortion[3];  c.k5 = distortion[4];  c.k6 = distortion[5]
        c.p1 = distortion[6];  c.p2 = distortion[7]
        c.s0 = distortion[8];  c.s1 = distortion[9]
        c.s2 = distortion[10]; c.s3 = distortion[11]
        c.distortion = numpy.array(distortion, dtype=numpy.float64)
        return c

    @staticmethod
    def create_dual(transition: float, focal: float,
                    k1: float, k2: float) -> "Camera":
        c = Camera()
        c.projection_type = "dual"
        c.transition = transition
        c.focal = focal
        c.k1 = k1
        c.k2 = k2
        return c

    @staticmethod
    def create_spherical() -> "Camera":
        c = Camera()
        c.projection_type = "spherical"
        return c

    @staticmethod
    def create_radial(focal: float, aspect_ratio: float,
                      principal_point: numpy.ndarray,
                      distortion: numpy.ndarray) -> "Camera":
        if len(distortion) != 2:
            raise RuntimeError("Radial model needs exactly 2 distortion coefficients")
        c = Camera()
        c.projection_type = "radial"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = distortion[0]
        c.k2 = distortion[1]
        c.distortion = numpy.array(distortion, dtype=numpy.float64)
        return c

    @staticmethod
    def create_simple_radial(focal: float, aspect_ratio: float,
                              principal_point: numpy.ndarray,
                              k1: float) -> "Camera":
        c = Camera()
        c.projection_type = "simple_radial"
        c.focal           = focal
        c.aspect_ratio    = aspect_ratio
        c.principal_point = numpy.array(principal_point, dtype=numpy.float64)
        c.k1 = k1
        c.distortion = numpy.array([k1], dtype=numpy.float64)
        return c

    @staticmethod
    def is_panorama(projection_type: str) -> bool:
        return projection_type.lower() in ("spherical", "equirectangular")

    # ------------------------------------------------------------------ #
    # Intrinsic matrix
    # ------------------------------------------------------------------ #

    def get_K(self) -> numpy.ndarray:
        K = numpy.zeros((3, 3), dtype=numpy.float64)
        K[0, 0] = self.focal
        K[1, 1] = self.focal * self.aspect_ratio
        K[0, 2] = self.principal_point[0]
        K[1, 2] = self.principal_point[1]
        K[2, 2] = 1.0
        return K

    def get_K_in_pixel(self) -> numpy.ndarray:
        """K matrix with coords in pixels (width/height must be set)."""
        K = self.get_K().copy()
        size = max(self.width, self.height) if self.width and self.height else 1
        K[0, 0] *= size;  K[1, 1] *= size
        K[0, 2]  = K[0, 2] * size + self.width  / 2.0
        K[1, 2]  = K[1, 2] * size + self.height / 2.0
        return K

    def get_parameters_map(self) -> dict:
        return {}

    def get_parameters_types(self) -> list:
        return []

    def get_parameters_values(self) -> numpy.ndarray:
        return numpy.zeros(10)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _undistort_normalized(self, x: float, y: float) -> Tuple[float, float]:
        """Remove radial + tangential distortion from normalised coords."""
        pt = numpy.array([[x, y]], dtype=numpy.float64)
        K  = self.get_K()
        dist = numpy.array(
            [self.k1, self.k2, self.p1, self.p2, self.k3],
            dtype=numpy.float64
        )
        undist = cv2.undistortPoints(
            pt.reshape(1, 1, 2), K, dist, P=None
        )
        return float(undist[0, 0, 0]), float(undist[0, 0, 1])

    def _distort_normalized(self, x: float, y: float) -> Tuple[float, float]:
        """Apply radial + tangential distortion to normalised coords."""
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r2 * r4
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
        xd = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        yd = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        return xd, yd

    # ------------------------------------------------------------------ #
    # Projection:  3-D point → 2-D pixel
    # ------------------------------------------------------------------ #

    def project(self, point: numpy.ndarray) -> numpy.ndarray:
        """Project one 3-D point to 2-D pixel coords."""
        point = numpy.asarray(point, dtype=numpy.float64)
        pt    = self._project_normalised(point)
        return self._normalised_to_pixel(pt[0], pt[1])

    def project_many(self, points: numpy.ndarray) -> numpy.ndarray:
        """Project (N,3) array of 3-D points to (N,2) pixel coords."""
        points = numpy.atleast_2d(numpy.asarray(points, dtype=numpy.float64))
        return numpy.array([self.project(p) for p in points])

    def _project_normalised(self, point: numpy.ndarray) -> Tuple[float, float]:
        """3-D → normalised (distorted) image coords."""
        pt = point.ravel()
        if self.projection_type.lower() == "spherical":
            lon = numpy.arctan2(pt[0], pt[2])
            lat = numpy.arctan2(-pt[1], numpy.sqrt(pt[0]**2 + pt[2]**2))
            return float(lon / (2 * numpy.pi)), float(-lat / (2 * numpy.pi))
        if pt[2] == 0:
            return 0.0, 0.0
        x = pt[0] / pt[2]
        y = pt[1] / pt[2]
        xd, yd = self._distort_normalized(x, y)
        return xd, yd

    def _normalised_to_pixel(self, xd: float, yd: float) -> numpy.ndarray:
        px = xd * self.focal + self.principal_point[0]
        py = yd * self.focal * self.aspect_ratio + self.principal_point[1]
        return numpy.array([px, py], dtype=numpy.float64)

    # ------------------------------------------------------------------ #
    # Bearing:  2-D pixel → unit 3-D direction vector
    # ------------------------------------------------------------------ #

    def pixel_bearing(self, point: numpy.ndarray) -> numpy.ndarray:
        """Convert one pixel (2,) to a unit bearing vector (3,)."""
        point = numpy.asarray(point, dtype=numpy.float64).ravel()

        if self.projection_type.lower() == "spherical":
            lon =  point[0] * 2 * numpy.pi
            lat = -point[1] * 2 * numpy.pi
            cos_lat = numpy.cos(lat)
            return numpy.array([
                cos_lat * numpy.sin(lon),
                -numpy.sin(lat),
                cos_lat * numpy.cos(lon),
            ], dtype=numpy.float64)

        # normalised (distorted) coords
        xd = (point[0] - self.principal_point[0]) / self.focal
        yd = (point[1] - self.principal_point[1]) / (self.focal * self.aspect_ratio)

        # undistort
        x, y = self._undistort_normalized(xd, yd)

        # to unit vector
        norm = numpy.sqrt(x * x + y * y + 1.0)
        return numpy.array([x / norm, y / norm, 1.0 / norm], dtype=numpy.float64)

    def pixel_bearing_many(self, points: numpy.ndarray) -> numpy.ndarray:
        """Convert (N,2) pixel array to (N,3) unit bearing vectors."""
        points = numpy.atleast_2d(numpy.asarray(points, dtype=numpy.float64))
        return numpy.array([self.pixel_bearing(p) for p in points])

    # ------------------------------------------------------------------ #
    # Coordinate conversion helpers
    # ------------------------------------------------------------------ #


    def bearing(self, point2d):
        """
        Back-project a 2D normalized image point to a unit bearing vector.
        Mirrors C++ Camera::Bearing() dispatch.
        """
        import numpy as _np
        x, y = float(point2d[0]), float(point2d[1])
        pt = self.projection_type

        if self.projection_type in ("perspective",):
            f = float(self.focal)
            b = _np.array([x / f, y / f, 1.0], dtype=_np.float64)

        elif self.projection_type == "brown":
            f  = float(self.focal)
            ar = float(getattr(self, "aspect_ratio", 1.0))
            cx = float(self.principal_point[0]) if hasattr(self, "principal_point") else 0.0
            cy = float(self.principal_point[1]) if hasattr(self, "principal_point") else 0.0
            xn = (x - cx) / f
            yn = (y - cy) / (f * ar)
            b = _np.array([xn, yn, 1.0], dtype=_np.float64)

        elif self.projection_type in ("fisheye", "fisheye_opencv",
                                       "fisheye62", "fisheye624"):
            f = float(self.focal)
            r = _np.sqrt(x * x + y * y)
            if r < 1e-10:
                b = _np.array([0.0, 0.0, 1.0], dtype=_np.float64)
            else:
                theta = _np.arctan2(r, f)
                b = _np.array([
                    _np.sin(theta) * x / r,
                    _np.sin(theta) * y / r,
                    _np.cos(theta),
                ], dtype=_np.float64)

        elif self.projection_type in ("spherical", "equirectangular"):
            lon = x * 2.0 * _np.pi
            lat = -y * 2.0 * _np.pi
            b = _np.array([
                _np.cos(lat) * _np.sin(lon),
                -_np.sin(lat),
                _np.cos(lat) * _np.cos(lon),
            ], dtype=_np.float64)

        else:
            b = _np.array([x, y, 1.0], dtype=_np.float64)

        norm = _np.linalg.norm(b)
        if norm < 1e-10:
            return _np.array([0.0, 0.0, 1.0], dtype=_np.float64)
        return b / norm


    def project(self, point3d):
        import numpy as _np
        p = _np.array(point3d, dtype=_np.float64)
        if abs(p[2]) < 1e-10:
            return _np.array([0.0, 0.0], dtype=_np.float64)

        if self.projection_type == "perspective":
            f = float(self.focal)
            return _np.array([f * p[0] / p[2], f * p[1] / p[2]], dtype=_np.float64)

        else:
            return _np.array([p[0]/p[2], p[1]/p[2]], dtype=_np.float64)

    def pixel_to_normalized_coordinates(self, point: numpy.ndarray) -> numpy.ndarray:
        """Pixel → undistorted normalised coords (2,)."""
        point = numpy.asarray(point, dtype=numpy.float64).ravel()
        xd = (point[0] - self.principal_point[0]) / self.focal
        yd = (point[1] - self.principal_point[1]) / (self.focal * self.aspect_ratio)
        x, y = self._undistort_normalized(xd, yd)
        return numpy.array([x, y], dtype=numpy.float64)

    def normalized_to_pixel_coordinates(self, point: numpy.ndarray) -> numpy.ndarray:
        """Undistorted normalised coords → pixel (2,)."""
        point = numpy.asarray(point, dtype=numpy.float64).ravel()
        xd, yd = self._distort_normalized(point[0], point[1])
        return self._normalised_to_pixel(xd, yd)

    def __repr__(self):
        return (f"Camera(type={self.projection_type}, "
                f"focal={self.focal:.4f}, id={self.id!r})")


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

class Pose:
    """
    Rigid body pose: rotation (Rodrigues vector) + translation.
    Transforms a 3-D world point P to camera coords:  X = R*P + t
    """

    def __init__(self,
                 rotation:    numpy.ndarray = None,
                 translation: numpy.ndarray = None) -> None:
        self.rotation    = numpy.array(rotation,    dtype=numpy.float64).ravel() \
                           if rotation    is not None else numpy.zeros(3)
        self.translation = numpy.array(translation, dtype=numpy.float64).ravel() \
                           if translation is not None else numpy.zeros(3)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    # def get_rotation_matrix(self) -> numpy.ndarray:
    #     """Rodrigues vector → 3×3 rotation matrix."""
    #     R, _ = cv2.Rodrigues(self.rotation.astype(numpy.float64))
    #     return R

    def get_rotation_matrix(self):
        rot = numpy.array(self.rotation, dtype=numpy.float64)
        R, _ = cv2.Rodrigues(rot)
        return R

    def set_rotation_matrix(self, R: numpy.ndarray):
        r, _ = cv2.Rodrigues(numpy.array(R, dtype=numpy.float64))
        self.rotation = r.ravel()

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #

    # def get_origin(self) -> numpy.ndarray:
    #     """Camera centre in world coordinates:  C = -R^T * t"""
    #     R = self.get_rotation_matrix()
    #     return -(R.T @ self.translation)

    def get_origin(self):
        R = self.get_rotation_matrix()
        t = numpy.array(self.translation, dtype=numpy.float64)
        return -(R.T @ t)

    def set_origin(self, origin):
        R = self.get_rotation_matrix()
        self.translation = -R @ numpy.array(origin, dtype=numpy.float64)

    # def set_origin(self, origin: numpy.ndarray):
    #     R = self.get_rotation_matrix()
    #     self.translation = -R @ numpy.array(origin, dtype=numpy.float64)

    def get_Rt(self):
        R = self.get_rotation_matrix()
        t = numpy.array(self.translation, dtype=numpy.float64)
        return numpy.hstack([R, t.reshape(3, 1)])

    # ------------------------------------------------------------------ #
    # Transform
    # ------------------------------------------------------------------ #

    # def transform(self, point: numpy.ndarray) -> numpy.ndarray:
    #     """World → camera:  X = R*P + t"""
    #     R = self.get_rotation_matrix()
    #     return R @ numpy.array(point, dtype=numpy.float64) + self.translation

    def transform(self, point):
        R = self.get_rotation_matrix()
        t = numpy.array(self.translation, dtype=numpy.float64)
        return R @ numpy.array(point, dtype=numpy.float64) + t

    # def transform_many(self, points: numpy.ndarray) -> numpy.ndarray:
    #     """World → camera for (N,3) array."""
    #     R      = self.get_rotation_matrix()
    #     points = numpy.atleast_2d(numpy.array(points, dtype=numpy.float64))
    #     return (R @ points.T).T + self.translation

    def transform_many(self, points):
        R      = self.get_rotation_matrix()
        t      = numpy.array(self.translation, dtype=numpy.float64)
        points = numpy.atleast_2d(numpy.array(points, dtype=numpy.float64))
        return (R @ points.T).T + t

    # def transform_inverse(self, point: numpy.ndarray) -> numpy.ndarray:
    #     """Camera → world:  P = R^T*(X - t)"""
    #     R = self.get_rotation_matrix()
    #     return R.T @ (numpy.array(point, dtype=numpy.float64) - self.translation)

    def transform_inverse(self, point):
        R = self.get_rotation_matrix()
        t = numpy.array(self.translation, dtype=numpy.float64)
        return R.T @ (numpy.array(point, dtype=numpy.float64) - t)

    # def transform_inverse_many(self, points: numpy.ndarray) -> numpy.ndarray:
    #     R      = self.get_rotation_matrix()
    #     points = numpy.atleast_2d(numpy.array(points, dtype=numpy.float64))
    #     return (R.T @ (points - self.translation).T).T

    def transform_inverse_many(self, points):
        R      = self.get_rotation_matrix()
        t      = numpy.array(self.translation, dtype=numpy.float64)
        points = numpy.atleast_2d(numpy.array(points, dtype=numpy.float64))
        return (R.T @ (points - t).T).T
    
    # ------------------------------------------------------------------ #
    # Composition / inversion
    # ------------------------------------------------------------------ #

    # def compose(self, other: "Pose") -> "Pose":
    #     """self ∘ other:  applies other first, then self."""
    #     R1 = self.get_rotation_matrix()
    #     R2 = other.get_rotation_matrix()
    #     R  = R1 @ R2
    #     t  = R1 @ other.translation + self.translation
    #     p  = Pose()
    #     p.set_rotation_matrix(R)
    #     p.translation = t
    #     return p

    def compose(self, other):
        R1 = self.get_rotation_matrix()
        R2 = other.get_rotation_matrix()
        R  = R1 @ R2
        t  = R1 @ numpy.array(other.translation, dtype=numpy.float64) + numpy.array(self.translation, dtype=numpy.float64)
        p  = Pose()
        p.set_rotation_matrix(R)
        p.translation = t
        return p

    # def inverse(self) -> "Pose":
    #     """Invert the pose."""
    #     R    = self.get_rotation_matrix()
    #     R_inv = R.T
    #     t_inv = -R_inv @ self.translation
    #     p = Pose()
    #     p.set_rotation_matrix(R_inv)
    #     p.translation = t_inv
    #     return p

    def inverse(self):
        p = Pose()
        R = self.get_rotation_matrix()
        t = numpy.array(self.translation, dtype=numpy.float64)
        p.rotation = list((-R.T @ t))
        p.translation = list(-(R.T @ t))
        return p
    
    def __repr__(self):
        return f"Pose(r={self.rotation}, t={self.translation})"


# ---------------------------------------------------------------------------
# Similarity  (rotation + translation + scale)
# ---------------------------------------------------------------------------

class Similarity:
    """Similarity transform: X' = s * R * X + t"""

    def __init__(self,
                 rotation:    numpy.ndarray = None,
                 translation: numpy.ndarray = None,
                 scale: float = 1.0) -> None:
        self.rotation    = numpy.array(rotation,    dtype=numpy.float64).ravel() \
                           if rotation    is not None else numpy.zeros(3)
        self.translation = numpy.array(translation, dtype=numpy.float64).ravel() \
                           if translation is not None else numpy.zeros(3)
        self.scale = float(scale)

    def _R(self) -> numpy.ndarray:
        R, _ = cv2.Rodrigues(self.rotation.astype(numpy.float64))
        return R

    def transform(self, point: numpy.ndarray) -> numpy.ndarray:
        R = self._R()
        return self.scale * (R @ numpy.array(point, dtype=numpy.float64)) + self.translation

    def transform_many(self, points: numpy.ndarray) -> numpy.ndarray:
        R = self._R()
        pts = numpy.atleast_2d(numpy.array(points, dtype=numpy.float64))
        return self.scale * (R @ pts.T).T + self.translation

    def inverse(self) -> "Similarity":
        R     = self._R()
        s_inv = 1.0 / self.scale
        R_inv = R.T
        t_inv = -s_inv * (R_inv @ self.translation)
        sim   = Similarity()
        r_vec, _ = cv2.Rodrigues(R_inv)
        sim.rotation    = r_vec.ravel()
        sim.translation = t_inv
        sim.scale       = s_inv
        return sim

    def __repr__(self):
        return f"Similarity(scale={self.scale:.4f})"


# ---------------------------------------------------------------------------
# Geometry functions
# ---------------------------------------------------------------------------

def _to_bearing(p: numpy.ndarray) -> numpy.ndarray:
    """Normalise a vector to unit length."""
    n = numpy.linalg.norm(p)
    return p / n if n > 1e-10 else p


def triangulate_two_bearings_midpoint(
    orig1: numpy.ndarray, bear1: numpy.ndarray,
    orig2: numpy.ndarray, bear2: numpy.ndarray,
) -> Tuple[bool, numpy.ndarray]:
    """Triangulate one point from two rays (midpoint method).
    orig1, orig2: camera origins (3,)
    bear1, bear2: unit bearing vectors (3,)
    """
    d1 = numpy.array(bear1, dtype=numpy.float64).ravel()
    d2 = numpy.array(bear2, dtype=numpy.float64).ravel()
    o1 = numpy.array(orig1, dtype=numpy.float64).ravel()
    o2 = numpy.array(orig2, dtype=numpy.float64).ravel()

    # normalise directions
    d1 = d1 / (numpy.linalg.norm(d1) + 1e-10)
    d2 = d2 / (numpy.linalg.norm(d2) + 1e-10)

    d1d1 = float(numpy.dot(d1, d1))
    d1d2 = float(numpy.dot(d1, d2))
    d2d2 = float(numpy.dot(d2, d2))
    det  = d1d1 * d2d2 - d1d2 * d1d2

    if abs(det) < 1e-10:
        return False, numpy.zeros(3)

    r    = o2 - o1
    A    = numpy.array([[d1d1, -d1d2],
                        [d1d2, -d2d2]], dtype=numpy.float64)
    b_vec = numpy.array([float(numpy.dot(d1, r)),
                         float(numpy.dot(d2, r))], dtype=numpy.float64)

    try:
        t = numpy.linalg.solve(A, b_vec)
    except numpy.linalg.LinAlgError:
        return False, numpy.zeros(3)

    p1    = o1 + t[0] * d1
    p2    = o2 + t[1] * d2
    point = (p1 + p2) * 0.5
    ok    = bool(t[0] > 0 and t[1] > 0)
    return ok, point


def triangulate_two_bearings_midpoint_many(
    b1: numpy.ndarray,
    b2: numpy.ndarray,
    R:  numpy.ndarray,
    t:  numpy.ndarray,
) -> list:
    """
    Triangulate N points từ hai sets of bearing vectors.
    Gọi từ matching.py:
        p = pygeometry.triangulate_two_bearings_midpoint_many(b1, b2, R, t)
        good_idx = [i for i in range(len(p)) if p[i][0]]
        points   = np.array([p[i][1] for i in range(len(p)) if p[i][0]])

    b1: (N,3) bearings in camera 1 frame
    b2: (N,3) bearings in camera 2 frame
    R:  (3,3) rotation  camera 2 relative to camera 1
    t:  (3,)  translation camera 2 relative to camera 1

    Returns: List of (bool, ndarray(3,)) — một tuple per point
    """
    b1 = numpy.atleast_2d(numpy.array(b1, dtype=numpy.float64))
    b2 = numpy.atleast_2d(numpy.array(b2, dtype=numpy.float64))
    R  = numpy.array(R,  dtype=numpy.float64)
    t  = numpy.array(t,  dtype=numpy.float64).ravel()

    # camera 1 origin = [0,0,0]
    # camera 2 origin = -R^T * t  (world coords)
    orig1 = numpy.zeros(3)
    orig2 = -R.T @ t

    # rotate b2 bearings từ camera 2 frame sang world frame
    b2_world = (R.T @ b2.T).T   # (N,3)

    result = []
    for bear1, bear2 in zip(b1, b2_world):
        ok, pt = triangulate_two_bearings_midpoint(orig1, bear1, orig2, bear2)
        result.append((ok, pt))

    return result

def triangulate_bearings_midpoint(
    origins:               numpy.ndarray,
    bearings:              numpy.ndarray,
    thresholds:            numpy.ndarray = None,   # ← thêm arg thứ 3
    min_ray_angle_radians: float = 0.01,           # ← đổi tên
    min_depth:             float = 0.001,          # ← thêm arg thứ 5
) -> Tuple[bool, numpy.ndarray]:
    """Triangulate from N rays using midpoint method."""
    origins  = numpy.atleast_2d(numpy.array(origins,  dtype=numpy.float64))
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    n = len(origins)
    if n < 2:
        return False, numpy.zeros(3)

    # Build linear system (DLT midpoint)
    A = numpy.zeros((3, 3), dtype=numpy.float64)
    b = numpy.zeros(3,      dtype=numpy.float64)
    for o, d in zip(origins, bearings):
        d = d / (numpy.linalg.norm(d) + 1e-10)
        M = numpy.eye(3) - numpy.outer(d, d)
        A += M
        b += M @ o
    try:
        point = numpy.linalg.solve(A, b)
    except numpy.linalg.LinAlgError:
        return False, numpy.zeros(3)

    # Kiểm tra min_depth — điểm phải ở phía trước tất cả cameras
    for o, d in zip(origins, bearings):
        d = d / (numpy.linalg.norm(d) + 1e-10)
        depth = numpy.dot(point - o, d)
        if depth < min_depth:
            return False, numpy.zeros(3)

    # Kiểm tra min ray angle giữa các cặp bearing
    if n >= 2:
        for i in range(n):
            for j in range(i + 1, n):
                di = bearings[i] / (numpy.linalg.norm(bearings[i]) + 1e-10)
                dj = bearings[j] / (numpy.linalg.norm(bearings[j]) + 1e-10)
                cos_angle = numpy.clip(numpy.dot(di, dj), -1.0, 1.0)
                angle = numpy.arccos(cos_angle)
                if angle >= min_ray_angle_radians:
                    return True, point
        return False, numpy.zeros(3)

    return True, point


def triangulate_bearings_dlt(
    poses:    List,
    bearings: numpy.ndarray,
) -> Tuple[bool, numpy.ndarray]:
    """DLT triangulation from list of Pose objects and unit bearings."""
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    A_rows = []
    for pose, b in zip(poses, bearings):
        R = pose.get_rotation_matrix()
        t = pose.translation
        P = numpy.hstack([R, t.reshape(3, 1)])   # 3×4
        bx, by, bz = b[0], b[1], b[2]
        A_rows.append(bx * P[2] - bz * P[0])
        A_rows.append(by * P[2] - bz * P[1])

    if not A_rows:
        return False, numpy.zeros(3)

    A = numpy.array(A_rows, dtype=numpy.float64)
    _, _, Vt = numpy.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return False, numpy.zeros(3)
    point = X[:3] / X[3]
    return True, point


def relative_pose_from_essential(
    E: numpy.ndarray,
    b1: numpy.ndarray,
    b2: numpy.ndarray,
) -> numpy.ndarray:
    """Recover R,t from essential matrix given bearing correspondences."""
    E  = numpy.array(E,  dtype=numpy.float64)
    b1 = numpy.atleast_2d(numpy.array(b1, dtype=numpy.float64))
    b2 = numpy.atleast_2d(numpy.array(b2, dtype=numpy.float64))

    _, Rs, ts, _ = cv2.recoverPose(E, b1[:, :2], b2[:, :2])
    pose = Pose()
    pose.set_rotation_matrix(Rs)
    pose.translation = ts.ravel()
    return pose


def essential_five_points(
    b1: numpy.ndarray,
    b2: numpy.ndarray,
) -> List[numpy.ndarray]:
    """Five-point essential matrix estimation."""
    b1 = numpy.atleast_2d(numpy.array(b1, dtype=numpy.float64))
    b2 = numpy.atleast_2d(numpy.array(b2, dtype=numpy.float64))
    E, mask = cv2.findEssentialMat(
        b1[:, :2], b2[:, :2],
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=0.01,
    )
    return [E] if E is not None else [numpy.zeros((3, 3))]


def essential_n_points(
    b1: numpy.ndarray,
    b2: numpy.ndarray,
) -> List[numpy.ndarray]:
    """N-point essential matrix estimation (uses 5-point internally)."""
    return essential_five_points(b1, b2)


def relative_pose_refinement(
    init_pose:   numpy.ndarray,
    b1:          numpy.ndarray,
    b2:          numpy.ndarray,
    iterations:  int = 100,
    threshold:   float = 1e-6,
) -> numpy.ndarray:
    """Refine relative pose (stub — returns init_pose unchanged)."""
    return init_pose


def relative_rotation_n_points(
    b1: numpy.ndarray,
    b2: numpy.ndarray,
) -> numpy.ndarray:
    """Estimate pure rotation from bearing correspondences."""
    b1 = numpy.atleast_2d(numpy.array(b1, dtype=numpy.float64))
    b2 = numpy.atleast_2d(numpy.array(b2, dtype=numpy.float64))
    # Kabsch / SVD rotation
    H  = b1.T @ b2
    U, _, Vt = numpy.linalg.svd(H)
    R  = Vt.T @ U.T
    if numpy.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    r, _ = cv2.Rodrigues(R)
    return r.ravel()


def absolute_pose_three_points(
    bearings: numpy.ndarray,
    points:   numpy.ndarray,
) -> List[numpy.ndarray]:
    """P3P pose estimation."""
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    points   = numpy.atleast_2d(numpy.array(points,   dtype=numpy.float64))
    try:
        _, rvecs, tvecs = cv2.solvePnP(
            points.astype(numpy.float32),
            bearings[:, :2].astype(numpy.float32),
            numpy.eye(3, dtype=numpy.float32),
            None,
            flags=cv2.SOLVEPNP_P3P,
        )
        pose = Pose(rvecs.ravel(), tvecs.ravel())
        return [pose]
    except Exception:
        return [Pose()]


def absolute_pose_n_points(
    bearings: numpy.ndarray,
    points:   numpy.ndarray,
) -> numpy.ndarray:
    """PnP pose estimation, returns 6-DOF vector [r(3), t(3)]."""
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    points   = numpy.atleast_2d(numpy.array(points,   dtype=numpy.float64))
    try:
        success, rvec, tvec = cv2.solvePnP(
            points.astype(numpy.float32),
            bearings[:, :2].astype(numpy.float32),
            numpy.eye(3, dtype=numpy.float32),
            None,
        )
        if success:
            return numpy.hstack([rvec.ravel(), tvec.ravel()])
    except Exception:
        pass
    return numpy.zeros(6)


def absolute_pose_n_points_known_rotation(
    bearings:    numpy.ndarray,
    points:      numpy.ndarray,
) -> numpy.ndarray:
    """Estimate translation given known rotation (least squares)."""
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    points   = numpy.atleast_2d(numpy.array(points,   dtype=numpy.float64))
    # Solve A*t = b  where A encodes cross-product constraints
    n = len(bearings)
    A = numpy.zeros((2 * n, 3), dtype=numpy.float64)
    b = numpy.zeros(2 * n,      dtype=numpy.float64)
    for i, (bear, pt) in enumerate(zip(bearings, points)):
        d = bear / numpy.linalg.norm(bear)
        A[2*i,   :] = [1, 0, -d[0]/d[2]] if abs(d[2]) > 1e-8 else [1, 0, 0]
        A[2*i+1, :] = [0, 1, -d[1]/d[2]] if abs(d[2]) > 1e-8 else [0, 1, 0]
        b[2*i]   = pt[0] - d[0]/d[2]*pt[2] if abs(d[2]) > 1e-8 else pt[0]
        b[2*i+1] = pt[1] - d[1]/d[2]*pt[2] if abs(d[2]) > 1e-8 else pt[1]
    try:
        t, _, _, _ = numpy.linalg.lstsq(A, b, rcond=None)
        return t
    except Exception:
        return numpy.zeros(3)


def point_refinement(
    origins:    numpy.ndarray,
    bearings:   numpy.ndarray,
    point:      numpy.ndarray,
    iterations: int = 10,          # ← thêm arg thứ 4
) -> numpy.ndarray:
    """Refine triangulated point using iterative midpoint refinement."""
    origins  = numpy.atleast_2d(numpy.array(origins,  dtype=numpy.float64))
    bearings = numpy.atleast_2d(numpy.array(bearings, dtype=numpy.float64))
    X        = numpy.array(point, dtype=numpy.float64)

    for _ in range(int(iterations)):
        A = numpy.zeros((3, 3), dtype=numpy.float64)
        b = numpy.zeros(3,      dtype=numpy.float64)
        for o, d in zip(origins, bearings):
            d = d / (numpy.linalg.norm(d) + 1e-10)
            M = numpy.eye(3) - numpy.outer(d, d)
            A += M
            b += M @ o
        try:
            X = numpy.linalg.solve(A, b)
        except numpy.linalg.LinAlgError:
            break

    return X


def epipolar_angle_two_bearings_many(
    b1: numpy.ndarray,
    b2: numpy.ndarray,
    R:  numpy.ndarray,
    t:  numpy.ndarray,
) -> numpy.ndarray:
    """Compute epipolar angles between two sets of bearing vectors."""
    b1 = numpy.atleast_2d(numpy.array(b1, dtype=numpy.float64))
    b2 = numpy.atleast_2d(numpy.array(b2, dtype=numpy.float64))
    R  = numpy.array(R, dtype=numpy.float64)
    t  = numpy.array(t, dtype=numpy.float64).ravel()

    t_hat  = t / (numpy.linalg.norm(t) + 1e-10)
    b2_rot = (R @ b2.T).T
    cross  = numpy.cross(b1, b2_rot)
    norms  = numpy.linalg.norm(cross, axis=1, keepdims=True)
    normal = cross / (norms + 1e-10)
    angles = numpy.abs(numpy.arcsin(
        numpy.clip(normal @ t_hat, -1.0, 1.0)
    ))
    return angles


def compute_camera_mapping(
    src_camera: Camera,
    dst_camera: Camera,
    width:  int,
    height: int,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Compute pixel mapping between two camera models."""
    ys, xs = numpy.meshgrid(numpy.arange(height), numpy.arange(width), indexing='ij')
    coords  = numpy.stack([xs.ravel(), ys.ravel()], axis=1).astype(numpy.float64)
    bearings = numpy.array([src_camera.pixel_bearing(p) for p in coords])
    dst_px   = numpy.array([dst_camera.project(b) for b in bearings])
    map_x = dst_px[:, 0].reshape(height, width).astype(numpy.float32)
    map_y = dst_px[:, 1].reshape(height, width).astype(numpy.float32)
    return map_x, map_y


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BROWN         = ProjectionType.BROWN
DUAL          = ProjectionType.DUAL
FISHEYE       = ProjectionType.FISHEYE
FISHEYE62     = ProjectionType.FISHEYE62
FISHEYE624    = ProjectionType.FISHEYE624
FISHEYE_OPENCV = ProjectionType.FISHEYE_OPENCV
PERSPECTIVE   = ProjectionType.PERSPECTIVE
RADIAL        = ProjectionType.RADIAL
SIMPLE_RADIAL = ProjectionType.SIMPLE_RADIAL
SPHERICAL     = ProjectionType.SPHERICAL

aspect_ratio = CameraParameters()
cx           = CameraParameters()
cy           = CameraParameters()
focal        = CameraParameters()
k1           = CameraParameters()
k2           = CameraParameters()
k3           = CameraParameters()
k4           = CameraParameters()
k5           = CameraParameters()
k6           = CameraParameters()
none         = CameraParameters()
p1           = CameraParameters()
p2           = CameraParameters()
s0           = CameraParameters()
s1           = CameraParameters()
s2           = CameraParameters()
s3           = CameraParameters()
transition   = CameraParameters()