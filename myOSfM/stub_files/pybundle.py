# myOSfM/pybundle.py (fake)
import numpy
from typing import Optional

# Fake classes từ stub .pyi
class BundleAdjuster:
    def __init__(self) -> None: pass
    def add_absolute_pan(self, arg0: str, arg1: float, arg2: float) -> None: pass
    def add_absolute_position_heatmap(self, arg0: str, arg1: str, arg2: float, arg3: float, arg4: float) -> None: pass
    def add_absolute_roll(self, arg0: str, arg1: float, arg2: float) -> None: pass
    def add_absolute_tilt(self, arg0: str, arg1: float, arg2: float) -> None: pass
    def add_absolute_up_vector(self, arg0: str, arg1: numpy.ndarray, arg2: float) -> None: pass
    def add_camera(self, arg0: str, arg1=None, arg2=None, arg3: bool=False) -> None: pass
    def add_common_position(self, arg0: str, arg1: str, arg2: float, arg3: float) -> None: pass
    def add_heatmap(self, arg0: str, arg1: list[float], arg2: int, arg3: float) -> None: pass
    def add_linear_motion(self, *args, **kwargs) -> None: pass
    def add_point(self, arg0: str, arg1: numpy.ndarray, arg2: bool) -> None: pass
    def add_point_prior(self, *args, **kwargs) -> None: pass
    def add_point_projection_observation(self, *args, **kwargs) -> None: pass
    def add_reconstruction(self, arg0: str, arg1: bool) -> None: pass
    def add_reconstruction_instance(self, arg0: str, arg1: float, arg2: str) -> None: pass
    def add_relative_motion(self, arg0) -> None: pass
    def add_relative_rotation(self, arg0) -> None: pass
    def add_rig_camera(self, *args, **kwargs) -> None: pass
    def add_rig_instance(self, *args, **kwargs) -> None: pass
    def add_rig_instance_position_prior(self, *args, **kwargs) -> None: pass
    def brief_report(self) -> str: return ""
    def full_report(self) -> str: return ""
    def get_camera(self, arg0: str): return None
    def get_covariance_estimation_valid(self) -> bool: return False
    def get_point(self, arg0: str): return None
    def get_reconstruction(self, arg0: str): return None
    def get_rig_camera_pose(self, arg0: str): return None
    def get_rig_instance_pose(self, arg0: str): return None
    def has_point(self, arg0: str) -> bool: return False
    def run(self) -> None: pass
    def set_adjust_absolute_position_std(self, arg0: bool) -> None: pass
    def set_compute_covariances(self, arg0: bool) -> None: pass
    def set_compute_reprojection_errors(self, arg0: bool) -> None: pass
    def set_gauge_fix_shots(self, arg0: str, arg1: str) -> None: pass
    def set_internal_parameters_prior_sd(self, *args, **kwargs) -> None: pass
    def set_linear_solver_type(self, arg0: str) -> None: pass
    def set_max_num_iterations(self, arg0: int) -> None: pass
    def set_num_threads(self, arg0: int) -> None: pass
    def set_point_projection_loss_function(self, arg0: str, arg1: float) -> None: pass
    def set_relative_motion_loss_function(self, arg0: str, arg1: float) -> None: pass
    def set_scale_sharing(self, arg0: str, arg1: bool) -> None: pass
    def set_use_analytic_derivatives(self, arg0: bool) -> None: pass

class Point:
    @property
    def id(self) -> str: return ""
    @property
    def p(self) -> numpy.ndarray: return numpy.array([])
    @property
    def reprojection_errors(self) -> dict: return {}
    @reprojection_errors.setter
    def reprojection_errors(self, arg0: dict) -> None: pass

class RAReconstruction:
    def __init__(self) -> None: pass
    @property
    def id(self) -> str: return ""
    @id.setter
    def id(self, arg0: str) -> None: pass
    @property
    def rx(self) -> float: return 0.0
    @rx.setter
    def rx(self, arg1: float) -> None: pass
    @property
    def ry(self) -> float: return 0.0
    @ry.setter
    def ry(self, arg1: float) -> None: pass
    @property
    def rz(self) -> float: return 0.0
    @rz.setter
    def rz(self, arg1: float) -> None: pass
    @property
    def scale(self) -> float: return 1.0
    @scale.setter
    def scale(self, arg1: float) -> None: pass
    @property
    def tx(self) -> float: return 0.0
    @tx.setter
    def tx(self, arg1: float) -> None: pass
    @property
    def ty(self) -> float: return 0.0
    @ty.setter
    def ty(self, arg1: float) -> None: pass
    @property
    def tz(self) -> float: return 0.0
    @tz.setter
    def tz(self, arg1: float) -> None: pass

class RARelativeMotionConstraint:
    def __init__(self, *args, **kwargs) -> None: pass
    @property
    def reconstruction(self) -> str: return ""
    @reconstruction.setter
    def reconstruction(self, arg0: str) -> None: pass
    @property
    def rx(self) -> float: return 0.0
    @rx.setter
    def rx(self, arg1: float) -> None: pass
    @property
    def ry(self) -> float: return 0.0
    @ry.setter
    def ry(self, arg1: float) -> None: pass
    @property
    def rz(self) -> float: return 0.0
    @rz.setter
    def rz(self, arg1: float) -> None: pass
    @property
    def shot(self) -> str: return ""
    @shot.setter
    def shot(self, arg0: str) -> None: pass
    @property
    def tx(self) -> float: return 0.0
    @tx.setter
    def tx(self, arg1: float) -> None: pass
    @property
    def ty(self) -> float: return 0.0
    @ty.setter
    def ty(self, arg1: float) -> None: pass
    @property
    def tz(self) -> float: return 0.0
    @tz.setter
    def tz(self, arg1: float) -> None: pass

class RAShot:
    def __init__(self) -> None: pass
    @property
    def id(self) -> str: return ""
    @id.setter
    def id(self, arg0: str) -> None: pass
    @property
    def rx(self) -> float: return 0.0
    @rx.setter
    def rx(self, arg1: float) -> None: pass
    @property
    def ry(self) -> float: return 0.0
    @ry.setter
    def ry(self, arg1: float) -> None: pass
    @property
    def rz(self) -> float: return 0.0
    @rz.setter
    def rz(self, arg1: float) -> None: pass
    @property
    def tx(self) -> float: return 0.0
    @tx.setter
    def tx(self, arg1: float) -> None: pass
    @property
    def ty(self) -> float: return 0.0
    @ty.setter
    def ty(self, arg1: float) -> None: pass
    @property
    def tz(self) -> float: return 0.0
    @tz.setter
    def tz(self, arg1: float) -> None: pass

class Reconstruction:
    def __init__(self) -> None: pass
    def get_scale(self, arg0: str) -> float: return 1.0
    def set_scale(self, arg0: str, arg1: float) -> None: pass
    @property
    def id(self) -> str: return ""
    @id.setter
    def id(self, arg0: str) -> None: pass

class ReconstructionAlignment:
    def __init__(self) -> None: pass
    def add_absolute_position_constraint(self, *args, **kwargs) -> None: pass
    def add_common_camera_constraint(self, *args, **kwargs) -> None: pass
    def add_common_point_constraint(self, *args, **kwargs) -> None: pass
    def add_reconstruction(self, *args, **kwargs) -> None: pass
    def add_relative_absolute_position_constraint(self, *args, **kwargs) -> None: pass
    def add_relative_motion_constraint(self, *args, **kwargs) -> None: pass
    def add_shot(self, *args, **kwargs) -> None: pass
    def brief_report(self) -> str: return ""
    def full_report(self) -> str: return ""
    def get_reconstruction(self, arg0: str) -> RAReconstruction: return RAReconstruction()
    def get_shot(self, arg0: str) -> RAShot: return RAShot()

class RelativeMotion:
    def __init__(self, *args, **kwargs) -> None: pass
    @property
    def rig_instance_i(self) -> str: return ""
    @rig_instance_i.setter
    def rig_instance_i(self, arg0: str) -> None: pass
    @property
    def rig_instance_j(self) -> str: return ""
    @rig_instance_j.setter
    def rig_instance_j(self, arg0: str) -> None: pass

class RelativeRotation:
    def __init__(self, *args, **kwargs) -> None: pass
    @property
    def r(self) -> numpy.ndarray: return numpy.array([])
    @r.setter
    def r(self, arg1: numpy.ndarray) -> None: pass
    @property
    def shot_i(self) -> str: return ""
    @shot_i.setter
    def shot_i(self, arg0: str) -> None: pass
    @property
    def shot_j(self) -> str: return ""
    @shot_j.setter
    def shot_j(self, arg0: str) -> None: pass

class StaticExtensionLoader:
    pass