# myOSfM/pydense.py (fake)
import numpy
from typing import Optional, List

__all__ = [
    "DepthmapCleaner",
    "DepthmapEstimator",
    "DepthmapPruner",
    "OpenMVSExporter",
    "StaticExtensionLoader",
]

class DepthmapCleaner:
    def __init__(self) -> None: pass
    def add_view(self, arg0: numpy.ndarray, arg1: numpy.ndarray, arg2: numpy.ndarray, arg3: numpy.ndarray) -> None: pass
    def clean(self) -> numpy.ndarray: return numpy.array([])
    def set_min_consistent_views(self, arg0: int) -> None: pass
    def set_same_depth_threshold(self, arg0: float) -> None: pass

class DepthmapEstimator:
    def __init__(self) -> None: pass
    def add_view(self, arg0: numpy.ndarray, arg1: numpy.ndarray, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray) -> None: pass
    def compute_brute_force(self) -> List: return []
    def compute_patch_match(self) -> List: return []
    def compute_patch_match_sample(self) -> List: return []
    def set_depth_range(self, arg0: float, arg1: float, arg2: int) -> None: pass
    def set_min_patch_sd(self, arg0: float) -> None: pass
    def set_patch_size(self, arg0: int) -> None: pass
    def set_patchmatch_iterations(self, arg0: int) -> None: pass

class DepthmapPruner:
    def __init__(self) -> None: pass
    def add_view(self, arg0: numpy.ndarray, arg1: numpy.ndarray, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: numpy.ndarray, arg5: numpy.ndarray, arg6: numpy.ndarray) -> None: pass
    def prune(self) -> List: return []
    def set_same_depth_threshold(self, arg0: float) -> None: pass

class OpenMVSExporter:
    def __init__(self) -> None: pass
    def add_camera(self, arg0: str, arg1: numpy.ndarray, arg2: int, arg3: int) -> None: pass
    def add_point(self, arg0: numpy.ndarray, arg1: List) -> None: pass
    def add_shot(self, arg0: str, arg1: str, arg2: str, arg3: str, arg4: numpy.ndarray, arg5: numpy.ndarray) -> None: pass
    def export(self, arg0: str) -> None: pass

class StaticExtensionLoader:
    pass