# myOSfM/pyfeatures.py (fake)
import numpy
from typing import Dict, List, Set, Tuple

__all__  = [
    "AKAZEOptions",
    "AkazeDescriptorType",
    "AkazeDiffusivityType",
    "akaze",
    "compute_vlad_descriptor",
    "compute_vlad_distances",
    "hahog",
    "match_using_words"
]

class AKAZEOptions:
    def __init__(self) -> None: pass
    @property
    def derivative_factor(self) -> float: return 0.0
    @derivative_factor.setter
    def derivative_factor(self, arg0: float) -> None: pass
    @property
    def descriptor(self): return None
    @descriptor.setter
    def descriptor(self, arg0) -> None: pass
    @property
    def descriptor_channels(self) -> int: return 0
    @descriptor_channels.setter
    def descriptor_channels(self, arg0: int) -> None: pass
    @property
    def descriptor_pattern_size(self) -> int: return 0
    @descriptor_pattern_size.setter
    def descriptor_pattern_size(self, arg0: int) -> None: pass
    @property
    def descriptor_size(self) -> int: return 0
    @descriptor_size.setter
    def descriptor_size(self, arg0: int) -> None: pass
    @property
    def diffusivity(self): return None
    @diffusivity.setter
    def diffusivity(self, arg0) -> None: pass
    @property
    def dthreshold(self) -> float: return 0.0
    @dthreshold.setter
    def dthreshold(self, arg0: float) -> None: pass
    @property
    def img_height(self) -> int: return 0
    @img_height.setter
    def img_height(self, arg0: int) -> None: pass
    @property
    def img_width(self) -> int: return 0
    @img_width.setter
    def img_width(self, arg0: int) -> None: pass
    @property
    def kcontrast(self) -> float: return 0.0
    @kcontrast.setter
    def kcontrast(self, arg0: float) -> None: pass
    @property
    def kcontrast_nbins(self) -> int: return 0
    @kcontrast_nbins.setter
    def kcontrast_nbins(self, arg0: int) -> None: pass
    @property
    def kcontrast_percentile(self) -> float: return 0.0
    @kcontrast_percentile.setter
    def kcontrast_percentile(self, arg0: float) -> None: pass
    @property
    def min_dthreshold(self) -> float: return 0.0
    @min_dthreshold.setter
    def min_dthreshold(self, arg0: float) -> None: pass
    @property
    def nsublevels(self) -> int: return 0
    @nsublevels.setter
    def nsublevels(self, arg0: int) -> None: pass
    @property
    def omax(self) -> int: return 0
    @omax.setter
    def omax(self, arg0: int) -> None: pass
    @property
    def omin(self) -> int: return 0
    @omin.setter
    def omin(self, arg0: int) -> None: pass
    @property
    def save_keypoints(self) -> bool: return False
    @save_keypoints.setter
    def save_keypoints(self, arg0: bool) -> None: pass
    @property
    def save_scale_space(self) -> bool: return False
    @save_scale_space.setter
    def save_scale_space(self, arg0: bool) -> None: pass
    @property
    def sderivatives(self) -> float: return 0.0
    @sderivatives.setter
    def sderivatives(self, arg0: float) -> None: pass
    @property
    def soffset(self) -> float: return 0.0
    @soffset.setter
    def soffset(self, arg0: float) -> None: pass
    @property
    def target_num_features(self) -> int: return 0
    @target_num_features.setter
    def target_num_features(self, arg0: int) -> None: pass
    @property
    def use_adaptive_suppression(self) -> bool: return False
    @use_adaptive_suppression.setter
    def use_adaptive_suppression(self, arg0: bool) -> None: pass
    @property
    def use_isotropic_diffusion(self) -> bool: return False
    @use_isotropic_diffusion.setter
    def use_isotropic_diffusion(self, arg0: bool) -> None: pass
    @property
    def verbosity(self) -> bool: return False
    @verbosity.setter
    def verbosity(self, arg0: bool) -> None: pass

# class AkazeDescriptorType:
#     SURF_UPRIGHT = None
#     SURF = None
#     MSURF_UPRIGHT = None
#     MSURF = None
#     MLDB_UPRIGHT = None
#     MLDB = None
#     __members__ = {}
#     @property
#     def name(self) -> str: return ""

class AkazeDescriptorType:
    def __init__(self, name: str):
        self._name = name
    @property
    def name(self) -> str:
        return self._name

# tạo các constant
AkazeDescriptorType.SURF_UPRIGHT = AkazeDescriptorType("SURF_UPRIGHT")
AkazeDescriptorType.SURF = AkazeDescriptorType("SURF")
AkazeDescriptorType.MSURF_UPRIGHT = AkazeDescriptorType("MSURF_UPRIGHT")
AkazeDescriptorType.MSURF = AkazeDescriptorType("MSURF")
AkazeDescriptorType.MLDB_UPRIGHT = AkazeDescriptorType("MLDB_UPRIGHT")
AkazeDescriptorType.MLDB = AkazeDescriptorType("MLDB")

# tạo dict members
AkazeDescriptorType.__members__ = {
    "SURF_UPRIGHT": AkazeDescriptorType.SURF_UPRIGHT,
    "SURF": AkazeDescriptorType.SURF,
    "MSURF_UPRIGHT": AkazeDescriptorType.MSURF_UPRIGHT,
    "MSURF": AkazeDescriptorType.MSURF,
    "MLDB_UPRIGHT": AkazeDescriptorType.MLDB_UPRIGHT,
    "MLDB": AkazeDescriptorType.MLDB,
}

class AkazeDiffusivityType:
    PM_G1 = None
    PM_G2 = None
    WEICKERT = None
    CHARBONNIER = None
    __members__ = {}
    @property
    def name(self) -> str: return ""

def akaze(arg0: numpy.ndarray, arg1: AKAZEOptions) -> tuple: return (), ()
def compute_vlad_descriptor(arg0: numpy.ndarray, arg1: numpy.ndarray) -> numpy.ndarray: return numpy.array([])
def compute_vlad_distances(arg0: Dict[str, numpy.ndarray], arg1: str, arg2: Set[str]) -> Tuple[List[float], List[str]]: return [], []
def hahog(image: numpy.ndarray, peak_threshold: float = 0.003, edge_threshold: float = 10, target_num_features: int = 0) -> tuple: return (), ()
def match_using_words(arg0: numpy.ndarray, arg1: numpy.ndarray, arg2: numpy.ndarray, arg3: numpy.ndarray, arg4: float = 0.0, arg5: int = 0) -> numpy.ndarray: return numpy.array([])