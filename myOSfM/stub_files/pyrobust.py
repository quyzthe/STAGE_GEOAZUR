"""
Pure Python implementation of pyrobust module.
Replaces the C++ pybind11 bindings from pyrobust.cc / instanciations.cc.

Implements RANSAC, MSAC, LMedS robust estimators for:
  - Essential matrix
  - Relative pose
  - Relative rotation
  - Absolute pose
  - Absolute pose (known rotation)
  - Similarity
  - Line fitting
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional
import cv2

__all__ = [
    "RansacType",
    "RobustEstimatorParams",
    "ScoreInfoLine",
    "ScoreInfoMatrix34d",
    "ScoreInfoMatrix3d",
    "ScoreInfoMatrix4d",
    "ScoreInfoVector3d",
    "ransac_absolute_pose",
    "ransac_absolute_pose_known_rotation",
    "ransac_essential",
    "ransac_line",
    "ransac_relative_pose",
    "ransac_relative_rotation",
    "ransac_similarity",
    "LMedS",
    "MSAC",
    "RANSAC",
]


# ---------------------------------------------------------------------------
# RansacType enum
# ---------------------------------------------------------------------------

class RansacType:
    RANSAC = None
    MSAC   = None
    LMedS  = None
    __members__ = {}

    def __init__(self, name: str, value: int):
        self.name  = name
        self.value = value

    def __repr__(self):
        return f"RansacType.{self.name}"

    def __eq__(self, other):
        if isinstance(other, RansacType):
            return self.value == other.value
        return NotImplemented

    def __hash__(self):
        return hash(self.value)

RansacType.RANSAC = RansacType("RANSAC", 0)
RansacType.MSAC   = RansacType("MSAC",   1)
RansacType.LMedS  = RansacType("LMedS",  2)
RansacType.__members__ = {
    "RANSAC": RansacType.RANSAC,
    "MSAC":   RansacType.MSAC,
    "LMedS":  RansacType.LMedS,
}

# module-level aliases (export_values)
RANSAC = RansacType.RANSAC
MSAC   = RansacType.MSAC
LMedS  = RansacType.LMedS


# ---------------------------------------------------------------------------
# RobustEstimatorParams
# ---------------------------------------------------------------------------

class RobustEstimatorParams:
    def __init__(self):
        self.iterations             = 1000
        self.probability            = 0.999
        self.use_local_optimization = True
        self.use_iteration_reduction = True

    def __repr__(self):
        return (f"RobustEstimatorParams(iterations={self.iterations}, "
                f"probability={self.probability})")


# ---------------------------------------------------------------------------
# ScoreInfo containers  (mirrors ScoreInfo<T> template)
# ---------------------------------------------------------------------------

class _ScoreInfo:
    def __init__(self):
        self.score:           float      = 0.0
        self.model:           np.ndarray = np.zeros((3, 3))
        self.lo_model:        np.ndarray = np.zeros((3, 3))
        self.inliers_indices: List[int]  = []

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"score={self.score:.4f}, "
                f"inliers={len(self.inliers_indices)})")


class ScoreInfoLine(_ScoreInfo):
    """ScoreInfo<Line::Type>  — model shape (2,)"""
    def __init__(self):
        super().__init__()
        self.model    = np.zeros(2)
        self.lo_model = np.zeros(2)

class ScoreInfoMatrix3d(_ScoreInfo):
    """ScoreInfo<Eigen::Matrix3d>  — model shape (3,3)"""
    def __init__(self):
        super().__init__()
        self.model    = np.zeros((3, 3))
        self.lo_model = np.zeros((3, 3))

class ScoreInfoMatrix4d(_ScoreInfo):
    """ScoreInfo<Eigen::Matrix4d>  — model shape (4,4)"""
    def __init__(self):
        super().__init__()
        self.model    = np.zeros((4, 4))
        self.lo_model = np.zeros((4, 4))

class ScoreInfoMatrix34d(_ScoreInfo):
    """ScoreInfo<Eigen::Matrix<double,3,4>>  — model shape (3,4)"""
    def __init__(self):
        super().__init__()
        self.model    = np.zeros((3, 4))
        self.lo_model = np.zeros((3, 4))

class ScoreInfoVector3d(_ScoreInfo):
    """ScoreInfo<Eigen::Vector3d>  — model shape (3,)"""
    def __init__(self):
        super().__init__()
        self.model    = np.zeros(3)
        self.lo_model = np.zeros(3)


# ---------------------------------------------------------------------------
# Internal RANSAC engine
# ---------------------------------------------------------------------------

def _ransac_score(residuals: np.ndarray,
                  threshold: float,
                  ransac_type: RansacType) -> Tuple[float, List[int]]:
    """Compute score and inlier list given residuals."""
    if ransac_type == RansacType.RANSAC:
        inliers = np.where(residuals < threshold)[0].tolist()
        score   = float(len(inliers))
    elif ransac_type == RansacType.MSAC:
        # truncated squared residuals
        sq = residuals ** 2
        thr_sq = threshold ** 2
        scores = np.where(sq < thr_sq, sq, thr_sq)
        score  = float(-np.sum(scores))        # higher = better → negate
        inliers = np.where(residuals < threshold)[0].tolist()
    else:  # LMedS
        inliers = np.where(residuals < threshold)[0].tolist()
        score   = float(len(inliers))
    return score, inliers


def _ransac(
    samples,                    # list of data items
    min_sample_size: int,
    fit_fn,                     # fn(subset) -> model or None
    residual_fn,                # fn(model, samples) -> residuals array
    threshold: float,
    params: RobustEstimatorParams,
    ransac_type: RansacType,
) -> Tuple[Optional[object], List[int], float]:
    """Generic RANSAC loop. Returns (best_model, inlier_indices, score)."""
    n = len(samples)
    if n < min_sample_size:
        return None, [], 0.0

    rng         = np.random.default_rng(42)
    best_model  = None
    best_score  = -1e18
    best_inliers: List[int] = []

    # adaptive iteration count
    max_iter = params.iterations
    iter_count = 0

    while iter_count < max_iter:
        idx    = rng.choice(n, min_sample_size, replace=False)
        subset = [samples[i] for i in idx]
        model  = fit_fn(subset)
        if model is None:
            iter_count += 1
            continue

        residuals        = residual_fn(model, samples)
        score, inliers   = _ransac_score(residuals, threshold, ransac_type)

        if score > best_score:
            best_score   = score
            best_model   = model
            best_inliers = inliers

            # update max iterations (standard formula)
            if params.use_iteration_reduction and len(inliers) > 0:
                w = len(inliers) / n
                w = max(w, 1e-9)
                denom = np.log(1.0 - w ** min_sample_size)
                if denom < -1e-10:
                    new_max = int(
                        np.log(1.0 - params.probability) / denom
                    )
                    max_iter = min(max_iter, max(new_max, min_sample_size))

        iter_count += 1

    # local optimisation: refit on all inliers
    if best_inliers and params.use_local_optimization:
        subset    = [samples[i] for i in best_inliers]
        lo_model  = fit_fn(subset) if len(subset) >= min_sample_size else None
        if lo_model is not None:
            residuals      = residual_fn(lo_model, samples)
            lo_score, lo_in = _ransac_score(residuals, threshold, ransac_type)
            if lo_score >= best_score:
                best_model   = lo_model
                best_score   = lo_score
                best_inliers = lo_in

    return best_model, best_inliers, best_score


# ---------------------------------------------------------------------------
# ransac_essential
# ---------------------------------------------------------------------------

def ransac_essential(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    params: RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoMatrix3d:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    result = ScoreInfoMatrix3d()

    if len(x1) < 5:
        return result

    samples = list(zip(x1, x2))

    # def fit(subset):
    #     s1 = np.array([s[0] for s in subset])
    #     s2 = np.array([s[1] for s in subset])
    #     E, mask = cv2.findEssentialMat(
    #         s1[:, :2], s2[:, :2],
    #         focal=1.0, pp=(0.0, 0.0),
    #         method=cv2.RANSAC, prob=0.999,
    #         threshold=threshold,
    #     )
    #     # TRƯỚC - trong fit của ransac_essential:
    #     return E if E is not None else None

    def fit(subset):
        s1 = np.array([s[0][:2] for s in subset])
        s2 = np.array([s[1][:2] for s in subset])
        try:
            E, _ = cv2.findEssentialMat(
                s1, s2, focal=1.0, pp=(0.0, 0.0),
                method=cv2.RANSAC, prob=0.999, threshold=threshold,
            )
            if E is None or E.size < 9:
                return None
            return E[:3, :3].copy()
        except (cv2.error, Exception):
            return None


    def residual(model, samps):
        pts1 = np.array([s[0][:2] for s in samps])
        pts2 = np.array([s[1][:2] for s in samps])
        F    = model
        ones = np.ones((len(pts1), 1))
        p1h  = np.hstack([pts1, ones])
        p2h  = np.hstack([pts2, ones])
        # Sampson distance
        Fp1  = (F @ p1h.T).T
        Ftp2 = (F.T @ p2h.T).T
        num  = np.sum(p2h * Fp1, axis=1) ** 2
        denom = Fp1[:, 0]**2 + Fp1[:, 1]**2 + Ftp2[:, 0]**2 + Ftp2[:, 1]**2
        denom = np.maximum(denom, 1e-10)
        return np.sqrt(np.abs(num / denom))

    model, inliers, score = _ransac(
        samples, 5, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model          = model
        result.lo_model       = model
        result.inliers_indices = inliers
        result.score          = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_relative_pose
# ---------------------------------------------------------------------------

def ransac_relative_pose(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    params: RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoMatrix34d:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    result = ScoreInfoMatrix34d()

    if len(x1) < 5:
        return result

    samples = list(zip(x1, x2))

    # # TRƯỚC - trong hàm fit của ransac_relative_pose:
    # def fit(subset):
    #     s1 = np.array([s[0][:2] for s in subset])
    #     s2 = np.array([s[1][:2] for s in subset])
    #     E, _ = cv2.findEssentialMat(
    #         s1, s2, focal=1.0, pp=(0.0, 0.0),
    #         method=cv2.RANSAC, prob=0.999, threshold=threshold,
    #     )
    #     if E is None:
    #         return None
    #     _, R, t, _ = cv2.recoverPose(
    #         E, s1, s2, focal=1.0, pp=(0.0, 0.0)
    #     )
    #     Rt = np.hstack([R, t])
    #     return Rt

    # SAU:
    def fit(subset):
        s1 = np.array([s[0][:2] for s in subset])
        s2 = np.array([s[1][:2] for s in subset])
        try:
            E, _ = cv2.findEssentialMat(
                s1, s2, focal=1.0, pp=(0.0, 0.0),
                method=cv2.RANSAC, prob=0.999, threshold=threshold,
            )
            if E is None or E.size < 9:
                return None
            E = E[:3, :3].copy()
            _, R, t, _ = cv2.recoverPose(E, s1, s2, focal=1.0, pp=(0.0, 0.0))
            return np.hstack([R, t])
        except (cv2.error, Exception):
            return None

    def residual(model, samps):
        R  = model[:, :3]
        t  = model[:, 3]
        res = []
        for s1, s2 in samps:
            b1 = s1 / (np.linalg.norm(s1) + 1e-10)
            b2 = s2 / (np.linalg.norm(s2) + 1e-10)
            b2_rot = R.T @ (b2 - t)
            b2_rot = b2_rot / (np.linalg.norm(b2_rot) + 1e-10)
            angle = np.arccos(np.clip(np.dot(b1, b2_rot), -1.0, 1.0))
            res.append(angle)
        return np.array(res)

    model, inliers, score = _ransac(
        samples, 5, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model
        result.lo_model        = model
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_relative_rotation
# ---------------------------------------------------------------------------

def ransac_relative_rotation(
    x1: np.ndarray,
    x2: np.ndarray,
    threshold: float,
    params: RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoMatrix3d:
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    result = ScoreInfoMatrix3d()

    if len(x1) < 2:
        return result

    samples = list(zip(x1, x2))

    def fit(subset):
        s1 = np.array([s[0] for s in subset], dtype=np.float64)
        s2 = np.array([s[1] for s in subset], dtype=np.float64)
        # normalise
        s1 = (s1.T / (np.linalg.norm(s1, axis=1) + 1e-10)).T
        s2 = (s2.T / (np.linalg.norm(s2, axis=1) + 1e-10)).T
        # Kabsch / SVD rotation
        H = s1.T @ s2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        return R

    def residual(model, samps):
        res = []
        for s1, s2 in samps:
            b1     = s1 / (np.linalg.norm(s1) + 1e-10)
            b2_rot = model @ (s2 / (np.linalg.norm(s2) + 1e-10))
            angle  = np.arccos(np.clip(np.dot(b1, b2_rot), -1.0, 1.0))
            res.append(angle)
        return np.array(res)

    model, inliers, score = _ransac(
        samples, 2, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model
        result.lo_model        = model
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_absolute_pose
# ---------------------------------------------------------------------------

def ransac_absolute_pose(
    bearings: np.ndarray,
    points:   np.ndarray,
    threshold: float,
    params:   RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoMatrix34d:
    bearings = np.asarray(bearings, dtype=np.float64)
    points   = np.asarray(points,   dtype=np.float64)
    result   = ScoreInfoMatrix34d()

    if len(bearings) < 4:
        return result

    samples = list(zip(bearings, points))

    def fit(subset):
        b = np.array([s[0] for s in subset], dtype=np.float32)
        p = np.array([s[1] for s in subset], dtype=np.float32)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                p, b[:, :2],
                np.eye(3, dtype=np.float32), None,
                flags=cv2.SOLVEPNP_P3P if len(subset) == 3 else cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                return None
            R, _ = cv2.Rodrigues(rvec)
            return np.hstack([R, tvec])
        except Exception:
            return None

    def residual(model, samps):
        R = model[:, :3]
        t = model[:, 3:4]
        res = []
        for bear, pt in samps:
            pt3 = pt.reshape(3, 1)
            proj = R @ pt3 + t
            if abs(proj[2, 0]) < 1e-10:
                res.append(1e6)
                continue
            bear_proj = proj.ravel() / np.linalg.norm(proj)
            bear_norm = bear / (np.linalg.norm(bear) + 1e-10)
            angle = np.arccos(np.clip(np.dot(bear_norm, bear_proj), -1.0, 1.0))
            res.append(angle)
        return np.array(res)

    model, inliers, score = _ransac(
        samples, 4, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model
        result.lo_model        = model
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_absolute_pose_known_rotation
# ---------------------------------------------------------------------------

def ransac_absolute_pose_known_rotation(
    bearings: np.ndarray,
    points:   np.ndarray,
    threshold: float,
    params:   RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoVector3d:
    bearings = np.asarray(bearings, dtype=np.float64)
    points   = np.asarray(points,   dtype=np.float64)
    result   = ScoreInfoVector3d()

    if len(bearings) < 2:
        return result

    samples = list(zip(bearings, points))

    def fit(subset):
        # Least-squares translation given unit bearing directions
        n = len(subset)
        A = np.zeros((2 * n, 3))
        b = np.zeros(2 * n)
        for i, (bear, pt) in enumerate(subset):
            d = bear / (np.linalg.norm(bear) + 1e-10)
            if abs(d[2]) > 1e-8:
                A[2*i,   :] = [1, 0, -d[0]/d[2]]
                A[2*i+1, :] = [0, 1, -d[1]/d[2]]
                b[2*i]      = pt[0] - d[0]/d[2]*pt[2]
                b[2*i+1]    = pt[1] - d[1]/d[2]*pt[2]
            else:
                A[2*i,   :] = [1, 0, 0]
                A[2*i+1, :] = [0, 1, 0]
                b[2*i]      = pt[0]
                b[2*i+1]    = pt[1]
        try:
            t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return t
        except Exception:
            return None

    def residual(model, samps):
        t = model
        res = []
        for bear, pt in samps:
            diff = pt - t
            bear_proj = diff / (np.linalg.norm(diff) + 1e-10)
            bear_norm = bear / (np.linalg.norm(bear) + 1e-10)
            angle = np.arccos(np.clip(np.dot(bear_norm, bear_proj), -1.0, 1.0))
            res.append(angle)
        return np.array(res)

    model, inliers, score = _ransac(
        samples, 2, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model
        result.lo_model        = model
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_similarity  (3-D point cloud alignment)
# ---------------------------------------------------------------------------

def ransac_similarity(
    points1: np.ndarray,
    points2: np.ndarray,
    threshold: float,
    params:   RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoMatrix4d:
    points1 = np.asarray(points1, dtype=np.float64)
    points2 = np.asarray(points2, dtype=np.float64)
    result  = ScoreInfoMatrix4d()

    if len(points1) < 3:
        return result

    samples = list(zip(points1, points2))

    def fit(subset):
        p1 = np.array([s[0] for s in subset])
        p2 = np.array([s[1] for s in subset])
        # Umeyama similarity transform
        n  = len(p1)
        mu1 = p1.mean(axis=0)
        mu2 = p2.mean(axis=0)
        c1  = p1 - mu1
        c2  = p2 - mu2
        var1 = np.mean(np.sum(c1**2, axis=1))
        if var1 < 1e-10:
            return None
        H   = c1.T @ c2 / n
        U, S, Vt = np.linalg.svd(H)
        d   = np.linalg.det(Vt.T @ U.T)
        D   = np.diag([1, 1, np.sign(d)])
        R   = Vt.T @ D @ U.T
        s   = np.sum(S * np.diag(D)) / var1
        t   = mu2 - s * R @ mu1
        # pack into (4,4) homogeneous matrix
        M       = np.eye(4)
        M[:3, :3] = s * R
        M[:3,  3] = t
        return M

    def residual(model, samps):
        sR = model[:3, :3]
        t  = model[:3,  3]
        res = []
        for p1, p2 in samps:
            p1_t = sR @ p1 + t
            res.append(np.linalg.norm(p1_t - p2))
        return np.array(res)

    model, inliers, score = _ransac(
        samples, 3, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model
        result.lo_model        = model
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result


# ---------------------------------------------------------------------------
# ransac_line  (2-D line fitting)
# ---------------------------------------------------------------------------

def ransac_line(
    points:   np.ndarray,
    threshold: float,
    params:   RobustEstimatorParams,
    ransac_type: RansacType,
) -> ScoreInfoLine:
    points = np.asarray(points, dtype=np.float64)
    result = ScoreInfoLine()

    if len(points) < 2:
        return result

    samples = list(points)

    def fit(subset):
        p1 = subset[0]
        p2 = subset[1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm = np.sqrt(dx*dx + dy*dy)
        if norm < 1e-10:
            return None
        # line params: (a, b) where ax + by = 1  (homogeneous)
        a =  dy / norm
        b = -dx / norm
        c = a * p1[0] + b * p1[1]
        return np.array([a, b, c])   # ax + by = c

    def residual(model, samps):
        a, b, c = model
        pts = np.array(samps)
        dist = np.abs(pts[:, 0] * a + pts[:, 1] * b - c)
        return dist

    model, inliers, score = _ransac(
        samples, 2, fit, residual, threshold, params, ransac_type
    )

    if model is not None:
        result.model           = model[:2]   # keep (a, b) as Line::Type
        result.lo_model        = model[:2]
        result.inliers_indices = inliers
        result.score           = float(len(inliers))
    return result